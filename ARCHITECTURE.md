# Architecture

_Last updated: 2026-08-28_

How Ophotofinder is put together. For setup and usage see [README.md](README.md);
for the pitfalls that shaped these choices see [DEVELOPMENT_LOG.md](DEVELOPMENT_LOG.md).

---

## The idea

A photo is hard to search because what you know about it lives in three unrelated
places: what it *looks like*, what it could be *described* as, and what is *recorded*
about it. Ophotofinder extracts all of them independently and never lets one become a
prerequisite for another. A photo with no caption is still findable by appearance; a
photo CLIP finds meaningless is still findable by the words printed in it.

## Indexing pipeline

```
                        ┌───────────────────────────────┐
   photo file  ────────▶│ open + ImageOps.exif_transpose│   ← once, before anything
                        └───────────────┬───────────────┘
                                        │  upright RGB image
          ┌────────────┬────────────────┼───────────────┬──────────────┐
          ▼            ▼                ▼               ▼              ▼
    ┌──────────┐ ┌───────────┐  ┌──────────────┐ ┌────────────┐ ┌───────────┐
    │   CLIP   │ │ Ollama VLM│  │  Tesseract   │ │ 2nd reader │ │   EXIF    │
    │ MPS/CUDA │ │  caption  │  │     OCR      │ │ Vision/    │ │  Pillow   │
    │  /CPU    │ │           │  │              │ │ RapidOCR   │ │           │
    └────┬─────┘ └─────┬─────┘  └──────┬───────┘ └─────┬──────┘ └─────┬─────┘
         │ 512-d       │ prose         │ text          │ text         │ date/camera/GPS
         │             └───────────────┴───────────────┴──────────────┘
         │                                    │
         │                          one text document per photo
         ▼                                    ▼ text embedding
  ┌───────────────┐                  ┌────────────────┐
  │Chroma          │                 │Chroma          │
  │photo_clip      │                 │photo_text      │
  └───────────────┘                  └────────────────┘
```

Both collections are keyed by `sha1(absolute path)`, so a hit in either resolves to
the same photo.

### The signals

| Signal | Produced by | Fails when | Consequence |
|---|---|---|---|
| CLIP image embedding | `clip_embed.py`, `openai/clip-vit-base-patch32` | never in practice | — |
| Caption | `caption.py` → Ollama vision model | model missing, broken, or too small for the prompt | recorded as `empty`/`error`, photo still indexed |
| OCR | `ocr.py` → Tesseract | binary absent, or no readable text | empty string, photo still indexed |
| Second text reader | `ocr_alt.py` → Apple Vision / RapidOCR / VLM | no backend installed | empty string, photo still indexed |
| EXIF | `imaging.py` → Pillow | tags absent | falls back to file mtime, recorded as `date_source` |

Two independent text readers exist because their failures are uncorrelated:
Tesseract is strong on flat document text and weak on real-world photographs and
non-Latin scripts; Apple Vision and RapidOCR are the reverse. Neither overwrites the
other — they occupy separate fields.

### Why `exif_transpose` happens first

`open_upright()` is the only way images enter the pipeline. Orientation is applied
once, at open, so CLIP, the VLM, both text readers and the thumbnailer all see the
same upright pixels. Doing it per-consumer would mean five chances to forget.

## Storage

```
$OPHOTOFINDER_HOME  (default ~/.ophotofinder)
├── chroma/            persistent Chroma client
│   ├── photo_clip     CLIP image vectors  + full metadata
│   └── photo_text     document vectors    + full metadata + document text
├── thumbs/            <sha1(path)>.jpg, long edge 512px, upright
├── settings.json      user defaults and watched folders
├── index.lock         PID lock, present only during a run
└── web.pid            present only while the web server runs

~/.ophotofinder-location    pointer to the data directory, if moved
```

The location pointer deliberately lives **outside** the data directory: `settings.json`
is stored inside it, so a location kept there would be lost the moment the directory
moved. `OPHOTOFINDER_HOME` overrides the pointer.

Metadata is duplicated into both collections so a hit from either side renders without
a second lookup. Chroma accepts only scalar metadata, so `None` and empty values are
dropped rather than stored (`_clean_meta`).

Key fields: `path`, `folder`, `filename`, `signature`, `caption`, `caption_status`,
`caption_detail`, `caption_model`, `ocr`, `has_ocr`, `ocr_alt`, `ocr_alt_method`,
`ocr_alt_conf`, `has_any_text`, `taken_at`, `date_source`, `year`, `month`, `camera`,
`lens`, `gps_lat`, `gps_lon`, `width`, `height`, `orientation`, `indexed_at`,
`pipeline`, and one `v_<component>` per pipeline component.

### Incremental re-indexing

`signature = sha1(resolved path + size + mtime)`. Existing signatures are loaded once
per run and unchanged files are skipped. `--reindex` bypasses the check.

### Resilience to a stale collection handle

`Store._call(which, method, ...)` catches Chroma's `NotFoundError`, re-opens both
collections **by name**, and retries once. It resolves the collection by attribute
name on each attempt, so the retry runs against the reopened handle rather than the
dead one. This is what keeps a long-running server alive when another process
recreates a collection.

## Embeddings

| Space | Model | Notes |
|---|---|---|
| Image + query text | CLIP ViT-B/32 via `transformers` | shared space; L2-normalised, cosine distance |
| Documents + query text | Chroma's bundled ONNX MiniLM | English-only; falls back to CLIP's text tower if unavailable |

Both are written with **explicit** embeddings — Chroma is never asked to embed
anything itself, so the two spaces cannot silently cross. Device selection is
`mps → cuda → cpu`, with a CPU fallback if a model refuses to move to MPS.

## Retrieval

Three rankings are produced independently and fused with reciprocal rank fusion:

```
score(photo) = Σ  1 / (60 + rank_in_that_result_list)
```

1. **CLIP** — the query through CLIP's text encoder against `photo_clip`.
2. **Text embedding** — the query through MiniLM against `photo_text`.
3. **Literal match** — CJK runs and Latin words from the query, matched verbatim via
   Chroma's `$contains`, ranked by how many terms hit.

Distances are deliberately *not* combined: the vector spaces are unrelated, so their
distances share no scale, but their orderings do. RRF also degrades cleanly when a
photo appears in only one list — exactly the case for an uncaptioned photo that CLIP
found.

The literal signal is not a nicety. The sentence embedder is English-only: measured on
real photos, *unrelated* Chinese text scored **higher** (0.573) than *related* Chinese
text (0.502), so semantic ranking over CJK is noise. Exact substring matching is
script-agnostic and repairs this completely; it also beats embeddings on names,
numbers and codes.

Each hit records `clip_rank`, `text_rank` and `literal_rank`, surfaced as
`visual #1 + text #3 + exact text ×2`, so it is always visible *why* something
matched. EXIF filters are pushed down into both Chroma queries as a `where` clause
rather than applied after fusion, so filtering never starves the result pool.

### The generation step

`--answer` is the only part that is generation rather than retrieval. The top eight
hits' captions, text and EXIF are formatted into a numbered context block and sent to
a local text model, instructed to cite `[n]` and to say plainly when the records do
not answer the question. It never reorders or filters the photo results, and the
model never sees pixels — only text extracted at index time. Retrieval is fully
usable without it.

## Pipeline versioning

Every signal is produced by a component carrying its own version
(`config.PIPELINE_VERSIONS`), and each record stores the versions it was built with
plus a `pipeline` signature string.

This exists because recomputation costs are wildly uneven — re-reading EXIF is nearly
free, re-captioning with a VLM dominates everything else. A single "index version"
would force a full re-caption for an EXIF bug fix.

`indexer.plan_upgrade()` reports which records are behind, per component.
`indexer.upgrade_index()` then, for each stale record: opens the image once, runs
**only** the stale components, reuses stored values for the rest, rebuilds the
document if any of its inputs moved, re-embeds the text (cheap), and re-embeds the
CLIP vector only if `clip` itself is stale.

Bumping a version is the whole migration story: no schema migration, no rebuild, no
separate state. Records missing a version field read as version 0, so an index built
before versioning existed upgrades correctly.

## Concurrency

One writer at a time, enforced by an `O_EXCL` PID lock (`lock.py`) shared by every
entry point, because concurrent runs corrupt Chroma. A lock owned by a dead PID is
reclaimed automatically.

**User-started runs take priority over background ones.** `index_folders()` and
`upgrade_index()` accept a `should_stop` callable polled between photos. A watch pass
passes the watcher's own flag; when a user starts a run, `Watcher.yield_to_user()`
sets that flag and waits for the pass to stop at the next photo boundary. Everything
already indexed is kept — indexing is incremental, so the interrupted pass simply
resumes on its next tick. Background checks stay paused until the user's run finishes.

The web layer additionally guards with an in-process flag, so a second request is
rejected with 409 before a thread is even started, and registers shutdown handlers
that ask an in-flight run to stop and join rather than being torn down mid-computation.

## Watched folders

`watcher.py` runs one background thread that re-indexes the configured folders on a
timer. It polls rather than subscribing to filesystem events: an index run already
skips files whose size and mtime are unchanged, so a periodic pass is cheap, needs no
platform-specific machinery, and catches changes made while the app was closed.

Only `watch_enabled` and interval changes wake the timer; saving an unrelated setting
must never start indexing the user did not ask for. Only an explicit "Check now"
forces an immediate pass.

## Machine and model suitability

`sysinfo.py` detects what the machine will actually compute on — Apple Silicon (chip
name, unified memory), NVIDIA/AMD (device name and VRAM, via torch or `nvidia-smi`),
Intel Mac, or CPU — and judges each model against it.

Model *size* is a poor predictor of speed; the **runtime match** dominates. An MLX
build (`format: safetensors` with an MLX quantisation) runs natively on Apple
Silicon's GPU and is the wrong build anywhere else; a GGUF build runs through
llama.cpp, offloading to Metal/CUDA/ROCm when present and to the CPU otherwise. This
is why a 6.5 GB MLX model can beat a 2 GB GGUF one on a Mac. Memory is a secondary
factor, measured against *total* accelerator memory rather than momentary free RAM.

`sysinfo.running_offload()` reads Ollama's `/api/ps` for ground truth on how much of a
loaded model actually sits on the GPU.

## Interfaces

Both interfaces are thin layers over the same `indexer` / `search` modules — neither
has logic the other lacks.

**CLI** (`cli.py`) — `argparse`, one function per subcommand, progress rendered from
the same callback the web UI consumes.

**Web** (`web.py` + `templates/index.html`) — Flask, no build step, no CDN, no JS
dependencies. Four tabs: Search, Library, Index (one-off runs), Settings. Indexing
runs in a background thread; the browser polls `/api/index/status`.

| Endpoint | Purpose |
|---|---|
| `GET /api/status` | counts, caption health, model inventory, run and watcher state |
| `POST /api/search` | query → hits (+ optional answer) |
| `GET /api/library` | browse the index, with facets, sorting and paging |
| `GET /api/thumb/<id>` | cached thumbnail |
| `GET /api/photo?path=` | original file — **index members only** |
| `GET /api/browse?path=` | in-page folder list — **home + real mounts only** |
| `POST /api/pick-folder` | open the OS folder dialog **on the server's machine** (loopback only) |
| `POST /api/index` | start a one-off run (preempts the watcher; 409 if one is active) |
| `POST /api/upgrade` | re-run only stale pipeline components |
| `GET /api/index/status` | progress, log, notes, summary |
| `POST /api/index/clear` | empty the index (photo files untouched) |
| `GET/POST /api/settings` | saved defaults |
| `GET/POST /api/settings/folders` | watched folders: add, remove, recursion |
| `POST /api/watch/run` | check watched folders now |
| `GET /api/storage`, `POST /api/storage/location` | size breakdown; relocate |
| `POST /api/ollama/test` | check an Ollama address before saving it |
| `GET /api/system`, `GET /api/pipeline` | machine capabilities; component versions |

## Safety properties

Enforced in code, not by convention:

- **Path confinement.** `/api/photo` resolves the requested path and serves it only if
  that exact path is in the index, so the web UI cannot become a file browser. Folder
  selection is restricted to `Path.home()` plus `/Volumes` entries, excluding the
  macOS boot-volume firmlink (which resolves to `/` and would otherwise expose the
  whole filesystem).
- **Native dialog only for a local browser.** `/api/pick-folder` refuses non-loopback
  callers, since the dialog opens on the server's desktop.
- **Single writer**, with user runs preempting background ones (above).
- **Bundles are not descended into.** `.photoslibrary`, `.fcpbundle`, `.app` and
  friends are recorded and reported, never walked.
- **Moving the index is a CLI-only operation.** A running server holds Chroma's SQLite
  files open; moving them underneath it would corrupt the index. The web UI sets the
  location and hands over the command.
- **No silent degradation.** Every photo carries `caption_status`, every run carries
  notes, and systemic captioning failure disables captioning loudly mid-run.

## Module map

| Module | Responsibility |
|---|---|
| `config.py` | paths, tunables, prompts, model preferences, pipeline versions |
| `settings.py` | persisted user defaults and watched folders |
| `ollama.py` | client; capability, context-length and size discovery, model selection |
| `sysinfo.py` | accelerator detection and model suitability |
| `imaging.py` | HEIF registration, upright open, EXIF, thumbnails |
| `ocr.py` | Tesseract wrapper and output cleaning |
| `ocr_alt.py` | second text reader: Apple Vision, RapidOCR, or a VLM |
| `clip_embed.py` | CLIP image/text embeddings, device selection |
| `text_embed.py` | document embeddings with CLIP fallback |
| `caption.py` | prompt strategy, prose validation, failure tracking |
| `scan.py` | folder walking, bundle detection, picker confinement |
| `lock.py` | single-writer PID lock |
| `store.py` | Chroma collections, upserts, metadata queries, stale-handle recovery |
| `indexer.py` | orchestrates a run; upgrade planning and execution |
| `search.py` | three-signal query, RRF fusion, answer generation |
| `watcher.py` | background folder checks, yielding to user runs |
| `storage.py` | data directory size, location, relocation |
| `native_dialog.py` | the OS folder chooser, per platform |
| `server_pid.py` | tracking and stopping the web server |
| `cli.py` / `web.py` | interfaces |

## Deliberate limits

- **One CLIP model.** ViT-B/32 is small and fast; larger variants would improve recall
  at a cost in index time. The model name is a parameter, not a constant.
- **English-only sentence embedder.** Other scripts are served by the literal signal
  rather than semantically. A multilingual embedder would be the upgrade.
- **No face recognition or geocoding.** GPS is stored as raw coordinates, so "photos
  from Paris" only works if a caption or OCR says so.
- **Chroma, single process.** Fine for personal libraries; the single-writer lock is
  the scaling limit, not the vector store.
- **Library paging sorts in Python** over the full metadata list, since Chroma cannot
  sort. Fine for a personal library; this is the first thing to change at six figures.
- **Captions are as good as the local VLM.** Caption quality is the main lever on
  text-side recall.

---

## Change log

| Date | Change |
|---|---|
| 2026-08-28 | Initial build: three signals, two Chroma collections, RRF fusion, CLI + web UI |
| 2026-08-28 | Pipeline versioning and selective `upgrade` |
| 2026-08-28 | Second text reader (`ocr_alt`): Apple Vision, RapidOCR, VLM |
| 2026-08-28 | Literal-match retrieval signal, for CJK and exact terms |
| 2026-08-28 | Web UI split into four tabs; model pickers consolidated into Settings |
| 2026-08-28 | Persistent settings, watched folders, background watcher |
| 2026-08-28 | Relocatable data directory; storage reporting; clear index |
| 2026-08-28 | Accelerator detection and model suitability assessment |
| 2026-08-28 | User runs preempt background watch runs; graceful shutdown |
