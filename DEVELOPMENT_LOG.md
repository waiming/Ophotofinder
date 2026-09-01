# Development log

_Last updated: 2026-08-28. All work below was done on 2026-08-28._

Local vision models fail in ways that are quiet and specific. This is the record of
each failure mode, how it was verified on real hardware, and where it is handled in
code. See [ARCHITECTURE.md](ARCHITECTURE.md) for the design these constraints
produced.

Environment used throughout: macOS (Apple Silicon, MPS), Python 3.13, Ollama 0.33.1,
`moondream:latest`, `llama3.2:3b`, transformers 5.16.1, Tesseract 5.5.2.

---

## 1. Ask Ollama for capabilities; do not infer them from names

_2026-08-28_

`POST /api/show` returns a `capabilities` list. Only models reporting `vision` can
caption.

```
moondream:latest        capabilities: ['completion', 'vision']
llama3.2:3b             capabilities: ['completion', 'tools']
```

Guessing from names is wrong in both directions — `llama3.2:3b` and
`llama3.2-vision` differ by a suffix and by capability. Naming a non-vision model is
now a clear upfront error rather than a run of empty captions.

**Code:** `ollama.py` — `show()`, `vision_models()`, `pick_vision_model()`.

## 2. Send `num_ctx` explicitly, set to the model's own maximum

_2026-08-28_

Ollama defaults `num_ctx` low, which truncates a base64 image plus prompt and
returns nothing. The real maximum is in the model metadata under an
architecture-namespaced key:

```
moondream:latest → general.architecture = phi2 → phi2.context_length = 2048
```

The key name varies with architecture (`phi2.`, `qwen2.`, `llama.`), so the arch is
read first and any `*.context_length` key is used as a fallback. Every generate call
sends the resolved value.

**Code:** `ollama.py` — `show()` context-length resolution; `generate(num_ctx=…)`.

## 3. Small VLMs return nothing for long prompts

_2026-08-28_

The most expensive finding, and measured directly on the same image and model:

| Prompt | Result |
|---|---|
| Long multi-part ("describe subjects, setting, objects, colours, indoors/outdoors…") | `''` — 0 characters, `prose=False` |
| `"Describe this image."` | 360 characters of usable prose, `prose=True` |

Small models also answer a prompt they cannot parse with bare grounding
coordinates — `[0.31, 0.71, 0.64, 0.87]` — which is not a caption but is not empty
either, so a naive truthiness check accepts it.

Handling: models at or under 3B parameters get the short prompt first and the long
one as fallback; larger models get the reverse. Any caption that is empty **or**
non-prose triggers the other prompt.

`is_prose()` rejects: empty, under 12 characters, fewer than 4 real words, pure
digits/brackets/punctuation, and strings under 60% alphabetic. Verified against
both real captions and real coordinate replies.

**Code:** `caption.py` — `is_prose()`, `Captioner.caption()`;
`config.py` — `SMALL_VLM_PARAM_LIMIT`, both prompts.

## 4. `exif_transpose` before OCR, CLIP and thumbnails

_2026-08-28_

A portrait photo stored sideways with `Orientation=6` — the standard iPhone case.
Same file, same Tesseract, one difference:

```
without exif_transpose:  'SYIVLS\nye\nAula'      ← mirrored gibberish
with    exif_transpose:  'STAIRS'
```

Handling: `open_upright()` is the only entry point for image data, applying
`ImageOps.exif_transpose` at open. Every downstream consumer receives upright
pixels; no consumer can forget.

**Code:** `imaging.py` — `open_upright()`.

## 5. Captioning must never fail silently

_2026-08-28_

Three mechanisms:

1. **Per-photo reason.** Every record stores `caption_status` (`ok` / `empty` /
   `skipped` / `error`) plus `caption_detail` naming which prompts were tried and
   how each failed (e.g. `primary:non-prose, fallback:non-prose`).
2. **Mid-run abort.** If the first 5 photos all caption empty, captioning switches
   off for the rest of the run and the reason is emitted as a warning to both CLI
   and web UI. Verified with a stubbed model returning `''` — tripped at photo 5
   with *"the first 5 photos all produced an empty or non-prose caption"*.
3. **End-of-run warning.** A run that indexed photos but produced zero captions says
   so explicitly in the summary.

**Code:** `caption.py` — `Captioner._maybe_disable()`; `indexer.py` — warn events.

## 6. `llama3.2-vision` does not load

_2026-08-28_

It reports `vision` but fails on current Ollama with
`unknown model architecture: 'mllama'`. Because the capability check passes, it must
be excluded by name, or auto-selection picks a model that cannot run.

It stays visible in `ophotofinder models`, marked broken, rather than being hidden —
the point is to explain the failure, not conceal the model.

**Code:** `config.py` — `BROKEN_VISION_MODELS`; `ollama.py` — `is_broken`.

## 7. HEIC, and the sealed `~/Pictures` library

_2026-08-28_

`pillow_heif.register_heif_opener()` runs at import; iPhone photos are HEIC and
Pillow cannot open them otherwise. AVIF registration is attempted where available.
Browsers cannot render HEIC, so the web UI serves the cached upright JPEG thumbnail
for those files instead of the original.

On macOS `~/Pictures` usually contains only `Photos Library.photoslibrary`, a sealed
package. Reporting "0 indexed" would be technically true and useless. Instead the
scanner detects bundle suffixes and explains:

> No loose image files here. This folder contains only sealed photo library
> package(s): Photos Library.photoslibrary. […] To index these photos, open
> Photos.app and use File > Export > Export Unmodified Originals…

**Code:** `imaging.py` — HEIF registration; `scan.py` — `BUNDLE_SUFFIXES`,
`ScanReport.explain_empty()`.

## 8. One indexing run at a time

_2026-08-28_

Concurrent runs corrupt Chroma. An `O_EXCL` PID lock file is shared by CLI and web.
Verified: with a web run holding the lock, a CLI run exits 3 with
*"An indexing run is already in progress (pid 80118, started …)"*, and a second web
request returns 409. A lock owned by a dead PID is reclaimed rather than blocking
forever — also verified.

**Code:** `lock.py`; `web.py` — `IndexJob`.

---

# Bugs found during the build

_All found and fixed 2026-08-28._

## transformers 5.x changed the CLIP return type

_2026-08-28_

`CLIPModel.get_image_features()` returned a bare tensor through transformers 4.x. In
5.16.1 it returns a `BaseModelOutputWithPooling` whose `pooler_output` holds the
projected embedding:

```
AttributeError: 'BaseModelOutputWithPooling' object has no attribute 'detach'
```

Fixed with a `_features()` unwrapper handling tensor, `pooler_output`, and tuple
returns, so the code works on both major versions.

## The folder picker was open to the whole filesystem

_2026-08-28_

The picker was meant to allow the home directory plus mounted volumes. On macOS
`/Volumes/Macintosh HD` is a firmlink that **resolves to `/`**, so the containment
check accepted every path on disk:

```
before:  /etc → 200    /usr/local → 200
after:   /etc → 403    /usr/local → 403    ~/Pictures → 200
```

Fixed by dropping any `/Volumes` entry resolving to `/`. Photo *serving* was never
affected — it checks index membership, and `/etc/passwd` was refused throughout.

## `clean` permanently broke a running web server

_2026-08-28_

`drop_all()` deleted each Chroma collection by name and recreated it. Recreation
assigns a **new collection id**, so any `Collection` handle held elsewhere — the
long-running web server's `Store`, most importantly — stayed bound to an id that no
longer existed. Every subsequent call raised, forever, until the server was
restarted:

```
chromadb.errors.NotFoundError: Collection [0bb9caa3-…] does not exist.
GET /api/status  500
```

Two fixes, because either alone is insufficient:

1. `drop_all()` now **empties** the collections by deleting their records instead of
   dropping the collections, so live handles stay valid.
2. `Store._call()` catches `NotFoundError`, re-opens both collections by name and
   retries once — so a handle made stale by anything else (an older `clean`, a
   manual reset, a restored backup) recovers rather than failing until restart.

The retry needed a second pass: the first version took an already-bound method
(`self._retry(self.clip.count)`), so re-opening rebound `self.clip` while the retry
still invoked the dead object. It now takes the attribute *name* and re-resolves the
collection on each attempt. A regression test that deletes collections out from under
a live `Store` covers `counts`, `get`, `query` and the search path.

## OCR rejected single-word signs

_2026-08-28_

The cleaner required 2+ words, which discarded exactly the content that matters most
for photo search: `STAIRS`, `INVOICE`, `EXIT`. Lowered to one real word plus a
minimum length. OCR hits across the test set went from 1/6 to 3/6.

---

## Tesseract is weak on real-world and non-Latin text

_2026-08-28_

Measured on three real photos:

| Photo | Tesseract | Apple Vision |
|---|---|---|
| Chinese news screenshot | `Now #i fl 54m ago o BIAARRS \| FSk2R 33 eae Be` (garbage) | `即日焦點 \| 打鼓嶺33歲寵物酒店女職員被唐狗襲擊死亡`, `許家印一審被判無期徒刑` — 173 CJK glyphs |
| Gallery wall text, angled | 207 chars of fragments (`pe who / erie €5, nd`) | 622 chars, confidence 1.00, fully readable |
| Distant storefront sign | `———— —— == BEAUSSU ge EAMERY` | `Time Out MARKET`, `LUNCH LAD` |

Two causes: only the `eng` language pack was installed, and Tesseract is built for
flat document text rather than photographs. Apple's Vision framework is on-device,
needs no download, and reads 30 languages. Added as a **second field**
(`ocr_alt`), never overwriting Tesseract's — they fail differently and both are worth
keeping.

A local VLM was tested as a third option and rejected for this setup: asked to
transcribe, moondream returned `!!!IMAGE!!!` and `[0.0, 0.0, 0.99, 0.99]`. The `vlm`
backend exists for capable models, and rejects junk replies rather than storing them.

### Keeping it cross-platform

Apple Vision is macOS-only, which would make the best text reader unavailable to
anyone publishing or running this elsewhere. RapidOCR (PP-OCRv3 on ONNX Runtime)
fills that gap: it installs from pip on Linux, Windows and macOS, and ships its
models inside the wheel, so it needs no download and stays fully offline.

Measured on the same three photos:

| Reader | Chinese screenshot | Angled Latin text | Speed |
|---|---|---|---|
| Tesseract | 0 CJK glyphs (garbage) | 207 chars of fragments | 0.6s |
| RapidOCR | **143 CJK glyphs** | 508 chars | 0.8s |
| Apple Vision | **173 CJK glyphs** | 622 chars, conf 1.00 | 1.4s |

`auto` therefore resolves Apple Vision → RapidOCR → VLM → none, and all four paths
were tested by stubbing availability: macOS picks `apple`; a non-Mac with the extra
picks `rapidocr` and reads the Chinese correctly; a non-Mac without it degrades to
Tesseract + CLIP + EXIF with an explanatory note rather than an error; asking
explicitly for a missing backend returns the pip command to install it.

### Vision's language order matters enormously

Fixing the language list is worse than letting Vision detect the script:

| Setting | Chinese screenshot | English gallery text |
|---|---|---|
| `en-US` first | 123 chars, **0 CJK glyphs** (`Now #`, `#ili`) | 622 chars, conf 1.00 |
| `zh-Hant` first | 347 chars, 174 CJK glyphs, conf 0.62 | 531 chars, conf **0.36** (mangled) |
| **automatic detection** | 346 chars, 173 CJK, conf **0.77** | 622 chars, conf **1.00** |

Automatic detection matched the best fixed order on both, so it is the default. This
was caught only because the first indexing run produced *worse* Chinese than a
standalone test — the config defaulted to `en-US` first.

## The text embedder cannot represent Chinese at all

_2026-08-28_

Storing Chinese text is useless if nothing can search it. `all-MiniLM-L6-v2` is
English-only, and it shows:

```
sim(chinese news A, chinese news B)      = 0.502
sim(chinese news A, unrelated chinese)   = 0.573   <- higher!
```

Both Chinese queries returned the wrong photo. Fixed by adding a third retrieval
signal: literal substring matching via Chroma's `$contains`, fused into RRF
alongside CLIP and the text embedding. All five test queries then returned the
correct photo, and the English/CLIP results were unchanged.

## New photos did not appear in the Library during a run

_2026-08-28_

Two independent causes, both measured before fixing:

**1. Write batching.** Records were flushed only every `CLIP_BATCH` (8) photos, so a
separate reader trailed the indexer by up to 8. Harmless on a fast run (a fraction of
a second) but with captioning at several seconds per photo it meant 30–60 seconds of
nothing appearing. Chroma was ruled out first: writes are visible immediately, both
in-process and across processes.

Fixed by flushing on **age as well as size** (`FLUSH_INTERVAL_SECONDS = 2.0`).
Measured with ~1s per photo: worst-case lag fell from 8 photos to 2.

**2. Sort order.** The Library sorts by capture date. Photos indexed later are not
necessarily *newer*, so they can land anywhere. Reproduced by indexing 20 recent
images then 3 older ones: the 3 new photos landed at positions 21–23 of 23 — the last
page — while the user refreshes page 1 and sees nothing.

Fixed by making the Library live during a run: it polls, shows a progress banner, and
offers a one-click switch to *most recently indexed first*. The sort was never wrong;
it just answered a different question than "what did I just add?".

## The Library filter could not see the second reader's text

_2026-08-28_

The Library's free-text box matched against `filename, caption, ocr, camera, folder`
— but not `ocr_alt`, where Apple Vision's output lives. Every Chinese character
extracted was therefore invisible to it:

```
before:  q='許家印'    -> 0 matches
         q='Time Out'  -> 0 matches
after:   q='許家印'    -> 2026-08-20 10.00.27.png
         q='Time Out'  -> 2026-08-25 16.42.50.jpg
```

Fixed by extending the fields to `ocr_alt`, `taken_at` and the indexed `document`.
The box was later removed from the UI entirely — it read as a second, weaker search
and confused more than it helped — but the endpoint still supports `q=`.

## The watcher indexed on every settings save

_2026-08-28_

The first version called `watcher.poke()` from the settings endpoint, so ticking any
checkbox immediately started an indexing pass — `running_now: True` right after
saving an unrelated preference. On a machine already under load that is close to the
worst possible behaviour.

Now only `watch_enabled` and interval changes wake the timer, and only an explicit
"Check now" forces a pass. Verified: `running_now` stays `False` after enabling
watching and after saving unrelated settings.

## Free memory was the wrong thing to measure

_2026-08-28_

The first model-suitability check compared model size against *currently free* RAM.
It produced a backwards answer: it rated `gemma4:e2b-mlx` (6.5 GB) worse than
`llama3.2:3b` (2.0 GB), while in practice the larger model ran better.

The metadata explains why:

```
gemma4:e2b-mlx    format=safetensors  quant=nvfp4     ← MLX runtime
llama3.2:3b       format=gguf         quant=Q4_K_M    ← llama.cpp runtime
```

MLX builds are compiled for Apple Silicon and run natively on the unified-memory GPU.
GGUF goes through llama.cpp, which offloads to Metal/CUDA/ROCm when present and falls
back to the CPU otherwise. **The runtime match dominates size**, and free RAM is noise
that changes minute to minute.

Rewritten around accelerator detection. Memory is now secondary and measured against
*total* accelerator memory. The same models are judged differently per machine:

```
Apple M2, 24 GB        NVIDIA RTX 4070        Linux CPU-only, 8 GB
✓ gemma4:e2b-mlx good  ✗ gemma4:e2b-mlx bad   ✗ gemma4:e2b-mlx bad
· llama3.2:3b    ok    ✓ llama3.2:3b    good  ! llama3.2:3b    warn
```

## `OLLAMA_HOST` was an import-time constant

_2026-08-28_

Making the server address configurable was not a matter of adding a setting: the
address was read into a module constant at import, so nothing could change it without
a restart. Now resolved per call via `ollama.host()` — saved setting, then
environment, then default — and changing it clears cached model metadata, since
capabilities and context lengths do not carry across servers.

## Background checks outranked the user

_2026-08-28_

The lock made concurrent runs impossible, but it was first-come-first-served: if a
watch pass held the lock, a run the user started was refused. That is backwards.

`index_folders()` and `upgrade_index()` now accept a `should_stop` callable polled
between photos, and `Watcher.yield_to_user()` sets it. Measured on a live race:

```
watcher active=True, lock held=True
→ yielded in 1.2s: "stood aside for a run you started after 1 photo(s)"
→ one-off acquired the lock, indexed 3
→ new watch tick refuses: "paused while you run an indexing job"
→ after the user's run: "19 new or changed photo(s)"
```

Nothing is lost: indexing is incremental, so an interrupted pass resumes next tick.

## An unreproduced segfault at interpreter exit

_2026-08-28 — open_

During the priority test above, the process exited with a segfault (139) *after* all
assertions had passed and the index was verified correct. Three targeted attempts to
reproduce it failed: two sequential runs in one process, a run in a background thread,
and exiting while a daemon thread was mid-run all exited cleanly.

Best current explanation is torch/MPS teardown in a contrived harness that loaded CLIP
four times across two threads — not application logic. It is recorded here rather than
dismissed, because it has not been explained.

Mitigation applied regardless: the web server registers `atexit` and signal handlers
that ask an in-flight run to stop and join it (up to 20s), so torch work is not torn
down mid-computation on shutdown.

# Verification performed

_As of 2026-08-28._

Fixtures were generated synthetically to exercise specific paths: a portrait JPEG
with `Orientation=6` and iPhone EXIF, a HEIC file, an OCR-bearing whiteboard, a
nested subfolder, and a non-image file.

| Check | Result |
|---|---|
| Capability + context discovery | `moondream` → `vision`, `phi2.context_length=2048` |
| Broken-model exclusion | `llama3.2-vision` flagged, auto-pick chose `moondream` |
| Long vs short prompt | 0 chars vs 360 chars, same image |
| `exif_transpose` effect on OCR | `SYIVLS ye Aula` → `STAIRS` |
| Full index run | 6/6 indexed, 6/6 captions, 3 OCR hits, 0 failures, MPS |
| HEIC indexing | `forest.heic` indexed and captioned |
| EXIF extraction | `2024-07-14T18:22:10`, `Apple iPhone 15 Pro`, `portrait` |
| Incremental re-run | 0 indexed, 6 skipped unchanged |
| CLIP semantic search | top-1 correct for "sunset over ocean", "red car", "trees in a forest" |
| OCR-driven search | "fire exit stairs sign" → correct photo at rank 1 |
| EXIF filter | `--camera "Apple iPhone 15 Pro"` → 1 correct result |
| RAG answer | correctly cited `[1]` and quoted "FIRE EXIT STAIRS" |
| `is_prose` | 6/6 cases correct, including coordinate replies |
| Caption auto-disable | disabled at photo 5 with a stubbed empty model |
| Lock: web vs CLI | CLI exit 3; second web POST 409; stale lock reclaimed |
| Path confinement | indexed 200; `/etc/passwd` 403; traversal 404 |
| Picker confinement | home 200; `/etc`, `/usr/local`, `/private/tmp` 403 |
| No tesseract on `PATH` | 6/6 still indexed, EXIF document intact, clear note |
| Sealed library detection | `~/Pictures` explained with export instructions |
| Second reader vs Tesseract | 143–173 CJK glyphs vs 0; 622 vs 207 chars on angled text |
| Vision language handling | automatic detection matched the best fixed order on both scripts |
| Literal retrieval signal | 5/5 queries correct, including Chinese; English results unchanged |
| Cross-platform reader fallback | non-Mac + extra → RapidOCR; without it → clean degradation |
| Pipeline upgrade | stale components re-run with **0** Ollama calls; `--only` respected |
| Stale Chroma handle | `counts`, `get`, `query` and search all recover after a collection delete |
| Web server stop | pid-file stop, discovery of a server with no pid file, port left alone if not ours |
| Native folder dialog | all four OS backends parse; cancel, out-of-bounds, non-folder, remote all guarded |
| Settings persistence | model, host and folder settings survive a restart |
| Watcher eagerness | `running_now` stays False after unrelated saves |
| One-off priority | watcher yields in 1.2s; user run proceeds; watch resumes after |
| Storage relocation | pointer honoured, move succeeds, non-empty target refused, reset works |
| Clear index | records and thumbnails removed; photo files verified still on disk |
| Model suitability | verdicts invert correctly across Apple Silicon / CUDA / CPU-only |
| Ollama address | saved host takes effect without restart; bare `host:port` normalised |

## Not yet verified

_As of 2026-08-28._

- **Caption quality on real photographs at scale.** Fixtures are synthetic, so
  moondream described them literally and sometimes wrongly. Retrieval ranked correctly
  regardless, but caption quality across a whole library is untested — run
  `--limit 20` on a real folder before committing to a full index.
- **Scale.** Largest runs were tens of photos. Chroma behaviour, RRF pool sizing and
  the Library's in-Python sort at 10k+ photos are unmeasured.
- **Non-Apple hardware.** CUDA, ROCm, Linux and Windows paths are implemented and
  unit-checked with simulated accelerators, but never executed on that hardware.
  RapidOCR was exercised on macOS only.
- **The segfault above** remains unexplained, though unreproducible in three attempts.
- **Long-running watcher.** Verified over minutes, not days; interval drift and
  behaviour across sleep/wake are untested.
