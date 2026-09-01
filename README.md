# Ophotofinder

A local RAG over your own photo folders. Ask a question in plain language, get the
matching photos back. Everything runs on your machine — Ollama for the models, no
cloud, no API keys.

```bash
ophotofinder index ~/Pictures/Exported
ophotofinder search "my dog on the beach at sunset"
ophotofinder web            # http://127.0.0.1:7777
```

Each photo is indexed on several independent signals — a CLIP image embedding, a
caption from a local vision model, two different text readers, and EXIF — so a photo
stays findable even when the others come up empty.

- [ARCHITECTURE.md](ARCHITECTURE.md) — how it fits together, and why.
- [DEVELOPMENT_LOG.md](DEVELOPMENT_LOG.md) — the local-model pitfalls this works
  around, and every bug found while building it.

## Requirements

- Python 3.10+, on **Linux, macOS or Windows**
- [Ollama](https://ollama.com) running locally, with a vision-capable model pulled
- `tesseract` — optional, for OCR
- Apple Silicon (MPS) or CUDA is used automatically when present; CPU otherwise

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate

pip install -e .                      # core
pip install -e '.[ocr]'               # + cross-platform second text reader (any OS)
pip install -e '.[macos]'             # + Apple Vision second reader (macOS only)

ollama pull moondream                 # a vision-capable model is required for captions
brew install tesseract                # optional; apt install tesseract-ocr on Linux
```

**Platform support.** Everything core is cross-platform: CLIP runs on MPS, CUDA or
CPU, and the rest is pure Python. The only macOS-specific piece is the optional
Apple Vision text reader, which has a cross-platform equivalent in `[ocr]`.

### Reading text in photos

Two independent text readers run per photo, stored in separate fields:

| Reader | Strengths | Platform |
|---|---|---|
| **Tesseract** | flat, high-contrast Latin text — screenshots, scans, documents | any (`brew`/`apt install tesseract`; English only by default) |
| **Apple Vision** | real-world text — angled, curved, unevenly lit — and 30 languages including Chinese, Cantonese, Japanese, Korean | macOS, `pip install 'ophotofinder[macos]'` |
| **RapidOCR** | real-world text, Chinese + English; PP-OCRv3 via ONNX, models bundled in the wheel so it stays offline | **any** — Linux, Windows, macOS: `pip install 'ophotofinder[ocr]'` |

They fail in different ways, so keeping two is the point. `--alt-ocr auto` (the
default) picks the best reader present — Apple Vision on macOS, otherwise RapidOCR,
otherwise a capable vision model, otherwise nothing. Choose explicitly with
`--alt-ocr apple|rapidocr|vlm|none`.

Measured on the same photo of a Chinese news screen: Tesseract read 0 Chinese
characters, RapidOCR 143, Apple Vision 173. On Latin text photographed at an angle,
Tesseract produced 207 characters of fragments where Apple Vision produced 622 at
full confidence.

With no second reader installed, indexing simply continues on Tesseract + CLIP +
EXIF and says so — nothing fails. `ophotofinder doctor` shows which readers this
machine has.

Apple Vision **detects the language automatically** by default, which measurably
beats any fixed order; override with `--alt-langs en-US,zh-Hant` if you need to.

`tesseract` is the only optional piece. Without it OCR is skipped and everything else
still works — CLIP, captions and EXIF are unaffected, so photos remain findable by
appearance, description and date/camera. You only lose searching for words *printed
inside* a photo. Runs say so explicitly (`tesseract binary not found: OCR disabled
for this run.`).

Verify the setup at any time:

```bash
ophotofinder doctor
```

```
Ophotofinder doctor
  data dir       /Users/you/.ophotofinder
  HEIC support   yes
  tesseract OCR  yes
  ollama         up at http://localhost:11434
  torch device   mps
  vision models  moondream:latest
  indexed        0 photos (clip) / 0 (text)
```

## Indexing

```bash
ophotofinder index ~/Photos                  # recursive by default
ophotofinder index ~/Photos ~/Screenshots    # several folders at once
ophotofinder index ~/Photos --limit 20       # try a sample first
```

Re-running on the same folder only processes files whose size or mtime changed;
everything else is skipped, so re-indexing is cheap.

| Flag | Effect |
|---|---|
| `--vision-model M` | caption with a specific Ollama model |
| `--no-caption` | skip captions (CLIP + OCR + EXIF only — much faster) |
| `--no-ocr` | skip Tesseract OCR |
| `--alt-ocr M` | second text reader: `auto` (default), `apple`, `rapidocr`, `vlm`, `none` |
| `--alt-langs L` | force languages for the second reader, e.g. `en-US,zh-Hant` |
| `--no-recursive` | do not descend into subfolders |
| `--reindex` | re-process files even if unchanged |
| `--limit N` | stop after N photos |
| `--quiet` | summary only, no per-photo lines |

Progress marks each photo with which signals landed, and the summary is explicit
about what did not:

```
[1/6] cap ---  beach_sunset.jpg      The image features a vibrant orange sky above...
[2/6] cap ocr  exit_sign.jpg         The image features a white sign that reads "FIRE...

--- indexing summary ---
  indexed              6
  failed               0
  captions ok/empty/off 6/0/0
  photos with OCR text 3
  note: captioning with moondream:latest (arch phi2, num_ctx=2048, short prompt first)
```

> **macOS:** `~/Pictures` normally holds only a sealed `.photoslibrary` package that
> no app can read directly. Export first — Photos.app → File → Export → Export
> Unmodified Originals — then index that folder. Ophotofinder tells you this rather
> than reporting "0 indexed".

## Searching

```bash
ophotofinder search "my dog on the beach at sunset"
ophotofinder search "whiteboard with writing" -k 20
ophotofinder search "receipts from last year" --year 2024 --with-text
ophotofinder search "where did I park" --answer
```

| Flag | Effect |
|---|---|
| `-k N` | number of results (default 10) |
| `--year Y` / `--camera C` / `--folder F` | filter on EXIF / location |
| `--with-text` | only photos containing readable text |
| `--clip-only` / `--text-only` | use a single signal |
| `--answer` | also generate an answer from the matches with a local text model |
| `--text-model M` | pick the model used by `--answer` |
| `--json` | machine-readable output |

Every result shows which signals found it, so a weak match is visible as one:

```
 1. /Users/you/Photos/exit_sign.jpg
    visual #1 + text #1  score 0.0328  2024-07-14 | Apple iPhone 15 Pro | portrait
    The image features a white sign with black text that reads "FIRE EXIT STAIRS"...
    OCR: STAIRS
```

## Web UI

```bash
ophotofinder web                 # http://127.0.0.1:7777
ophotofinder web --port 8080 --host 0.0.0.0
ophotofinder web --stop          # stop a server running in the background
```

Starting a server records its pid, so `--stop` shuts it down cleanly — including a
server started from another shell, which is found by matching the process rather
than the port. Anything else holding the port is reported and left alone:

```
Port 7777 is already in use by Python (pid 94923).
That process was not started by Ophotofinder, so it is left alone. Use --port to pick a free port.
```

`ophotofinder status` shows whether a server is running, and starting a second one on
a busy port now fails with an explanation instead of a Flask traceback.

Same capabilities in a browser: search with filters, a signal-source badge on every
hit, a lightbox, and an indexing panel with a folder picker, live progress and the
same warnings the CLI prints.

It has four tabs:

- **Search** — ask a question, get ranked photos with a signal-source badge on each.
- **Index** — choose a folder, set per-run model and reader options, watch progress.
- **Settings** — default models, watched folders, and whether photos are kept
  current automatically. Technical detail (pipeline versions, second-reader choice,
  machine info) is folded away under *Advanced technical settings*.
- **Library** — browse everything indexed, as an album. Each tile shows its caption
  and flags anything notable (missing caption, OCR present). Narrow by folder,
  camera, caption state or OCR presence, and sort by capture date, filename, folder
  or index time, with a
  Refresh button that re-reads the index (picking up photos and folders added since
  the page was opened). While an indexing run is active it reloads by itself and
  shows progress. Note that the default sort is by **capture date**, so a photo taken
  in 2019 but indexed today lands on a later page — the banner offers a one-click
  switch to *most recently indexed first*, which is usually what you want while
  indexing. Clicking a photo opens a detail panel — on a wide screen the
  grid makes room for it, so the other photos stay visible and clickable — showing
  every recorded signal: caption and its status,
  OCR text, EXIF (date, camera, lens, GPS, dimensions, orientation), which model
  captioned it, when it was indexed, and the exact document text that was embedded.

Dates with no EXIF fall back to the file's modification time and are labelled
`(file date — no EXIF)` so a guess is never mistaken for a real capture date.

Models are selectable in both places, each defaulting to `auto`:

- **Caption model** — in the indexing panel, listing every vision-capable model
  (models that report `vision` but cannot run are shown disabled and labelled).
- **Answer model** — next to the *answer with local LLM* checkbox in the search
  filters, listing every completion-capable model with its max context.

### Choosing a folder

Three ways, in the Index tab:

1. **📂 Choose folder…** opens your operating system's own folder dialog —
   `osascript` on macOS, `FolderBrowserDialog` on Windows, zenity/kdialog on Linux.
2. **Paste a path** — type or paste `~/Pictures/Exported` and press Enter.
3. **Browse folders here** — an in-page list, with sealed library packages marked.

A web page cannot open a native dialog that yields a filesystem path: browsers report
`C:\fakepath\...` by design, and `<input type="file">` hands over file *bytes*, not
locations. That is fine for a page that processes an image in the browser, but this
server needs paths — it re-opens photos to serve them, to skip unchanged files on a
re-index, and to re-run a single component during `upgrade`. Making `<input
type="file">` work would mean uploading the whole library to a server on the same
disk. So the native dialog is opened by the *server*, which does have filesystem
access. It is hidden automatically when the browser is not on the same machine, since
the dialog would otherwise open on the wrong desktop.

Two limits are enforced server-side regardless of how a folder is chosen: photos are
served **only** for paths already in the index, and folder selection is confined to
your home directory and real mounted volumes.

## Watched folders

The **Index** tab is for one-off runs over a folder you pick. For folders that keep
changing, add them under **Settings → Watched folders**: they are re-checked on a
timer and anything new or changed is indexed automatically. Each folder has its own
*subfolders* toggle, and folders can be added or removed at any time.

```bash
ophotofinder config --add-folder ~/Pictures/Exported
ophotofinder config --add-folder ~/Desktop/Screenshots --no-recursive
ophotofinder config --remove-folder ~/Desktop/Screenshots
ophotofinder config --set watch_enabled=true --set watch_interval_minutes=30
ophotofinder config                                    # show everything saved
```

The **Index** tab always wins. Starting a one-off run asks any in-flight background
check to stand aside; it stops at the next photo (keeping everything already done),
releases the lock, and your run begins — typically within a second or two. Background
checks stay paused until your run finishes, then resume from where they left off.
The two can never run at the same time: a single on-disk lock allows one writer, and
the watcher refuses to start while a user run holds or is waiting for it.

Settings shows exactly where things stand — `Last checked 2026-08-28 23:47:34
(2m ago) — 3 new or changed photo(s)` and `Next check 2026-08-29 00:17:34 (in 29m)`,
refreshed while the tab is open.

Watching polls rather than subscribing to filesystem events: an index run already
skips files whose size and mtime are unchanged, so a periodic pass is cheap, needs no
platform-specific machinery, and also catches changes made while the app was closed.
A pass that finds an indexing run already in progress simply waits for the next one.

## Models and the Ollama server

**Settings → Models** is the single place models are chosen: the vision model used to
describe photos, and the text model used to answer questions. Both default to `auto`.
The search bar shows which model would answer and links here rather than offering its
own picker, so there is one place to change it. A `--vision-model` flag on a single
run still wins over the saved default.

**Settings → Ollama server** sets the address. Leave it blank for the default
(`http://localhost:11434`, or `OLLAMA_HOST` if set). A bare `host:port` is accepted and
normalised. **Test** checks the address before you commit to it, reporting how many
models it found. Changing the address takes effect immediately — no restart — and
cached model metadata from the old server is discarded.

```bash
ophotofinder config --set vision_model=moondream:latest
ophotofinder config --set text_model=llama3.2:3b
ophotofinder config --set ollama_host=192.168.1.10:11434
ophotofinder config --set ollama_host=            # back to the default
```

`ophotofinder models` lists what is installed, with each model's runtime, size and
how well it suits this machine. A single run can still override the saved default:

```bash
ophotofinder index ~/Photos --vision-model moondream
ophotofinder search "birthday cake" --answer --text-model llama3.2:3b
```

## Will the model run well here?

Model size is a poor predictor of speed. What matters far more is whether the
model's **build matches the machine's accelerator**:

| Build | Runs on | Notes |
|---|---|---|
| **MLX** (`format: safetensors`, MLX quantisation) | Apple Silicon only | Native to the unified-memory GPU; usually the fastest option on a Mac, and the wrong build anywhere else |
| **GGUF** | anywhere | llama.cpp: offloads to Metal, CUDA or ROCm when present, otherwise CPU |

This is why a 6.5 GB MLX model can comfortably beat a 2 GB GGUF one on Apple
Silicon — and why the same MLX model is a poor choice on an NVIDIA box.

Everything is **detected on the machine it runs on**: Apple Silicon (chip name and
unified memory), NVIDIA/AMD (device name and VRAM, via torch or `nvidia-smi`), Intel
Mac, or CPU-only. The same model is therefore judged differently on different
computers:

```
Apple M2, 24 GB          NVIDIA RTX 4070, 12 GB     Linux CPU-only, 8 GB
✓ gemma4:e2b-mlx  good   ✗ gemma4:e2b-mlx    bad    ✗ gemma4:e2b-mlx   bad
· llama3.2:3b     ok     ✓ llama3.2:3b       good   ! llama3.2:3b      warn
```

Four verdicts — **good** (matched to the accelerator), **ok**, **warn** (yellow: will
run but slowly, e.g. no GPU), **bad** (red: wrong build for this hardware, or larger
than the machine's memory). Shown by `ophotofinder models`, in Settings as you pick a
model, and as a warning before an index run starts.

`ophotofinder doctor` also reports the detected accelerator and, for anything Ollama
currently has loaded, how much of it actually sits on the GPU — the ground truth,
straight from `/api/ps`:

```
  accelerator    Apple M2 — GPU via Metal, unified 24.0 GB shared with the system
    loaded now   moondream:latest: fully on the GPU
```

## Keeping an index current

The indexing pipeline will keep improving. Rather than forcing a full re-index every
time, each signal is produced by a **versioned component**, and every photo records
the versions it was built with:

```
pipeline caption1.clip1.document2.exif2.ocr2.ocr_alt1
```

When a component improves, bump its version in `config.PIPELINE_VERSIONS`. Photos
built with an older version are then *identifiably* stale, and only the changed
component is re-run:

```bash
ophotofinder upgrade --check          # what is behind, and what it would cost
ophotofinder upgrade                  # re-run only the stale components
ophotofinder upgrade --only ocr_alt   # just one component
```

```
pipeline caption1.clip1.document2.exif2.ocr2.ocr_alt1
  348 photo(s) in the index
  0 up to date, 348 behind

  behind, by component:
    ocr_alt    v1     348 photo(s)   (medium to recompute)
    exif       v2     348 photo(s)   (cheap to recompute)
```

This matters because costs are wildly uneven: re-reading EXIF is nearly free, while
re-captioning with a vision model dominates everything. Adding the second text reader
upgraded photos **without a single call to the vision model** — captions were already
current, so they were reused untouched.

`ophotofinder status` flags stale photos, the **Settings** tab shows the same table
with an upgrade button, and each photo's detail panel shows the versions it carries.

### When you change the pipeline

| Change | Bump | Effect |
|---|---|---|
| better/different caption model or prompt | `caption` | expensive — re-runs the VLM |
| new or improved text reader | `ocr_alt` | medium |
| OCR extraction or cleaning fix | `ocr` | medium |
| EXIF parsing fix | `exif` | cheap |
| different CLIP model | `clip` | medium — re-embeds images |
| change to what goes into the embedded text | `document` | cheap — rebuilds and re-embeds text |

Bumping any document input rebuilds the document automatically.

## Where the index lives

**Settings → Advanced → Storage** shows the location, total size, and a pie chart
breaking down the vector database against thumbnails, plus free space on that disk. You can
point Ophotofinder somewhere else — an external drive, say.

```bash
ophotofinder storage                          # location and size
ophotofinder storage --move-to /Volumes/Ext/ophotofinder   # move it and use it
ophotofinder storage --set-location DIR       # use DIR, leave existing data behind
ophotofinder storage --reset-location         # back to ~/.ophotofinder
```

The chosen location is remembered in `~/.ophotofinder-location`, deliberately
*outside* the data directory — `settings.json` lives inside it, so a location stored
there would be lost the moment the directory moved. `OPHOTOFINDER_HOME` still
overrides everything.

Changing the location takes effect on restart. **Moving** existing data is a CLI
operation (`--move-to`): a running server holds Chroma's SQLite files open, and
moving them underneath it would corrupt the index. The web UI therefore sets the
location and tells you the command to bring your data across. A move refuses a
non-empty target folder.

## Cleaning the index

Removes records from the index. **Photo files are never touched.**

There is also a **Clear index** button in the **Danger zone** on the main Settings
page, which empties the index and its thumbnails after a confirmation. As with `clean`,
**photo files are never touched**.

```bash
ophotofinder clean                       # wipe the whole index (asks first)
ophotofinder clean --folder ~/Photos/Old # only photos under one folder
ophotofinder clean --missing             # drop records whose file is gone
ophotofinder clean --yes                 # skip the confirmation prompt
```

`--missing` is the one to reach for after moving or deleting photos: the index keeps
pointing at paths that no longer exist, and this prunes exactly those. Matching
thumbnails are removed too, and cleaning is refused while an indexing run is active.

To reset everything including cached models, delete the data directory:
`rm -rf ~/.ophotofinder`.

## Command reference

```
ophotofinder doctor                      check the local setup
ophotofinder models                      models, runtimes, and how they suit this machine
ophotofinder index FOLDER...             index photos (one-off)
ophotofinder search QUERY                search the index
ophotofinder status                      what is in the index
ophotofinder config                      saved defaults and watched folders
ophotofinder storage                     where the index lives and how big it is
ophotofinder upgrade                     re-run only components that are behind
ophotofinder clean                       remove records from the index
ophotofinder web                         run the web UI
ophotofinder web --stop                  stop a background web server
```

`--device mps|cuda|cpu` overrides device auto-detection. It is a global flag, so
it goes *before* the subcommand: `ophotofinder --device cpu index ~/Photos`.

Run `ophotofinder <command> --help` for each command's options.

## Configuration

| Variable | Default |
|---|---|
| `OPHOTOFINDER_HOME` | `~/.ophotofinder` — Chroma DB + thumbnail cache |
| `OLLAMA_HOST` | `http://localhost:11434` — overridden by the saved `ollama_host` setting |
| `OPHOTOFINDER_OLLAMA_TIMEOUT` | `180` seconds |

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `No installed model reports the 'vision' capability` | `ollama pull moondream` |
| Captions all empty, run switches captioning off | The model cannot caption on this Ollama build. Check `ophotofinder models`, then `--vision-model moondream`. |
| `0 indexed` on `~/Pictures` | Sealed `.photoslibrary`; export originals first (see Indexing). |
| `An indexing run is already in progress` | Only one run may write at a time. Wait, or check `ophotofinder status`. |
| `Address already in use` / `Port 7777 is in use` | A server is already running: `ophotofinder web --stop`, or use `--port`. |
| HEIC files skipped | `pip install pillow-heif` |
| No second text reader (non-macOS) | `pip install 'ophotofinder[ocr]'` — works on Linux and Windows too. |
| Nothing found for an obvious query | Confirm captions landed: `ophotofinder status` shows `captions ok=N`. |

`ophotofinder doctor` diagnoses most of these in one command.

## Licence

See [LICENSE](LICENSE).
