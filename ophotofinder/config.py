"""Paths, tunables and the small set of facts we hard-learned about local models."""
from __future__ import annotations

import os
from pathlib import Path

APP_NAME = "Ophotofinder"

# Where the Chroma database and thumbnail cache live.
# Where the data directory's location is remembered. This pointer must live
# OUTSIDE the data directory itself -- settings.json is stored inside it, so a
# location kept there would be lost the moment the directory moved.
LOCATION_POINTER = Path.home() / ".ophotofinder-location"


def _resolve_data_dir() -> Path:
    """OPHOTOFINDER_HOME wins, then the saved pointer, then the default."""
    env = os.environ.get("OPHOTOFINDER_HOME")
    if env:
        return Path(env).expanduser()
    try:
        saved = LOCATION_POINTER.read_text().strip()
        if saved:
            return Path(saved).expanduser()
    except OSError:
        pass
    return Path.home() / ".ophotofinder"


def data_dir_source() -> str:
    if os.environ.get("OPHOTOFINDER_HOME"):
        return "environment"
    try:
        if LOCATION_POINTER.read_text().strip():
            return "chosen"
    except OSError:
        pass
    return "default"


DATA_DIR = _resolve_data_dir()
CHROMA_DIR = DATA_DIR / "chroma"
THUMB_DIR = DATA_DIR / "thumbs"
LOCK_PATH = DATA_DIR / "index.lock"
WEB_PID_PATH = DATA_DIR / "web.pid"

CLIP_COLLECTION = "photo_clip"
TEXT_COLLECTION = "photo_text"

DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"

def normalise_host(raw: str) -> str:
    """Accept 'localhost:11434', '1.2.3.4', or a full URL."""
    raw = (raw or "").strip().rstrip("/")
    if not raw:
        return ""
    if not raw.startswith(("http://", "https://")):
        raw = "http://" + raw
    return raw


# The environment default. A saved setting takes precedence -- see ollama.host().
OLLAMA_HOST = normalise_host(os.environ.get("OLLAMA_HOST", "")) or "http://localhost:11434"

# Extensions we will try to open. HEIC/HEIF need pillow-heif registered.
IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tif", ".tiff",
    ".heic", ".heif", ".avif",
}

THUMB_MAX = 512          # long edge of the cached thumbnail, px
CLIP_BATCH = 8

# A partial batch is also written out once it is this old, so photos become
# visible to the Library (and to search) while a long run is still going,
# instead of waiting for a full batch of CLIP_BATCH.
FLUSH_INTERVAL_SECONDS = 2.0

# Captioning prompts. The long one gives much richer captions on capable VLMs,
# but small models (moondream, ~1.4B) return nothing at all -- or raw grounding
# coordinates like "[0.31, 0.71, 0.64, 0.87]" -- when handed a multi-part prompt.
LONG_CAPTION_PROMPT = (
    "Describe this photo in two or three plain sentences for a search index. "
    "Mention the main subjects, what they are doing, the setting, notable "
    "objects, colours, and whether it is indoors or outdoors. "
    "Write prose only. Do not use lists, headings or coordinates."
)
SHORT_CAPTION_PROMPT = "Describe this image."

# Models at or below this parameter count get the short prompt first.
SMALL_VLM_PARAM_LIMIT = 3_000_000_000

# Vision models known to be broken on current Ollama builds.
# llama3.2-vision fails to load with "unknown model architecture: 'mllama'".
BROKEN_VISION_MODELS = {"llama3.2-vision", "llama3.2-vision:latest"}

# Preference order when the user does not name a model.
PREFERRED_VISION_MODELS = ["moondream", "qwen2.5vl", "qwen2-vl", "llava", "bakllava", "minicpm-v", "gemma3"]
PREFERRED_TEXT_MODELS = ["llama3.2", "llama3.1", "qwen2.5", "mistral", "gemma3", "phi3"]

# If this many photos in a row caption empty at the start of a run, captioning
# is switched off for the rest of the run and the reason is reported loudly.
CAPTION_FAILURE_STREAK = 5

OLLAMA_TIMEOUT = float(os.environ.get("OPHOTOFINDER_OLLAMA_TIMEOUT", "180"))

# Last-resort num_ctx when a model publishes no <arch>.context_length at all.
# Every Ollama call sends an explicit num_ctx; this is only reached when the
# model metadata is silent, and it is reported when it happens.
FALLBACK_NUM_CTX = 4096

# Languages for the second text reader (Apple Vision). Order is a preference
# hint; every one of these is recognised on-device with no download.
ALT_OCR_LANGS = [
    s for s in os.environ.get(
        "OPHOTOFINDER_ALT_OCR_LANGS", "en-US,zh-Hant,zh-Hans,ja-JP"
    ).split(",") if s.strip()
]


def ensure_dirs() -> None:
    for d in (DATA_DIR, CHROMA_DIR, THUMB_DIR):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Pipeline versioning
#
# Each signal is produced by an independent component, and each component has
# its own version. Bump one when its *output would change* for the same photo:
# a better model, a different prompt, a bug fix in extraction. Records store the
# version they were built with, so `ophotofinder upgrade` can find exactly which
# photos are behind and re-run *only* the stale components -- re-reading EXIF is
# cheap, re-captioning with a VLM is not.
#
# History:
#   exif     1 -> 2  mtime fallback applied at every exit; date_source recorded
#   ocr      1 -> 2  single-word signs accepted ("STAIRS", "INVOICE")
#   ocr_alt  0 -> 1  second text reader added (Apple Vision / RapidOCR)
#   document 1 -> 2  second reader's text folded into the embedded document
# --------------------------------------------------------------------------

PIPELINE_VERSIONS: dict[str, int] = {
    "clip": 1,
    "caption": 1,
    "ocr": 2,
    "ocr_alt": 1,
    "exif": 2,
    "document": 2,
}

# Components whose staleness forces the document to be rebuilt and re-embedded.
DOCUMENT_INPUTS = ("caption", "ocr", "ocr_alt", "exif")

# Roughly how expensive each component is to recompute, for reporting.
COMPONENT_COST = {
    "clip": "medium", "caption": "expensive", "ocr": "medium",
    "ocr_alt": "medium", "exif": "cheap", "document": "cheap",
}


def version_key(component: str) -> str:
    return f"v_{component}"


def pipeline_signature() -> str:
    return ".".join(f"{k}{v}" for k, v in sorted(PIPELINE_VERSIONS.items()))
