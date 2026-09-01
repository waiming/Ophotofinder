"""User settings, persisted as JSON next to the index.

These are *defaults*, not overrides: an explicit CLI flag or API argument always
wins. The point is that the choices a user makes in the Settings tab -- which
models to use, which folders to keep watched -- survive a restart.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from .config import DATA_DIR, ensure_dirs

SETTINGS_PATH = DATA_DIR / "settings.json"

DEFAULTS: dict[str, Any] = {
    # "" means "use OLLAMA_HOST from the environment, else localhost"
    "ollama_host": "",

    # "" means "pick automatically"
    "vision_model": "",          # captions
    "text_model": "",            # the answer step
    "alt_ocr": "auto",           # second text reader
    "caption_enabled": True,
    "ocr_enabled": True,

    # Watched folders: [{"path": str, "recursive": bool}]
    "folders": [],
    "watch_enabled": False,
    "watch_interval_minutes": 30,

    # Re-process photos automatically when the indexing method improves.
    "auto_upgrade": False,
}

_lock = threading.Lock()
_cache: dict[str, Any] | None = None


def load(refresh: bool = False) -> dict[str, Any]:
    global _cache
    if _cache is not None and not refresh:
        return dict(_cache)
    data = dict(DEFAULTS)
    try:
        stored = json.loads(SETTINGS_PATH.read_text())
        if isinstance(stored, dict):
            data.update({k: v for k, v in stored.items() if k in DEFAULTS})
    except (OSError, ValueError):
        pass
    data["folders"] = [f for f in data.get("folders", []) if isinstance(f, dict) and f.get("path")]
    _cache = dict(data)
    return dict(data)


def save(values: dict[str, Any]) -> dict[str, Any]:
    """Merge ``values`` into the stored settings and return the result."""
    global _cache
    with _lock:
        data = load(refresh=True)
        for k, v in values.items():
            if k in DEFAULTS:
                data[k] = v
        ensure_dirs()
        tmp = SETTINGS_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(SETTINGS_PATH)       # atomic, so a crash cannot truncate it
        _cache = dict(data)
        return dict(data)


def get(key: str, fallback: Any = None) -> Any:
    return load().get(key, DEFAULTS.get(key, fallback))


# ------------------------------------------------------------ watched folders

def folders() -> list[dict[str, Any]]:
    return load().get("folders", [])


def add_folder(path: str, recursive: bool = True) -> dict[str, Any]:
    p = str(Path(path).expanduser().resolve())
    current = folders()
    for f in current:
        if f["path"] == p:
            f["recursive"] = bool(recursive)
            return save({"folders": current})
    current.append({"path": p, "recursive": bool(recursive)})
    return save({"folders": current})


def remove_folder(path: str) -> dict[str, Any]:
    p = str(Path(path).expanduser().resolve())
    return save({"folders": [f for f in folders() if f["path"] != p]})


def set_folder_recursive(path: str, recursive: bool) -> dict[str, Any]:
    p = str(Path(path).expanduser().resolve())
    current = folders()
    for f in current:
        if f["path"] == p:
            f["recursive"] = bool(recursive)
    return save({"folders": current})
