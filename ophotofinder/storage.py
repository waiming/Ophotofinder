"""Where the index lives, how big it is, and moving it elsewhere."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from .config import CHROMA_DIR, DATA_DIR, LOCATION_POINTER, THUMB_DIR, data_dir_source


def dir_size(path: Path) -> tuple[int, int]:
    """(bytes, file count) for a directory tree; missing dirs count as zero."""
    total = files = 0
    if not path.exists():
        return 0, 0
    for root, _dirs, names in os.walk(path, onerror=lambda e: None):
        for n in names:
            try:
                total += (Path(root) / n).stat().st_size
                files += 1
            except OSError:
                continue
    return total, files


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


def report() -> dict:
    """Size breakdown of the data directory, plus where it is and why."""
    parts = []
    for label, path in (("vector database", CHROMA_DIR), ("thumbnails", THUMB_DIR)):
        size, files = dir_size(path)
        parts.append({"label": label, "path": str(path), "bytes": size,
                      "human": human(size), "files": files})
    total, files = dir_size(DATA_DIR)
    other = total - sum(p["bytes"] for p in parts)
    if other > 0:
        parts.append({"label": "other", "path": str(DATA_DIR), "bytes": other,
                      "human": human(other), "files": 0})

    free = 0
    try:
        free = shutil.disk_usage(DATA_DIR if DATA_DIR.exists() else Path.home()).free
    except OSError:
        pass

    return {
        "path": str(DATA_DIR),
        "source": data_dir_source(),
        "total_bytes": total, "total_human": human(total), "total_files": files,
        "free_bytes": free, "free_human": human(free),
        "parts": parts,
    }


def set_location(new_dir: str) -> Path:
    """Remember a new data directory for next start. Does not move anything."""
    target = Path(new_dir).expanduser()
    target.mkdir(parents=True, exist_ok=True)
    if not os.access(target, os.W_OK):
        raise PermissionError(f"{target} is not writable")
    LOCATION_POINTER.write_text(str(target.resolve()))
    return target.resolve()


def clear_location() -> None:
    LOCATION_POINTER.unlink(missing_ok=True)


def move_data(new_dir: str) -> tuple[Path, str]:
    """Move the index to ``new_dir`` and point at it.

    Only safe when nothing holds the database open, so this is a CLI operation:
    a running server keeps Chroma's SQLite files open, and moving them under it
    would corrupt the index.
    """
    src = DATA_DIR
    dst = Path(new_dir).expanduser().resolve()
    if dst == src.resolve():
        return dst, "already there"

    dst.mkdir(parents=True, exist_ok=True)
    if any(dst.iterdir()):
        raise FileExistsError(f"{dst} is not empty - choose an empty folder")

    moved = []
    if src.exists():
        for item in src.iterdir():
            shutil.move(str(item), str(dst / item.name))
            moved.append(item.name)
        try:
            src.rmdir()
        except OSError:
            pass                      # something else lives there; leave it

    LOCATION_POINTER.write_text(str(dst))
    return dst, f"moved {', '.join(moved) if moved else 'nothing (index was empty)'}"
