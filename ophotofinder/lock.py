"""A single-holder lock: two concurrent indexing runs corrupt the Chroma index."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .config import LOCK_PATH, ensure_dirs


class IndexBusy(RuntimeError):
    pass


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True


def read_holder() -> dict | None:
    try:
        data = json.loads(LOCK_PATH.read_text())
    except (OSError, ValueError):
        return None
    if not _alive(int(data.get("pid", -1))):
        return None
    return data


class IndexLock:
    """Exclusive O_EXCL lock file; a stale lock from a dead process is reclaimed."""

    def __init__(self, note: str = ""):
        self.note = note
        self.fd = None

    def __enter__(self) -> "IndexLock":
        ensure_dirs()
        holder = read_holder()
        if holder:
            raise IndexBusy(
                f"An indexing run is already in progress (pid {holder.get('pid')}, "
                f"started {holder.get('started')}, {holder.get('note', '')}). "
                f"Only one run may write to the index at a time."
            )
        if LOCK_PATH.exists():
            LOCK_PATH.unlink(missing_ok=True)   # stale, owner is gone
        try:
            self.fd = os.open(str(LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise IndexBusy("An indexing run is already in progress.") from None
        os.write(self.fd, json.dumps({
            "pid": os.getpid(),
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "note": self.note,
        }).encode())
        return self

    def __exit__(self, *exc) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        LOCK_PATH.unlink(missing_ok=True)
