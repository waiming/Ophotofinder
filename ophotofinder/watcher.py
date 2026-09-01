"""Keep the watched folders indexed, in the background.

Polling, not filesystem events: an index run already skips files whose size and
mtime are unchanged, so a periodic pass over the watched folders is cheap and
has no platform-specific dependencies. It also catches changes that happened
while the app was closed, which an event watcher cannot.

Only one indexing run may write at a time, so a tick that finds the lock held
simply skips and waits for the next one.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable

from . import settings
from .lock import IndexBusy, read_holder


def _ago(when: float | None) -> str:
    if not when:
        return ""
    secs = max(0, int(time.time() - when))
    if secs < 60:
        return f"{secs}s ago"
    if secs < 3600:
        return f"{secs // 60}m ago"
    if secs < 86400:
        return f"{secs // 3600}h {(secs % 3600) // 60}m ago"
    return f"{secs // 86400}d ago"


def _until(when: float | None) -> str:
    if not when:
        return ""
    secs = int(when - time.time())
    if secs <= 0:
        return "due now"
    if secs < 60:
        return f"in {secs}s"
    if secs < 3600:
        return f"in {secs // 60}m"
    return f"in {secs // 3600}h {(secs % 3600) // 60}m"


class Watcher:
    """A single background thread that re-indexes watched folders on a timer."""

    def __init__(self, run_index: Callable[..., Any], run_upgrade: Callable[..., Any]):
        self._run_index = run_index
        self._run_upgrade = run_upgrade
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._force = threading.Event()
        # Set while a user-started run wants the lock. The in-flight watch run
        # polls this and stops at the next photo boundary.
        self._yield = threading.Event()
        self.last_run: float | None = None
        self.last_result: str = ""
        self.next_run: float | None = None
        self.active = False

    # ------------------------------------------------------------- lifecycle

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="ophotofinder-watcher")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def poke(self, force: bool = False) -> None:
        """Wake the loop.

        ``force`` runs a pass immediately ("Check now"). Without it the loop only
        re-reads settings and recomputes its timer -- saving an unrelated setting
        must never kick off indexing the user did not ask for.
        """
        if force:
            self._force.set()
        self._wake.set()

    def yield_to_user(self, timeout: float = 30.0) -> bool:
        """Stand aside for a run the user started.

        Returns True once this watcher is no longer indexing. A user-started run
        always takes priority; the watch pass resumes on its next tick and, being
        incremental, loses no work.
        """
        self._yield.set()
        deadline = time.time() + timeout
        while self.active and time.time() < deadline:
            time.sleep(0.2)
        return not self.active

    def resume(self) -> None:
        """Allow watch runs again once the user's run has finished."""
        self._yield.clear()

    def should_stop(self) -> bool:
        return self._yield.is_set() or self._stop.is_set()

    def status(self) -> dict[str, Any]:
        s = settings.load()
        return {
            "enabled": bool(s.get("watch_enabled")),
            "interval_minutes": int(s.get("watch_interval_minutes", 30)),
            "folders": s.get("folders", []),
            "running_now": self.active,
            "last_run": self.last_run,
            "last_run_iso": (time.strftime("%Y-%m-%d %H:%M:%S",
                                           time.localtime(self.last_run))
                             if self.last_run else ""),
            "last_run_ago": _ago(self.last_run),
            "last_result": self.last_result,
            "next_run": self.next_run,
            "next_run_iso": (time.strftime("%Y-%m-%d %H:%M:%S",
                                           time.localtime(self.next_run))
                             if self.next_run else ""),
            "next_run_in": _until(self.next_run),
            "yielding": self._yield.is_set(),
            "auto_upgrade": bool(s.get("auto_upgrade")),
        }

    # ------------------------------------------------------------------ loop

    def _loop(self) -> None:
        # A short initial delay so startup is not competing with the first page load.
        self._wake.wait(20)
        self._wake.clear()
        due = time.time()

        while not self._stop.is_set():
            s = settings.load(refresh=True)
            interval = max(1, int(s.get("watch_interval_minutes", 30))) * 60
            forced = self._force.is_set()
            self._force.clear()

            # Run only when the timer is actually due, or the user asked explicitly.
            if s.get("watch_enabled") and s.get("folders") and (forced or time.time() >= due):
                self._tick(s)
                due = time.time() + interval
            elif not s.get("watch_enabled"):
                due = time.time() + interval

            self.next_run = due if s.get("watch_enabled") else None
            self._wake.wait(max(5, min(interval, max(1, due - time.time()))))
            self._wake.clear()

    def _tick(self, s: dict[str, Any]) -> None:
        if self._yield.is_set():
            self.last_result = "paused while you run an indexing job"
            self.last_run = time.time()
            return
        if read_holder():
            self.last_result = "skipped: another indexing run is in progress"
            self.last_run = time.time()
            return

        folders = [f for f in s.get("folders", []) if Path(f["path"]).is_dir()]
        missing = [f["path"] for f in s.get("folders", []) if not Path(f["path"]).is_dir()]
        if not folders:
            self.last_result = ("no watched folder is reachable"
                                if missing else "no folders to watch")
            self.last_run = time.time()
            return

        self.active = True
        added = 0
        try:
            # Recursive and non-recursive folders need separate runs, since the
            # flag applies to the whole call.
            for recursive in (True, False):
                group = [Path(f["path"]) for f in folders if bool(f.get("recursive", True)) is recursive]
                if not group:
                    continue
                if self._yield.is_set():
                    break
                stats = self._run_index(
                    group,
                    should_stop=self.should_stop,
                    vision_model=s.get("vision_model") or None,
                    caption=bool(s.get("caption_enabled", True)),
                    do_ocr=bool(s.get("ocr_enabled", True)),
                    alt_ocr=s.get("alt_ocr", "auto"),
                    recursive=recursive,
                )
                added += getattr(stats, "indexed", 0)

            if self._yield.is_set():
                self.last_result = ("stood aside for a run you started"
                                    + (f" after {added} photo(s)" if added else ""))
                return
            note = f"{added} new or changed photo(s)"
            if s.get("auto_upgrade"):
                up = self._run_upgrade(should_stop=self.should_stop)
                n = getattr(up, "indexed", 0)
                if n:
                    note += f", {n} upgraded to the current method"
            if missing:
                note += f" ({len(missing)} watched folder(s) unreachable)"
            self.last_result = note
        except IndexBusy:
            self.last_result = "skipped: another indexing run started first"
        except Exception as e:
            self.last_result = f"failed: {type(e).__name__}: {e}"
        finally:
            self.active = False
            self.last_run = time.time()
