"""Track a running web server so it can be stopped again.

The server records its pid, host and port on start and clears the file on exit.
Stopping only ever signals a pid this file recorded -- never whatever happens to
hold the port, since that could be an unrelated program.
"""
from __future__ import annotations

import atexit
import json
import os
import signal
import time

from .config import WEB_PID_PATH, ensure_dirs


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True


def read() -> dict | None:
    """The recorded server, or None if absent or stale."""
    try:
        data = json.loads(WEB_PID_PATH.read_text())
    except (OSError, ValueError):
        return None
    if not _alive(int(data.get("pid", -1))):
        WEB_PID_PATH.unlink(missing_ok=True)      # stale, owner is gone
        return None
    return data


def write(host: str, port: int) -> None:
    ensure_dirs()
    WEB_PID_PATH.write_text(json.dumps({
        "pid": os.getpid(), "host": host, "port": port,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }))

    def cleanup(*_):
        WEB_PID_PATH.unlink(missing_ok=True)

    atexit.register(cleanup)
    for sig in (signal.SIGTERM, signal.SIGINT):
        prev = signal.getsignal(sig)

        def handler(signum, frame, _prev=prev):
            cleanup()
            if callable(_prev):
                _prev(signum, frame)
            else:
                raise SystemExit(0)

        try:
            signal.signal(sig, handler)
        except ValueError:      # not on the main thread
            pass


def stop(port: int | None = None, timeout: float = 8.0) -> tuple[bool, str]:
    """Stop the recorded server. Returns (stopped, message)."""
    rec = read()
    if rec and port and str(rec.get("port")) != str(port):
        rec = None                      # a different server was asked for
    if not rec:
        # No pid file: look for a real `ophotofinder web` process instead.
        others = discover(port)
        if not others:
            return False, "no running Ophotofinder web server found"
        if len(others) > 1:
            listing = ", ".join(f"pid {o['pid']} on {o['addr']}" for o in others)
            return False, (f"several Ophotofinder web servers are running ({listing}). "
                           f"Stop one with: ophotofinder web --stop --port <port>")
        rec = {"pid": others[0]["pid"], "host": "", "port": others[0]["port"]}

    pid = int(rec["pid"])
    where = f"pid {pid} on {rec.get('host') or '127.0.0.1'}:{rec.get('port')}"
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        WEB_PID_PATH.unlink(missing_ok=True)
        return True, f"server was already gone ({where})"
    except PermissionError:
        return False, f"not permitted to stop {where}"

    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _alive(pid):
            WEB_PID_PATH.unlink(missing_ok=True)
            return True, f"stopped {where}"
        time.sleep(0.2)

    try:
        os.kill(pid, signal.SIGKILL)      # did not go quietly
        time.sleep(0.4)
    except ProcessLookupError:
        pass
    WEB_PID_PATH.unlink(missing_ok=True)
    return True, f"force-stopped {where} (it ignored SIGTERM)"


def port_holder(port: int) -> str:
    """Describe whatever is listening on a port, for a useful error message."""
    import subprocess

    try:
        out = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip().splitlines()
    except Exception:
        return ""
    if len(out) < 2:
        return ""
    parts = out[1].split()
    return f"{parts[0]} (pid {parts[1]})" if len(parts) > 1 else ""

def _cmdline(pid: int) -> str:
    import subprocess
    try:
        return subprocess.run(["ps", "-p", str(pid), "-o", "command="],
                              capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return ""


def discover(port: int | None = None) -> list[dict]:
    """Find running Ophotofinder web servers that left no pid file.

    Servers started before this pid file existed, or from another shell, are
    still stoppable: listening pids are matched against their command line, so
    only an actual `ophotofinder web` process is ever a candidate. An unrelated
    program on the same port is never touched.
    """
    import subprocess

    args = ["lsof", "-nP", "-iTCP", "-sTCP:LISTEN"]
    if port:
        args = ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"]
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=6).stdout
    except Exception:
        return []

    found: dict[int, dict] = {}
    for line in out.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 9:
            continue
        try:
            pid = int(parts[1])
        except ValueError:
            continue
        if pid in found or pid == os.getpid():
            continue
        cmd = _cmdline(pid)
        if "ophotofinder" not in cmd or " web" not in f" {cmd} ":
            continue
        addr = parts[8]
        found[pid] = {"pid": pid, "addr": addr,
                      "port": addr.rsplit(":", 1)[-1], "cmd": cmd}
    return list(found.values())

