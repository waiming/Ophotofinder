"""Open the operating system's own folder chooser.

A browser cannot hand a web page a real filesystem path -- ``<input
webkitdirectory>`` uploads file contents under relative names, and
``showDirectoryPicker()`` returns an opaque handle. Neither can tell the server
which directory to index.

The server, however, runs on the user's own machine, so *it* can open the native
dialog and read back a genuine path. Only ever used for a server bound to
loopback: on a remote server the dialog would appear on the wrong desktop, so it
is refused there and the in-page browser is used instead.
"""
from __future__ import annotations

import platform
import shutil
import subprocess
from pathlib import Path

DIALOG_TIMEOUT = 300        # the user may take a while to choose


class DialogUnavailable(RuntimeError):
    pass


def backend() -> str:
    """Which native chooser this machine can offer, if any."""
    system = platform.system()
    if system == "Darwin" and shutil.which("osascript"):
        return "osascript"
    if system == "Windows":
        return "powershell"
    if system == "Linux":
        for tool in ("zenity", "kdialog", "qarma"):
            if shutil.which(tool):
                return tool
    return ""


def available() -> tuple[bool, str]:
    b = backend()
    if b:
        return True, b
    if platform.system() == "Linux":
        return False, "no zenity/kdialog found (apt install zenity)"
    return False, f"no native folder dialog on {platform.system()}"


def choose_folder(start: str | None = None) -> str | None:
    """Show the OS folder chooser. Returns a path, or None if cancelled."""
    b = backend()
    if not b:
        raise DialogUnavailable(available()[1])

    start_dir = str(Path(start).expanduser()) if start else str(Path.home())

    if b == "osascript":
        script = (
            f'set startFolder to POSIX file "{start_dir}" as alias\n'
            'try\n'
            '  set chosen to choose folder with prompt "Choose a photo folder to index" '
            'default location startFolder\n'
            '  return POSIX path of chosen\n'
            'on error number -128\n'
            '  return ""\n'
            'end try'
        )
        cmd = ["osascript", "-e", script]
    elif b == "powershell":
        ps = (
            "Add-Type -AssemblyName System.Windows.Forms;"
            "$d = New-Object System.Windows.Forms.FolderBrowserDialog;"
            f"$d.SelectedPath = '{start_dir}';"
            "$d.Description = 'Choose a photo folder to index';"
            "if ($d.ShowDialog() -eq 'OK') { Write-Output $d.SelectedPath }"
        )
        cmd = ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps]
    elif b == "kdialog":
        cmd = ["kdialog", "--getexistingdirectory", start_dir]
    else:      # zenity / qarma
        cmd = [b, "--file-selection", "--directory", f"--filename={start_dir}/",
               "--title=Choose a photo folder to index"]

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=DIALOG_TIMEOUT)
    except subprocess.TimeoutExpired:
        raise DialogUnavailable("the folder dialog timed out") from None
    except OSError as e:
        raise DialogUnavailable(f"could not open the dialog: {e}") from None

    path = (r.stdout or "").strip()
    if not path:
        return None                       # cancelled
    return str(Path(path))
