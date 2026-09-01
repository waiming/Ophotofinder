"""Walking photo folders, and explaining what we found when it is nothing."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .config import IMAGE_EXTS

# macOS bundles that look like directories but are sealed application packages.
BUNDLE_SUFFIXES = {
    ".photoslibrary", ".photolibrary", ".aplibrary", ".migratedaplibrary",
    ".app", ".fcpbundle", ".imovielibrary", ".theater", ".lrdata", ".lrcat",
}
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".Trash", "Library"}


def is_bundle(path: Path) -> bool:
    return path.suffix.lower() in BUNDLE_SUFFIXES


@dataclass
class ScanReport:
    roots: list[Path] = field(default_factory=list)
    files: list[Path] = field(default_factory=list)
    bundles: list[Path] = field(default_factory=list)
    unreadable: list[Path] = field(default_factory=list)

    def explain_empty(self) -> str:
        """A truthful message for the 'found nothing' case, instead of '0 indexed'."""
        lines = []
        if self.bundles:
            names = ", ".join(b.name for b in self.bundles[:4])
            lines.append(
                f"No loose image files here. This folder contains only sealed photo "
                f"library package(s): {names}. macOS keeps ~/Pictures as a "
                f"'.photoslibrary' bundle that apps cannot read directly."
            )
            lines.append(
                "To index these photos, open Photos.app and use File > Export > "
                "Export Unmodified Originals into a normal folder (e.g. "
                "~/Pictures/Exported), then point ophotofinder at that folder."
            )
        if self.unreadable:
            lines.append(
                f"{len(self.unreadable)} folder(s) could not be read (permissions). "
                f"macOS may need Full Disk Access for your terminal: System Settings > "
                f"Privacy & Security > Full Disk Access."
            )
        if not lines:
            exts = ", ".join(sorted(e.lstrip('.') for e in IMAGE_EXTS))
            lines.append(f"No files with a known image extension ({exts}) were found.")
        return " ".join(lines)


def scan(roots: list[Path], recursive: bool = True, follow_symlinks: bool = False) -> ScanReport:
    report = ScanReport(roots=list(roots))
    seen: set[Path] = set()

    for root in roots:
        root = root.expanduser()
        if not root.exists():
            report.unreadable.append(root)
            continue
        if root.is_file():
            if root.suffix.lower() in IMAGE_EXTS:
                report.files.append(root.resolve())
            continue

        for dirpath, dirnames, filenames in os.walk(root, followlinks=follow_symlinks,
                                                    onerror=lambda e: report.unreadable.append(
                                                        Path(getattr(e, "filename", root)))):
            here = Path(dirpath)
            keep = []
            for d in dirnames:
                child = here / d
                if is_bundle(child):
                    report.bundles.append(child)
                    continue
                if d.startswith(".") or d in SKIP_DIRS:
                    continue
                keep.append(d)
            dirnames[:] = keep if recursive else []

            for fn in filenames:
                if fn.startswith("."):
                    continue
                p = here / fn
                if p.suffix.lower() in IMAGE_EXTS:
                    rp = p.resolve()
                    if rp not in seen:
                        seen.add(rp)
                        report.files.append(rp)

    report.files.sort()
    return report


def picker_roots() -> list[Path]:
    """The only places the web folder picker is allowed to look.

    On macOS the boot volume appears under /Volumes as a firmlink that resolves
    to "/" -- accepting it would silently widen the picker to the whole
    filesystem, so any volume resolving to the root is dropped and only the
    home directory plus genuinely separate mounts remain.
    """
    roots = [Path.home()]
    vols = Path("/Volumes")
    if vols.is_dir():
        try:
            for p in sorted(vols.iterdir()):
                if not p.is_dir():
                    continue
                try:
                    if p.resolve() == Path("/"):
                        continue          # boot-volume firmlink, not a real mount
                except OSError:
                    continue
                roots.append(p)
        except PermissionError:
            pass
    return roots


def within_picker_roots(path: Path) -> bool:
    try:
        target = path.expanduser().resolve()
    except OSError:
        return False
    for root in picker_roots():
        try:
            target.relative_to(root.resolve())
            return True
        except ValueError:
            continue
    return False


def list_subdirs(path: Path, limit: int = 400) -> list[dict]:
    """Directory listing for the picker, bundles marked rather than hidden."""
    out = []
    try:
        entries = sorted(path.iterdir(), key=lambda p: p.name.lower())
    except (PermissionError, OSError):
        return out
    for p in entries:
        if p.name.startswith(".") or not p.is_dir():
            continue
        out.append({"name": p.name, "path": str(p), "bundle": is_bundle(p)})
        if len(out) >= limit:
            break
    return out
