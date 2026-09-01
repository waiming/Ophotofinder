"""Image loading, EXIF extraction and thumbnails.

Two non-negotiables live here:

* ``pillow_heif`` is registered at import time -- iPhone photos are HEIC and
  Pillow cannot open them otherwise.
* Every image is passed through ``ImageOps.exif_transpose`` immediately after
  opening. Without it every portrait photo is handed to OCR, CLIP and the
  thumbnailer rotated 90 degrees; OCR in particular returns pure gibberish.
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from .config import IMAGE_EXTS, THUMB_DIR, THUMB_MAX

HEIF_OK = False
try:
    import pillow_heif

    pillow_heif.register_heif_opener()
    # AVIF lives behind a separate registration on newer pillow-heif builds.
    if hasattr(pillow_heif, "register_avif_opener"):
        try:
            pillow_heif.register_avif_opener()
        except Exception:
            pass
    HEIF_OK = True
except Exception:  # pragma: no cover - environment dependent
    HEIF_OK = False

Image.MAX_IMAGE_PIXELS = 400_000_000


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def open_upright(path: Path) -> Image.Image:
    """Open an image and apply the EXIF orientation. Always use this."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def file_signature(path: Path) -> str:
    """Cheap change-detector: path + size + mtime. Used as the record id seed."""
    st = path.stat()
    raw = f"{path.resolve()}|{st.st_size}|{int(st.st_mtime)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def path_id(path: Path) -> str:
    return hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()


def thumbnail_path(path: Path) -> Path:
    return THUMB_DIR / f"{path_id(path)}.jpg"


def write_thumbnail(img: Image.Image, path: Path) -> Path:
    out = thumbnail_path(path)
    thumb = img.copy()
    thumb.thumbnail((THUMB_MAX, THUMB_MAX), Image.LANCZOS)
    out.parent.mkdir(parents=True, exist_ok=True)
    thumb.save(out, "JPEG", quality=82)
    return out


def to_jpeg_bytes(img: Image.Image, max_edge: int = 896, quality: int = 88) -> bytes:
    """Downscaled JPEG for the VLM. Big base64 payloads are slow and blow the context."""
    work = img.copy()
    work.thumbnail((max_edge, max_edge), Image.LANCZOS)
    buf = io.BytesIO()
    work.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


# --------------------------------------------------------------------------- EXIF

@dataclass
class Exif:
    taken_at: str = ""          # ISO-8601, empty when unknown
    year: str = ""
    month: str = ""
    camera: str = ""
    lens: str = ""
    gps_lat: float | None = None
    gps_lon: float | None = None
    width: int = 0
    height: int = 0
    orientation: str = ""
    date_source: str = ""       # "exif" | "mtime" | "" (unknown)
    raw: dict[str, Any] = field(default_factory=dict)

    def as_text(self) -> str:
        bits = []
        if self.taken_at:
            try:
                dt = datetime.fromisoformat(self.taken_at)
                bits.append(f"Taken {dt.strftime('%A %d %B %Y at %H:%M')}.")
            except ValueError:
                bits.append(f"Taken {self.taken_at}.")
        if self.camera:
            bits.append(f"Camera: {self.camera}.")
        if self.lens:
            bits.append(f"Lens: {self.lens}.")
        if self.gps_lat is not None and self.gps_lon is not None:
            bits.append(f"Location: {self.gps_lat:.5f}, {self.gps_lon:.5f}.")
        if self.orientation:
            bits.append(f"{self.orientation} orientation.")
        return " ".join(bits)


_EXIF_TAGS = {
    271: "Make", 272: "Model", 306: "DateTime",
    36867: "DateTimeOriginal", 36868: "DateTimeDigitized",
    42036: "LensModel", 34853: "GPSInfo", 274: "Orientation",
}


def _to_deg(value) -> float | None:
    try:
        d, m, s = (float(x) for x in value)
        return d + m / 60.0 + s / 3600.0
    except Exception:
        return None


def _with_mtime_fallback(ex: "Exif", path: Path) -> "Exif":
    """Date photos with no usable EXIF from the filesystem mtime.

    Applied at *every* exit from ``read_exif`` -- plenty of real photos (PNGs,
    screenshots, anything re-encoded) carry no EXIF block at all, and without a
    date they would drop out of date sorting and year filters entirely.
    """
    if not ex.taken_at:
        try:
            dt = datetime.fromtimestamp(path.stat().st_mtime)
            ex.taken_at = dt.isoformat()
            ex.year, ex.month = str(dt.year), f"{dt.month:02d}"
            ex.date_source = "mtime"
        except Exception:
            pass
    return ex


def read_exif(path: Path, img: Image.Image | None = None) -> Exif:
    ex = Exif()
    try:
        src = img if img is not None else Image.open(path)
        ex.width, ex.height = src.size
        ex.orientation = "portrait" if src.height > src.width else (
            "square" if src.height == src.width else "landscape")
        raw = src.getexif()
    except Exception:
        return _with_mtime_fallback(ex, path)
    if not raw:
        return _with_mtime_fallback(ex, path)

    vals: dict[str, Any] = {}
    for tag, name in _EXIF_TAGS.items():
        if tag in raw:
            vals[name] = raw.get(tag)

    make = str(vals.get("Make", "") or "").strip().strip("\x00")
    model = str(vals.get("Model", "") or "").strip().strip("\x00")
    if model and make and not model.lower().startswith(make.lower()):
        ex.camera = f"{make} {model}"
    else:
        ex.camera = model or make
    ex.lens = str(vals.get("LensModel", "") or "").strip().strip("\x00")

    for key in ("DateTimeOriginal", "DateTimeDigitized", "DateTime"):
        val = vals.get(key)
        if not val:
            continue
        try:
            dt = datetime.strptime(str(val).strip(), "%Y:%m:%d %H:%M:%S")
        except ValueError:
            continue
        ex.taken_at = dt.isoformat()
        ex.year, ex.month = str(dt.year), f"{dt.month:02d}"
        ex.date_source = "exif"
        break

    gps = vals.get("GPSInfo")
    if isinstance(gps, dict) or hasattr(gps, "items"):
        try:
            g = dict(gps)
            lat, lon = _to_deg(g.get(2)), _to_deg(g.get(4))
            if lat is not None and lon is not None:
                if str(g.get(1, "N")).upper().startswith("S"):
                    lat = -lat
                if str(g.get(3, "E")).upper().startswith("W"):
                    lon = -lon
                ex.gps_lat, ex.gps_lon = round(lat, 6), round(lon, 6)
        except Exception:
            pass

    return _with_mtime_fallback(ex, path)
