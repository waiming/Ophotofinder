"""Tesseract OCR over the already-upright image."""
from __future__ import annotations

import re
import shutil

from PIL import Image

_WORD = re.compile(r"[A-Za-z]{3,}")

TESSERACT_OK = shutil.which("tesseract") is not None


def ocr_image(img: Image.Image, min_words: int = 1) -> str:
    """Return cleaned OCR text, or '' when there is nothing meaningful.

    ``img`` must already have had ``ImageOps.exif_transpose`` applied -- a
    sideways photo makes Tesseract emit long strings of nonsense.
    """
    if not TESSERACT_OK:
        return ""
    try:
        import pytesseract

        work = img
        if max(work.size) > 1600:
            work = work.copy()
            work.thumbnail((1600, 1600), Image.LANCZOS)
        text = pytesseract.image_to_string(work)
    except Exception:
        return ""

    lines = [" ".join(ln.split()) for ln in (text or "").splitlines()]
    cleaned = " ".join(ln for ln in lines if ln)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    # Signs and labels are often a single word ("STAIRS", "INVOICE"), so one
    # real word is enough -- but it must be a word, not stray character noise.
    if len(_WORD.findall(cleaned)) < min_words or len(cleaned) < 3:
        return ""
    return cleaned[:2000]
