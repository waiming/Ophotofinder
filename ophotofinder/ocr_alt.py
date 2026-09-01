"""A second, independent text reader.

Tesseract is fast and good at flat, high-contrast, Latin-script text (screenshots,
scans, documents). It is weak at text photographed in the real world -- angled,
curved, lit unevenly, on a sign across the street -- and it can only read the
language packs installed, which is usually English alone.

This module adds a second reader whose failures are uncorrelated with Tesseract's,
stored in its own field so neither overwrites the other:

* ``apple``  -- Apple's Vision framework (macOS). On-device, no download, and it
  reads 30 languages including Traditional/Simplified Chinese, Cantonese,
  Japanese and Korean. Much stronger on real-world photographs.
* ``rapidocr`` -- PP-OCRv3 via ONNX Runtime. Cross-platform (Linux, Windows,
  macOS), models ship inside the wheel so it stays fully offline, and it reads
  Chinese and English out of the box.
* ``vlm``    -- ask the local vision model to transcribe the text. Only useful
  with a capable VLM; small models return junk, which is rejected here.

``auto`` picks the best reader present: Apple Vision on macOS, else RapidOCR,
else a capable VLM, else nothing. Non-macOS users get a working second reader by
installing the ``ocr`` extra; nothing here is macOS-only by design.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass

from PIL import Image

from .config import ALT_OCR_LANGS

_WORD = re.compile(r"[A-Za-z]{3,}")
_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")
_COORDS = re.compile(r"^[\s\[\]\(\),.;:0-9-]+$")


@dataclass
class AltText:
    text: str = ""
    method: str = ""            # apple | vlm | ""
    confidence: float = 0.0
    detail: str = ""


# ------------------------------------------------------------------ Apple Vision

_vision_error = ""


def apple_available() -> tuple[bool, str]:
    """Is Apple's Vision text recogniser usable here?"""
    global _vision_error
    import platform

    if platform.system() != "Darwin":
        return False, "not macOS"
    try:
        import Quartz  # noqa: F401
        import Vision  # noqa: F401
        from Foundation import NSData  # noqa: F401
    except ImportError:
        _vision_error = ("pyobjc not installed: "
                         "pip install pyobjc-framework-Vision pyobjc-framework-Quartz")
        return False, _vision_error
    return True, ""


def apple_languages() -> list[str]:
    ok, _ = apple_available()
    if not ok:
        return []
    import Vision

    langs = Vision.VNRecognizeTextRequest.alloc().init() \
        .supportedRecognitionLanguagesAndReturnError_(None)
    return list(langs[0]) if langs and langs[0] else []


def read_apple(img: Image.Image, languages: list[str] | None = None) -> AltText:
    ok, why = apple_available()
    if not ok:
        return AltText(method="apple", detail=why)
    import Quartz
    import Vision
    from Foundation import NSData

    buf = io.BytesIO()
    img.save(buf, "PNG")            # lossless: recognition is sensitive to artefacts
    raw = buf.getvalue()
    data = NSData.dataWithBytes_length_(raw, len(raw))
    src = Quartz.CGImageSourceCreateWithData(data, None)
    if src is None:
        return AltText(method="apple", detail="could not decode image")
    cg = Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)
    if cg is None:
        return AltText(method="apple", detail="could not decode image")

    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(0)                     # 0 = accurate, 1 = fast
    req.setUsesLanguageCorrection_(True)

    # Language order matters enormously: forcing "en-US" first reads a Chinese
    # screenshot as gibberish ("Now #", "#ili"), while forcing "zh-Hant" first
    # wrecks English text (confidence 1.00 -> 0.36). Automatic detection matched
    # the best fixed order on both, so it is the default; an explicit list from
    # the user wins, and a fixed list is the fallback where auto is unsupported.
    used_auto = False
    if not languages:
        try:
            req.setAutomaticallyDetectsLanguage_(True)
            used_auto = True
        except Exception:
            used_auto = False
    if not used_auto:
        req.setRecognitionLanguages_(list(languages or ALT_OCR_LANGS))

    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg, None)
    ok2, err = handler.performRequests_error_([req], None)
    if not ok2:
        return AltText(method="apple", detail=f"vision error: {err}")

    lines, confs = [], []
    for obs in (req.results() or []):
        cand = obs.topCandidates_(1)
        if cand and len(cand):
            s = cand[0].string()
            if s and s.strip():
                lines.append(s.strip())
                confs.append(float(cand[0].confidence()))
    text = "\n".join(lines).strip()
    conf = sum(confs) / len(confs) if confs else 0.0
    if not _meaningful(text):
        return AltText(method="apple", confidence=conf, detail="no readable text")
    return AltText(text=text[:4000], method="apple", confidence=round(conf, 3),
                   detail="auto-detect" if used_auto else "langs=" + ",".join(languages or ALT_OCR_LANGS))


# --------------------------------------------------------------------- RapidOCR

_rapid = None
_rapid_error = ""


def rapidocr_available() -> tuple[bool, str]:
    try:
        import rapidocr_onnxruntime  # noqa: F401
        return True, ""
    except ImportError:
        try:
            import rapidocr  # noqa: F401
            return True, ""
        except ImportError:
            return False, ("rapidocr not installed: "
                           "pip install rapidocr-onnxruntime")


def _rapid_engine():
    """One shared engine -- construction loads three ONNX models."""
    global _rapid, _rapid_error
    if _rapid is not None or _rapid_error:
        return _rapid
    try:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError:
            from rapidocr import RapidOCR
        _rapid = RapidOCR()
    except Exception as e:
        _rapid_error = f"{type(e).__name__}: {e}"
        _rapid = None
    return _rapid


def read_rapidocr(img: Image.Image, max_edge: int = 2000) -> AltText:
    ok, why = rapidocr_available()
    if not ok:
        return AltText(method="rapidocr", detail=why)
    engine = _rapid_engine()
    if engine is None:
        return AltText(method="rapidocr", detail=_rapid_error or "engine unavailable")

    import numpy as np

    work = img
    if max(work.size) > max_edge:
        work = work.copy()
        work.thumbnail((max_edge, max_edge), Image.LANCZOS)
    try:
        result, _ = engine(np.array(work))
    except Exception as e:
        return AltText(method="rapidocr", detail=f"{type(e).__name__}: {e}")

    lines, confs = [], []
    for row in (result or []):
        # rows are (box, text, confidence)
        if len(row) >= 3 and str(row[1]).strip():
            lines.append(str(row[1]).strip())
            try:
                confs.append(float(row[2]))
            except (TypeError, ValueError):
                pass
    text = "\n".join(lines).strip()
    conf = sum(confs) / len(confs) if confs else 0.0
    if not _meaningful(text):
        return AltText(method="rapidocr", confidence=conf, detail="no readable text")
    return AltText(text=text[:4000], method="rapidocr", confidence=round(conf, 3))


# ------------------------------------------------------------------------- VLM

VLM_OCR_PROMPT = (
    "Transcribe every piece of text visible in this image, exactly as written, "
    "preserving the original language. Output only the transcribed text, nothing else."
)


def read_vlm(img: Image.Image, model, num_ctx: int | None = None) -> AltText:
    """Ask the vision model to transcribe. Junk answers are rejected, not stored."""
    from . import ollama
    from .imaging import to_jpeg_bytes

    try:
        out = ollama.generate(
            model, VLM_OCR_PROMPT, images=[to_jpeg_bytes(img, max_edge=1024)],
            num_ctx=num_ctx, num_predict=400,
        ).strip()
    except Exception as e:
        return AltText(method="vlm", detail=f"{type(e).__name__}: {e}")

    # Small models answer an OCR prompt with placeholders or grounding boxes.
    if (not out or out.strip("!") .strip().upper() == "IMAGE"
            or _COORDS.match(out) or not _meaningful(out)):
        return AltText(method="vlm", detail=f"unusable reply: {out[:60]!r}")
    return AltText(text=out[:4000], method="vlm")


# ---------------------------------------------------------------------- shared

def _meaningful(text: str) -> bool:
    """Real words in any script, rather than punctuation noise."""
    t = (text or "").strip()
    if len(t) < 2:
        return False
    return bool(_WORD.search(t) or _CJK.search(t) or re.search(r"\d{2,}", t))


def read(img: Image.Image, method: str = "auto", languages: list[str] | None = None,
         vlm_model=None, vlm_num_ctx: int | None = None) -> AltText:
    """Run the chosen second reader. ``auto`` prefers Apple Vision when present."""
    if method in ("none", "off", ""):
        return AltText(detail="disabled")
    if method == "auto":
        method = best_available(bool(vlm_model))
        if method == "none":
            return AltText(detail="no second reader available")
    if method == "apple":
        return read_apple(img, languages)
    if method == "rapidocr":
        return read_rapidocr(img)
    if method == "vlm":
        if not vlm_model:
            return AltText(method="vlm", detail="no vision model available")
        return read_vlm(img, vlm_model, vlm_num_ctx)
    return AltText(detail=f"unknown method {method!r}")


def best_available(have_vlm: bool = False) -> str:
    """The strongest second reader present on this machine."""
    if apple_available()[0]:
        return "apple"
    if rapidocr_available()[0]:
        return "rapidocr"
    return "vlm" if have_vlm else "none"


def describe_available() -> list[tuple[str, bool, str]]:
    """(name, usable, note) for every backend -- used by `doctor`."""
    ok_a, why_a = apple_available()
    ok_r, why_r = rapidocr_available()
    return [
        ("apple", ok_a, "Apple Vision, 30 languages, macOS only" if ok_a else why_a),
        ("rapidocr", ok_r, "PP-OCRv3 ONNX, cross-platform, Chinese + English"
         if ok_r else why_r),
    ]
