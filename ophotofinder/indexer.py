"""The indexing run: three independent signals per photo, two Chroma collections."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import ollama
from .caption import Captioner
from .clip_embed import ClipEmbedder
from .config import (ALT_OCR_LANGS, CLIP_BATCH, DOCUMENT_INPUTS, FLUSH_INTERVAL_SECONDS,
                     PIPELINE_VERSIONS, ensure_dirs, pipeline_signature, version_key)
from .imaging import (HEIF_OK, Exif, file_signature, open_upright, path_id,
                      read_exif, write_thumbnail)
from .lock import IndexLock
from . import ocr_alt
from .ocr import TESSERACT_OK, ocr_image
from .scan import ScanReport, scan
from .store import Store
from .text_embed import TextEmbedder

Progress = Callable[[dict], None]


@dataclass
class IndexStats:
    found: int = 0
    indexed: int = 0
    skipped_unchanged: int = 0
    failed: int = 0
    caption_ok: int = 0
    caption_empty: int = 0
    caption_skipped: int = 0
    ocr_hits: int = 0
    alt_ocr_hits: int = 0
    started: float = field(default_factory=time.time)
    notes: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    stopped_early: bool = False
    scan_report: ScanReport | None = None

    @property
    def elapsed(self) -> float:
        return time.time() - self.started

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "scan_report"}
        d["elapsed"] = round(self.elapsed, 1)
        return d


def _clean_meta(d: dict) -> dict:
    """Chroma only accepts scalar metadata values, and never None."""
    out = {}
    for k, v in d.items():
        if v is None or v == "":
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def build_document(caption: str, ocr: str, exif_text: str, path: Path,
                   ocr_alt: str = "") -> str:
    parts = []
    if caption:
        parts.append(caption)
    if ocr:
        parts.append(f"Text visible in the photo: {ocr}")
    if ocr_alt and ocr_alt.strip() != (ocr or "").strip():
        parts.append(f"Text read by the second reader: {ocr_alt}")
    if exif_text:
        parts.append(exif_text)
    parts.append(f"File: {path.name} in folder {path.parent.name}.")
    return "\n".join(parts)


def index_folders(
    roots: list[Path],
    *,
    vision_model: str | None = None,
    caption: bool = True,
    do_ocr: bool = True,
    alt_ocr: str = "auto",
    alt_langs: list[str] | None = None,
    recursive: bool = True,
    reindex: bool = False,
    limit: int | None = None,
    clip_model: str | None = None,
    device: str | None = None,
    progress: Progress | None = None,
    store: Store | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> IndexStats:
    """Index every image under ``roots``. Holds the exclusive index lock.

    ``should_stop`` is polled between photos. A background watch run passes one
    so a user-started run can take over promptly: everything already processed
    is flushed and kept, and the next run picks up where this one left off.
    """
    ensure_dirs()
    stats = IndexStats()

    def emit(**kw):
        if progress:
            progress(kw)

    with IndexLock(note=f"indexing {', '.join(str(r) for r in roots)}"):
        report = scan([Path(r) for r in roots], recursive=recursive)
        stats.scan_report = report
        stats.found = len(report.files)

        if not HEIF_OK:
            stats.notes.append(
                "pillow-heif is not available: HEIC/HEIF (iPhone) photos will be skipped."
            )
        if do_ocr and not TESSERACT_OK:
            do_ocr = False
            stats.notes.append("tesseract binary not found: OCR disabled for this run.")

        alt_method = alt_ocr
        if alt_method == "auto":
            alt_method = ocr_alt.best_available(have_vlm=caption)
            if alt_method == "none":
                stats.notes.append(
                    "no second text reader available on this machine "
                    "(install one with: pip install 'ophotofinder[ocr]')")
        if alt_method == "apple":
            ok_alt, why_alt = ocr_alt.apple_available()
            if ok_alt:
                stats.notes.append(
                    "second text reader: Apple Vision ("
                    + (", ".join(alt_langs) if alt_langs else "automatic language detection")
                    + ")")
            else:
                alt_method = "none"
                stats.notes.append(f"second text reader unavailable: {why_alt}")
        elif alt_method == "rapidocr":
            ok_alt, why_alt = ocr_alt.rapidocr_available()
            if ok_alt:
                stats.notes.append("second text reader: RapidOCR (PP-OCRv3, Chinese + English)")
            else:
                alt_method = "none"
                stats.notes.append(f"second text reader unavailable: {why_alt}")
        elif alt_method == "vlm":
            stats.notes.append("second text reader: local vision model transcription")
        elif alt_method in ("none", "off", ""):
            alt_method = "none"

        if not report.files:
            stats.notes.append(report.explain_empty())
            emit(event="done", **stats.as_dict())
            return stats

        files = report.files[:limit] if limit else report.files

        # ---- models ------------------------------------------------------
        captioner = None
        if caption:
            try:
                info = ollama.pick_vision_model(vision_model)
                captioner = Captioner(info)
                stats.notes.append(
                    f"captioning with {info.name} (arch {info.architecture or '?'}, "
                    f"num_ctx={captioner.num_ctx}"
                    f"{' [fallback]' if captioner.ctx_is_fallback else ' (model maximum)'}, "
                    f"{'short' if info.is_small else 'detailed'} prompt first)"
                )
            except ollama.OllamaError as e:
                stats.notes.append(f"captioning disabled: {e}")
                caption = False
        else:
            stats.notes.append("captioning disabled by request")

        clip = ClipEmbedder(clip_model or None, device) if clip_model else ClipEmbedder(device=device)
        clip.load()
        stats.notes.append(f"CLIP {clip.model_name} on {clip.device}")
        texter = TextEmbedder(clip_fallback=clip)

        st = store or Store()
        known = {} if reindex else st.known_signatures()

        emit(event="start", total=len(files), **stats.as_dict())

        batch: list[dict] = []
        last_flush = [time.time()]

        def flush():
            if not batch:
                return
            last_flush[0] = time.time()
            vecs = clip.embed_images([b["image"] for b in batch])
            tvecs = texter.embed([b["document"] for b in batch])
            st.upsert_clip([b["id"] for b in batch], vecs, [b["meta"] for b in batch])
            st.upsert_text([b["id"] for b in batch], tvecs,
                           [b["document"] for b in batch], [b["meta"] for b in batch])
            for b in batch:
                b["image"].close()
            batch.clear()

        for n, path in enumerate(files, 1):
            if should_stop is not None and should_stop():
                stats.stopped_early = True
                stats.notes.append(
                    f"stopped early after {stats.indexed} photo(s) to make way for "
                    f"a run you started; the rest are picked up next time")
                break
            pid = path_id(path)
            try:
                sig = file_signature(path)
            except OSError as e:
                stats.failed += 1
                stats.errors.append(f"{path}: {e}")
                continue

            if not reindex and known.get(pid) == sig:
                stats.skipped_unchanged += 1
                emit(event="skip", n=n, total=len(files), path=str(path))
                continue

            try:
                img = open_upright(path)          # EXIF orientation applied here, once
            except Exception as e:
                stats.failed += 1
                stats.errors.append(f"{path}: cannot open ({type(e).__name__}: {e})")
                emit(event="fail", n=n, total=len(files), path=str(path), error=str(e))
                continue

            try:
                ex = read_exif(path, img)
                write_thumbnail(img, path)

                ocr_text = ocr_image(img) if do_ocr else ""
                if ocr_text:
                    stats.ocr_hits += 1

                alt = ocr_alt.AltText(detail="disabled")
                if alt_method != "none":
                    alt = ocr_alt.read(
                        img, method=alt_method, languages=alt_langs,
                        vlm_model=captioner.model.name if captioner else None,
                        vlm_num_ctx=captioner.num_ctx if captioner else None,
                    )
                    if alt.text:
                        stats.alt_ocr_hits += 1

                if captioner is not None:
                    cap = captioner.caption(img)
                    if cap.status == "ok":
                        stats.caption_ok += 1
                    elif cap.status == "skipped":
                        stats.caption_skipped += 1
                    else:
                        stats.caption_empty += 1
                    if not captioner.enabled and captioner.disabled_reason not in "".join(stats.notes):
                        msg = (f"CAPTIONING SWITCHED OFF mid-run: {captioner.disabled_reason}. "
                               f"The remaining photos are indexed on CLIP + OCR + EXIF only.")
                        stats.notes.append(msg)
                        emit(event="warn", message=msg)
                else:
                    from .caption import CaptionResult
                    cap = CaptionResult(status="skipped", detail="captioning off")
                    stats.caption_skipped += 1

                document = build_document(cap.text, ocr_text, ex.as_text(), path, alt.text)
                meta = _clean_meta({
                    "path": str(path),
                    "folder": str(path.parent),
                    "filename": path.name,
                    "signature": sig,
                    "caption": cap.text,
                    "caption_status": cap.status,
                    "caption_detail": cap.detail,
                    "caption_model": captioner.model.name if captioner else "",
                    "ocr": ocr_text,
                    "has_ocr": bool(ocr_text),
                    "ocr_alt": alt.text,
                    "ocr_alt_method": alt.method,
                    "ocr_alt_conf": alt.confidence or None,
                    "ocr_alt_detail": "" if alt.text else alt.detail,
                    "has_ocr_alt": bool(alt.text),
                    "has_any_text": bool(ocr_text or alt.text),
                    "taken_at": ex.taken_at,
                    "year": ex.year,
                    "month": ex.month,
                    "camera": ex.camera,
                    "lens": ex.lens,
                    "gps_lat": ex.gps_lat,
                    "gps_lon": ex.gps_lon,
                    "width": ex.width,
                    "height": ex.height,
                    "orientation": ex.orientation,
                    "date_source": ex.date_source,
                    "indexed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "pipeline": pipeline_signature(),
                    **{version_key(k): v for k, v in PIPELINE_VERSIONS.items()},
                })

                small = img.copy()
                small.thumbnail((384, 384))
                img.close()
                batch.append({"id": pid, "image": small, "document": document, "meta": meta})
                stats.indexed += 1
                emit(event="photo", n=n, total=len(files), path=str(path),
                     caption=cap.text, caption_status=cap.status,
                     ocr=bool(ocr_text), ocr_alt=bool(alt.text))

                # Flush on size *or* age: a slow run (captioning takes seconds per
                # photo) would otherwise keep up to CLIP_BATCH photos invisible.
                if (len(batch) >= CLIP_BATCH
                        or time.time() - last_flush[0] >= FLUSH_INTERVAL_SECONDS):
                    flush()
            except Exception as e:
                stats.failed += 1
                stats.errors.append(f"{path}: {type(e).__name__}: {e}")
                emit(event="fail", n=n, total=len(files), path=str(path), error=str(e))

        flush()

        if captioner is not None:
            stats.notes.append(captioner.summary())
        if caption and stats.indexed and stats.caption_ok == 0:
            stats.notes.append(
                "WARNING: not one photo produced a usable caption. Search will rely on "
                "CLIP and OCR only. Check `ophotofinder models` and try --vision-model moondream."
            )
        emit(event="done", **stats.as_dict())

    return stats


# --------------------------------------------------------------------------
# Upgrading an existing index to a newer pipeline
# --------------------------------------------------------------------------

def stale_components(meta: dict) -> list[str]:
    """Which components in this record are behind the current pipeline."""
    out = []
    for comp, want in PIPELINE_VERSIONS.items():
        have = meta.get(version_key(comp))
        try:
            have = int(have)
        except (TypeError, ValueError):
            have = 0                     # indexed before versioning existed
        if have < want:
            out.append(comp)
    return out


def plan_upgrade(store: Store | None = None) -> dict:
    """What an upgrade would do, without doing it."""
    st = store or Store()
    records = st.all_records_full()
    per_component: dict[str, int] = {}
    stale_ids: list[str] = []
    missing_file = 0

    for r in records:
        stale = stale_components(r)
        if not stale:
            continue
        if not r.get("path") or not Path(r["path"]).exists():
            missing_file += 1
            continue
        stale_ids.append(r["id"])
        for c in stale:
            per_component[c] = per_component.get(c, 0) + 1

    return {
        "total": len(records),
        "stale": len(stale_ids),
        "up_to_date": len(records) - len(stale_ids) - missing_file,
        "missing_file": missing_file,
        "per_component": per_component,
        "current": dict(PIPELINE_VERSIONS),
        "signature": pipeline_signature(),
    }


def upgrade_index(
    *,
    vision_model: str | None = None,
    alt_ocr: str = "auto",
    alt_langs: list[str] | None = None,
    device: str | None = None,
    limit: int | None = None,
    only: list[str] | None = None,
    progress: Progress | None = None,
    store: Store | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> IndexStats:
    """Re-run only the components that are behind, reusing everything else.

    This is the cheap path: bumping the EXIF version re-reads EXIF for every
    photo but does not re-run the vision model, which would otherwise dominate
    the cost of any pipeline change.
    """
    ensure_dirs()
    stats = IndexStats()

    def emit(**kw):
        if progress:
            progress(kw)

    with IndexLock(note="upgrading index to " + pipeline_signature()):
        st = store or Store()
        records = st.all_records_full()

        todo = []
        for r in records:
            stale = stale_components(r)
            if only:
                stale = [c for c in stale if c in only]
            if not stale:
                continue
            if not r.get("path") or not Path(r["path"]).exists():
                stats.errors.append(f"{r.get('path') or r['id']}: file is gone")
                stats.failed += 1
                continue
            todo.append((r, stale))
        if limit:
            todo = todo[:limit]

        stats.found = len(todo)
        if not todo:
            stats.notes.append(f"index is already at pipeline {pipeline_signature()}")
            emit(event="done", **stats.as_dict())
            return stats

        needs_caption = any("caption" in s for _, s in todo)
        needs_clip = any("clip" in s for _, s in todo)

        captioner = None
        if needs_caption:
            try:
                info = ollama.pick_vision_model(vision_model)
                captioner = Captioner(info)
                stats.notes.append(f"re-captioning with {info.name}")
            except ollama.OllamaError as e:
                stats.notes.append(f"cannot re-caption: {e}")

        clip = ClipEmbedder(device=device)
        clip.load()
        texter = TextEmbedder(clip_fallback=clip)
        stats.notes.append(f"CLIP {clip.model_name} on {clip.device}")

        alt_method = alt_ocr
        if alt_method == "auto":
            alt_method = ocr_alt.best_available(have_vlm=captioner is not None)

        emit(event="start", total=len(todo), **stats.as_dict())

        clip_batch: list[tuple[str, object, dict]] = []
        text_batch: list[tuple[str, str, dict]] = []
        last_flush = [time.time()]

        def flush():
            if clip_batch:
                vecs = clip.embed_images([b[1] for b in clip_batch])
                st.upsert_clip([b[0] for b in clip_batch], vecs, [b[2] for b in clip_batch])
                for b in clip_batch:
                    b[1].close()
                clip_batch.clear()
            if text_batch:
                tvecs = texter.embed([b[1] for b in text_batch])
                st.upsert_text([b[0] for b in text_batch], tvecs,
                               [b[1] for b in text_batch], [b[2] for b in text_batch])
                text_batch.clear()
            last_flush[0] = time.time()

        for n, (rec, stale) in enumerate(todo, 1):
            if should_stop is not None and should_stop():
                stats.stopped_early = True
                stats.notes.append(
                    f"stopped early after {stats.indexed} photo(s) to make way for "
                    f"a run you started")
                break
            path = Path(rec["path"])
            meta = {k: v for k, v in rec.items() if k not in ("id", "document", "thumb", "photo")}
            try:
                img = open_upright(path) if stale else None
            except Exception as e:
                stats.failed += 1
                stats.errors.append(f"{path}: {type(e).__name__}: {e}")
                emit(event="fail", n=n, total=len(todo), path=str(path), error=str(e))
                continue

            try:
                if "exif" in stale:
                    ex = read_exif(path, img)
                    meta.update({
                        "taken_at": ex.taken_at, "year": ex.year, "month": ex.month,
                        "camera": ex.camera, "lens": ex.lens,
                        "gps_lat": ex.gps_lat, "gps_lon": ex.gps_lon,
                        "width": ex.width, "height": ex.height,
                        "orientation": ex.orientation, "date_source": ex.date_source,
                    })
                    write_thumbnail(img, path)

                if "ocr" in stale:
                    meta["ocr"] = ocr_image(img) if TESSERACT_OK else ""
                    meta["has_ocr"] = bool(meta["ocr"])

                if "ocr_alt" in stale and alt_method != "none":
                    alt = ocr_alt.read(
                        img, method=alt_method, languages=alt_langs,
                        vlm_model=captioner.model.name if captioner else None,
                        vlm_num_ctx=captioner.num_ctx if captioner else None,
                    )
                    meta.update({
                        "ocr_alt": alt.text, "ocr_alt_method": alt.method,
                        "ocr_alt_conf": alt.confidence or None,
                        "ocr_alt_detail": "" if alt.text else alt.detail,
                        "has_ocr_alt": bool(alt.text),
                    })
                    if alt.text:
                        stats.alt_ocr_hits += 1

                if "caption" in stale and captioner is not None:
                    cap = captioner.caption(img)
                    meta.update({
                        "caption": cap.text, "caption_status": cap.status,
                        "caption_detail": cap.detail,
                        "caption_model": captioner.model.name,
                    })
                    if cap.status == "ok":
                        stats.caption_ok += 1

                meta["has_any_text"] = bool(meta.get("ocr") or meta.get("ocr_alt"))
                meta["indexed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                for comp in stale:
                    meta[version_key(comp)] = PIPELINE_VERSIONS[comp]
                # document is rebuilt whenever any of its inputs moved
                if any(c in stale for c in DOCUMENT_INPUTS) or "document" in stale:
                    meta[version_key("document")] = PIPELINE_VERSIONS["document"]
                meta["pipeline"] = pipeline_signature()
                meta = _clean_meta(meta)

                document = build_document(
                    str(meta.get("caption", "")), str(meta.get("ocr", "")),
                    Exif(
                        taken_at=str(meta.get("taken_at", "")),
                        camera=str(meta.get("camera", "")),
                        lens=str(meta.get("lens", "")),
                        gps_lat=meta.get("gps_lat"), gps_lon=meta.get("gps_lon"),
                        orientation=str(meta.get("orientation", "")),
                    ).as_text(),
                    path, str(meta.get("ocr_alt", "")),
                )
                text_batch.append((rec["id"], document, meta))

                if "clip" in stale:
                    small = img.copy()
                    small.thumbnail((384, 384))
                    clip_batch.append((rec["id"], small, meta))

                if img is not None:
                    img.close()
                stats.indexed += 1
                emit(event="photo", n=n, total=len(todo), path=str(path),
                     caption=str(meta.get("caption", "")),
                     caption_status=str(meta.get("caption_status", "")),
                     ocr=bool(meta.get("ocr")), ocr_alt=bool(meta.get("ocr_alt")),
                     upgraded=",".join(stale))

                if (len(text_batch) >= CLIP_BATCH
                        or time.time() - last_flush[0] >= FLUSH_INTERVAL_SECONDS):
                    flush()
            except Exception as e:
                stats.failed += 1
                stats.errors.append(f"{path}: {type(e).__name__}: {e}")
                emit(event="fail", n=n, total=len(todo), path=str(path), error=str(e))

        flush()
        stats.notes.append(f"index now at pipeline {pipeline_signature()}")
        emit(event="done", **stats.as_dict())
    return stats
