"""Retrieval: query both collections independently, then fuse the rankings.

The two signals live in incomparable vector spaces, so their distances cannot
be averaged. Reciprocal rank fusion combines the *orderings* instead, which is
scale-free and behaves well when one signal is missing entirely (a photo with
no caption still ranks on CLIP alone).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import ollama
from .clip_embed import ClipEmbedder
from .store import Store
from .text_embed import TextEmbedder

RRF_K = 60


# CJK/Kana/Hangul runs, and Latin/digit words. Used for literal matching, which
# is the only signal that works for scripts the sentence embedder cannot represent.
_CJK_RUN = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]{2,}")
_WORD_RUN = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]{2,}")


def literal_terms(query: str, limit: int = 8) -> list[str]:
    """Terms worth looking for verbatim in the indexed text."""
    terms = _CJK_RUN.findall(query) + _WORD_RUN.findall(query)
    return list(dict.fromkeys(terms))[:limit]


@dataclass
class Hit:
    id: str
    path: str
    score: float = 0.0
    clip_rank: int | None = None
    text_rank: int | None = None
    literal_rank: int | None = None
    literal_terms: int = 0
    clip_score: float | None = None
    text_score: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def why(self) -> str:
        bits = []
        if self.clip_rank is not None:
            bits.append(f"visual #{self.clip_rank}")
        if self.text_rank is not None:
            bits.append(f"text #{self.text_rank}")
        if self.literal_rank is not None:
            bits.append(f"exact text ×{self.literal_terms}")
        return " + ".join(bits) or "-"

    def as_dict(self) -> dict:
        d = {
            "id": self.id, "path": self.path, "score": round(self.score, 5),
            "clip_rank": self.clip_rank, "text_rank": self.text_rank,
            "literal_rank": self.literal_rank, "literal_terms": self.literal_terms,
            "why": self.why,
        }
        d.update({k: self.meta.get(k) for k in (
            "filename", "folder", "caption", "caption_status", "caption_detail",
            "ocr", "ocr_alt", "ocr_alt_method", "ocr_alt_conf",
            "taken_at", "year", "camera", "gps_lat", "gps_lon", "orientation",
        )})
        return d


def build_where(year: str | None = None, camera: str | None = None,
                folder: str | None = None, has_ocr: bool | None = None) -> dict | None:
    clauses = []
    if year:
        clauses.append({"year": {"$eq": str(year)}})
    if camera:
        clauses.append({"camera": {"$eq": camera}})
    if folder:
        clauses.append({"folder": {"$eq": str(folder)}})
    if has_ocr is not None:
        clauses.append({"has_ocr": {"$eq": bool(has_ocr)}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


class Searcher:
    def __init__(self, store: Store | None = None, clip_model: str | None = None,
                 device: str | None = None):
        self.store = store or Store()
        self.clip = ClipEmbedder(clip_model, device) if clip_model else ClipEmbedder(device=device)
        self.text = TextEmbedder(clip_fallback=self.clip)

    def search(self, query: str, k: int = 12, pool: int | None = None,
               where: dict | None = None, use_clip: bool = True,
               use_text: bool = True) -> list[Hit]:
        pool = pool or max(k * 4, 40)
        hits: dict[str, Hit] = {}

        def touch(hid: str) -> Hit:
            if hid not in hits:
                hits[hid] = Hit(id=hid, path="")
            return hits[hid]

        clip_n = self.store._call("clip", "count")
        if use_clip and clip_n:
            qv = self.clip.embed_texts([query])[0]
            res = self.store._call(
                "clip", "query",
                query_embeddings=[qv.tolist()],
                n_results=min(pool, clip_n),
                where=where, include=["metadatas", "distances"],
            )
            for rank, (hid, meta, dist) in enumerate(zip(
                    res["ids"][0], res["metadatas"][0], res["distances"][0]), 1):
                h = touch(hid)
                h.clip_rank, h.clip_score = rank, round(1.0 - float(dist), 4)
                h.meta = {**(meta or {}), **h.meta}
                h.score += 1.0 / (RRF_K + rank)

        text_n = self.store._call("text", "count")
        if use_text and text_n:
            tv = self.text.embed([query])[0]
            res = self.store._call(
                "text", "query",
                query_embeddings=[tv.tolist()],
                n_results=min(pool, text_n),
                where=where, include=["metadatas", "distances", "documents"],
            )
            for rank, (hid, meta, dist, doc) in enumerate(zip(
                    res["ids"][0], res["metadatas"][0], res["distances"][0],
                    res["documents"][0]), 1):
                h = touch(hid)
                h.text_rank, h.text_score = rank, round(1.0 - float(dist), 4)
                h.meta = {**(meta or {}), **h.meta}
                h.meta.setdefault("document", doc or "")
                h.score += 1.0 / (RRF_K + rank)

        if use_text:
            terms = literal_terms(query)
            if terms:
                counts = self.store.literal_matches(terms, limit=max(pool, 200))
                # More matched terms is a stronger signal, so rank by that.
                for rank, (hid, n) in enumerate(
                        sorted(counts.items(), key=lambda kv: -kv[1]), 1):
                    h = touch(hid)
                    h.literal_rank, h.literal_terms = rank, n
                    h.score += 1.0 / (RRF_K + rank)

        ranked = sorted(hits.values(), key=lambda h: -h.score)[:k]
        missing = [h.id for h in ranked if not h.meta.get("path")]
        if missing:
            extra = self.store.get_meta(missing)
            for h in ranked:
                if h.id in extra:
                    h.meta = {**extra[h.id], **h.meta}
        for h in ranked:
            h.path = h.meta.get("path", "")
        return [h for h in ranked if h.path]


def answer(query: str, hits: list[Hit], model: str | None = None,
           max_photos: int = 8) -> str:
    """Optional generation step: let a local text model answer from what we found."""
    info = ollama.pick_text_model(model)
    if info is None:
        return ""

    lines = []
    for i, h in enumerate(hits[:max_photos], 1):
        m = h.meta
        desc = m.get("caption") or "(no caption available)"
        parts = [f"[{i}] {Path(h.path).name}: {desc}"]
        best_text = m.get("ocr_alt") or m.get("ocr")
        if best_text:
            parts.append(f"    text in image: {str(best_text)[:300]}")
        if m.get("taken_at"):
            parts.append(f"    taken: {m['taken_at']}")
        if m.get("camera"):
            parts.append(f"    camera: {m['camera']}")
        lines.append("\n".join(parts))
    context = "\n".join(lines) or "(no photos matched)"

    prompt = (
        "You are answering a question about someone's personal photo library. "
        "Use only the photo records below. Refer to photos by their [number]. "
        "If the records do not answer the question, say so plainly.\n\n"
        f"PHOTO RECORDS:\n{context}\n\nQUESTION: {query}\n\nANSWER:"
    )
    return ollama.generate(
        info.name, prompt,
        num_ctx=info.context_length or ollama.resolve_num_ctx(info.name),
        num_predict=350, temperature=0.2,
    )
