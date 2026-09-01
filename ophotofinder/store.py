"""Chroma storage: two collections, one per family of signal.

* ``photo_clip`` -- CLIP image vectors. Queried with a CLIP *text* vector, so a
  plain-language query reaches photos that have no caption at all.
* ``photo_text`` -- the caption + OCR + EXIF sentence, embedded with a text
  model. Good at names, dates, signs, written words in the photo.

Both are written with explicit embeddings; Chroma is never asked to embed
anything itself, so the two spaces can never get silently crossed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import chromadb
from chromadb.config import Settings

try:  # chromadb >= 0.5
    from chromadb.errors import NotFoundError
except ImportError:  # pragma: no cover - older chromadb
    class NotFoundError(Exception):
        pass

from .config import CHROMA_DIR, CLIP_COLLECTION, TEXT_COLLECTION, ensure_dirs

_HNSW = {"hnsw:space": "cosine"}


class Store:
    def __init__(self, directory: Path | None = None):
        ensure_dirs()
        self.directory = Path(directory or CHROMA_DIR)
        self.client = chromadb.PersistentClient(
            path=str(self.directory),
            settings=Settings(anonymized_telemetry=False, allow_reset=False),
        )
        self._open()

    def _open(self) -> None:
        self.clip = self.client.get_or_create_collection(
            CLIP_COLLECTION, metadata=_HNSW, embedding_function=None
        )
        self.text = self.client.get_or_create_collection(
            TEXT_COLLECTION, metadata=_HNSW, embedding_function=None
        )

    def _call(self, which: str, method: str, *args, **kwargs):
        """Call a Chroma collection method, re-opening once if it went stale.

        Another process may recreate a collection (an older `clean`, a manual
        reset, a restored backup), leaving this handle bound to an id that no
        longer exists. The collection is looked up by attribute name on every
        attempt, so the retry runs against the *reopened* handle rather than
        the dead one.
        """
        try:
            return getattr(getattr(self, which), method)(*args, **kwargs)
        except NotFoundError:
            self._open()
            return getattr(getattr(self, which), method)(*args, **kwargs)

    # ---------------------------------------------------------------- writing

    def upsert_clip(self, ids, embeddings, metadatas) -> None:
        if ids:
            self._call("clip", "upsert", ids=list(ids),
                        embeddings=[list(map(float, e)) for e in embeddings],
                        metadatas=list(metadatas))

    def upsert_text(self, ids, embeddings, documents, metadatas) -> None:
        if ids:
            self._call("text", "upsert", ids=list(ids),
                        embeddings=[list(map(float, e)) for e in embeddings],
                        documents=list(documents), metadatas=list(metadatas))

    def delete(self, ids: Iterable[str]) -> None:
        ids = list(ids)
        if not ids:
            return
        for coll in (self.clip, self.text):
            try:
                coll.delete(ids=ids)
            except Exception:
                pass

    # ---------------------------------------------------------------- reading

    def known_signatures(self) -> dict[str, str]:
        """id -> file signature, for incremental re-indexing."""
        out: dict[str, str] = {}
        offset, limit = 0, 1000
        while True:
            batch = self._call("clip", "get", include=["metadatas"], limit=limit, offset=offset)
            ids = batch.get("ids") or []
            metas = batch.get("metadatas") or []
            for i, m in zip(ids, metas):
                out[i] = (m or {}).get("signature", "")
            if len(ids) < limit:
                break
            offset += limit
        return out

    def indexed_paths(self) -> set[str]:
        """Every absolute path in the index. The web server serves nothing else."""
        paths: set[str] = set()
        offset, limit = 0, 1000
        while True:
            batch = self._call("clip", "get", include=["metadatas"], limit=limit, offset=offset)
            ids = batch.get("ids") or []
            for m in batch.get("metadatas") or []:
                p = (m or {}).get("path")
                if p:
                    paths.add(p)
            if len(ids) < limit:
                break
            offset += limit
        return paths

    def get_meta(self, ids: list[str]) -> dict[str, dict[str, Any]]:
        if not ids:
            return {}
        res = self._call("text", "get", ids=ids, include=["metadatas", "documents"])
        out: dict[str, dict[str, Any]] = {}
        for i, m, d in zip(res.get("ids") or [], res.get("metadatas") or [],
                           res.get("documents") or []):
            meta = dict(m or {})
            meta["document"] = d or ""
            out[i] = meta
        missing = [i for i in ids if i not in out]
        if missing:
            res = self._call("clip", "get", ids=missing, include=["metadatas"])
            for i, m in zip(res.get("ids") or [], res.get("metadatas") or []):
                out[i] = dict(m or {})
        return out

    def records(self) -> list[tuple[str, str]]:
        """(id, path) for every indexed photo."""
        out: list[tuple[str, str]] = []
        offset, limit = 0, 1000
        while True:
            batch = self._call("clip", "get", include=["metadatas"], limit=limit, offset=offset)
            ids = batch.get("ids") or []
            for i, m in zip(ids, batch.get("metadatas") or []):
                out.append((i, (m or {}).get("path", "")))
            if len(ids) < limit:
                break
            offset += limit
        return out

    def drop_all(self) -> None:
        """Empty both collections, keeping the collections themselves.

        Deleting and recreating a collection gives it a new id, which
        invalidates every ``Collection`` handle held elsewhere -- a long-running
        web server would then raise ``NotFoundError`` on every call until it was
        restarted. Removing the records leaves all live handles valid.
        """
        for coll in (self.clip, self.text):
            while True:
                batch = coll.get(include=[], limit=2000)
                ids = batch.get("ids") or []
                if not ids:
                    break
                coll.delete(ids=ids)

    def all_records_full(self, where: dict | None = None) -> list[dict]:
        """Every record with its metadata and document text.

        Chroma cannot sort, so browsing the library pages in Python over the
        full metadata list. Fine for a personal library; this is the scaling
        limit if one ever grows to six figures.
        """
        out: list[dict] = []
        offset, limit = 0, 1000
        while True:
            batch = self._call(
                "text", "get", include=["metadatas", "documents"],
                limit=limit, offset=offset, where=where
            )
            ids = batch.get("ids") or []
            for i, m, d in zip(ids, batch.get("metadatas") or [],
                               batch.get("documents") or []):
                rec = dict(m or {})
                rec["id"] = i
                rec["document"] = d or ""
                out.append(rec)
            if len(ids) < limit:
                break
            offset += limit
        return out

    def cameras(self) -> list[str]:
        seen: set[str] = set()
        offset, limit = 0, 1000
        while True:
            batch = self._call("clip", "get", include=["metadatas"], limit=limit, offset=offset)
            ids = batch.get("ids") or []
            for m in batch.get("metadatas") or []:
                c = (m or {}).get("camera")
                if c:
                    seen.add(c)
            if len(ids) < limit:
                break
            offset += limit
        return sorted(seen)

    def literal_matches(self, terms: list[str], limit: int = 200) -> dict[str, int]:
        """id -> how many of ``terms`` appear literally in its document.

        Exact substring search, which is script-agnostic. The sentence embedder
        is English-only, so this is the only thing that makes text in other
        scripts findable at all -- and it also beats embeddings on names,
        numbers and codes.
        """
        counts: dict[str, int] = {}
        for term in terms:
            if not term:
                continue
            seen_this_term: set[str] = set()
            for variant in dict.fromkeys([term, term.lower(), term.upper(), term.capitalize()]):
                try:
                    res = self._call("text", "get", where_document={"$contains": variant},
                                     include=[], limit=limit)
                except Exception:
                    continue
                for i in res.get("ids") or []:
                    seen_this_term.add(i)
            for i in seen_this_term:
                counts[i] = counts.get(i, 0) + 1
        return counts

    def counts(self) -> dict[str, int]:
        return {"clip": self._call("clip", "count"), "text": self._call("text", "count")}

    def folders(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        offset, limit = 0, 1000
        while True:
            batch = self._call("clip", "get", include=["metadatas"], limit=limit, offset=offset)
            ids = batch.get("ids") or []
            for m in batch.get("metadatas") or []:
                f = (m or {}).get("folder")
                if f:
                    counts[f] = counts.get(f, 0) + 1
            if len(ids) < limit:
                break
            offset += limit
        return dict(sorted(counts.items()))

    def caption_health(self) -> dict[str, int]:
        stats: dict[str, int] = {}
        offset, limit = 0, 1000
        while True:
            batch = self._call("clip", "get", include=["metadatas"], limit=limit, offset=offset)
            ids = batch.get("ids") or []
            for m in batch.get("metadatas") or []:
                s = (m or {}).get("caption_status", "unknown")
                stats[s] = stats.get(s, 0) + 1
            if len(ids) < limit:
                break
            offset += limit
        return stats
