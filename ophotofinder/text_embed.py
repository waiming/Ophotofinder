"""Text embeddings for the caption/OCR/EXIF collection.

Preference order, all local:
  1. Chroma's bundled ONNX MiniLM (small, fast, good at sentence similarity).
  2. CLIP's text tower, as a fallback if that model cannot be materialised.
"""
from __future__ import annotations

import threading
from typing import Sequence

import numpy as np


class TextEmbedder:
    def __init__(self, clip_fallback=None):
        self._fn = None
        self._backend = ""
        self._clip = clip_fallback
        self._lock = threading.Lock()

    def load(self) -> None:
        if self._fn is not None or self._backend == "clip":
            return
        with self._lock:
            if self._fn is not None or self._backend == "clip":
                return
            try:
                from chromadb.utils import embedding_functions

                fn = embedding_functions.DefaultEmbeddingFunction()
                fn(["warmup"])            # force the model download/lazy init now
                self._fn = fn
                self._backend = "onnx-minilm"
            except Exception:
                if self._clip is None:
                    raise
                self._backend = "clip"

    @property
    def backend(self) -> str:
        self.load()
        return self._backend

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        self.load()
        texts = [t if t.strip() else " " for t in texts]
        if self._backend == "clip":
            return self._clip.embed_texts(texts)
        arr = np.asarray(self._fn(list(texts)), dtype="float32")
        norms = np.linalg.norm(arr, axis=-1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms
