"""CLIP image/text embeddings via transformers, on MPS / CUDA / CPU.

CLIP puts images and text in one shared vector space, which is what lets a
plain-language query match a photo with no caption, no OCR and no EXIF -- the
first of the three independent signals in the index.
"""
from __future__ import annotations

import threading
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

from .config import DEFAULT_CLIP_MODEL


def pick_device(preferred: str | None = None) -> str:
    import torch

    if preferred and preferred != "auto":
        return preferred
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class ClipEmbedder:
    """Lazily-loaded CLIP. Thread-safe enough for the single-worker web server."""

    def __init__(self, model_name: str = DEFAULT_CLIP_MODEL, device: str | None = None):
        self.model_name = model_name
        self.device = pick_device(device)
        self._model = None
        self._proc = None
        self._lock = threading.Lock()

    def load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            import torch
            from transformers import CLIPModel, CLIPProcessor

            model = CLIPModel.from_pretrained(self.model_name)
            model.eval()
            try:
                model.to(self.device)
            except Exception:
                # e.g. an MPS build that rejects a dtype -- CPU always works.
                self.device = "cpu"
                model.to("cpu")
            self._proc = CLIPProcessor.from_pretrained(self.model_name)
            self._model = model
            self._torch = torch

    @property
    def dim(self) -> int:
        self.load()
        return int(self._model.config.projection_dim)

    @staticmethod
    def _features(out):
        """Unwrap the projected embedding.

        transformers <5 returned a bare tensor from ``get_image_features`` /
        ``get_text_features``; transformers >=5 returns a ``BaseModelOutputWithPooling``
        whose ``pooler_output`` holds the projected vector. Support both.
        """
        import torch

        if isinstance(out, torch.Tensor):
            return out
        pooled = getattr(out, "pooler_output", None)
        if pooled is not None:
            return pooled
        if isinstance(out, (tuple, list)):
            return out[1] if len(out) > 1 else out[0]
        raise TypeError(f"unexpected CLIP output type: {type(out)!r}")

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(arr, axis=-1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    def embed_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        self.load()
        torch = self._torch
        inputs = self._proc(images=list(images), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = self._features(self._model.get_image_features(**inputs))
        return self._normalize(feats.detach().cpu().float().numpy())

    def embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        self.load()
        torch = self._torch
        texts = list(texts)
        inputs = self._proc(
            text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = self._features(self._model.get_text_features(**inputs))
        return self._normalize(feats.detach().cpu().float().numpy())
