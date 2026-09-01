"""Captioning through a local Ollama vision model.

Hard-won rules encoded here:

* Only models reporting the ``vision`` capability are ever used (checked in
  ``ollama.pick_vision_model``).
* ``num_ctx`` is sent explicitly, set to the model's own declared maximum.
* Small VLMs (moondream, ~1.4B) return an empty string -- or bare grounding
  coordinates such as ``[0.31, 0.71, 0.64, 0.87]`` -- when given a long
  multi-part prompt. They get the short "Describe this image." first, and any
  caption that comes back empty or non-prose is retried with the other prompt.
* Nothing fails silently: every photo records *why* a caption is missing
  (``ok`` / ``empty`` / ``skipped``), and if the first few photos all caption
  empty, captioning is switched off for the run and said out loud.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from PIL import Image

from . import ollama
from .config import (
    CAPTION_FAILURE_STREAK,
    LONG_CAPTION_PROMPT,
    SHORT_CAPTION_PROMPT,
)
from .imaging import to_jpeg_bytes

# A bare grounding box, e.g. "[0.31, 0.71, 0.64, 0.87]" -- moondream's answer to
# a prompt it could not parse. It is not a caption.
_COORDS = re.compile(r"^[\s\[\]\(\),.;:0-9-]+$")
_WORD = re.compile(r"[A-Za-z]{3,}")


def is_prose(text: str) -> bool:
    """True when a caption looks like a sentence rather than junk."""
    t = (text or "").strip()
    if len(t) < 12:
        return False
    if _COORDS.match(t):          # pure coordinates / punctuation
        return False
    words = _WORD.findall(t)
    if len(words) < 4:
        return False
    # Mostly digits and brackets with a word or two smuggled in is still junk.
    alpha = sum(c.isalpha() or c.isspace() for c in t)
    return alpha / len(t) > 0.6


@dataclass
class CaptionResult:
    text: str = ""
    status: str = "skipped"      # ok | empty | skipped | error
    detail: str = ""
    prompt_used: str = ""


class Captioner:
    """Stateful across a run so it can notice systemic failure and stop."""

    def __init__(self, model: ollama.ModelInfo, enabled: bool = True):
        self.model = model
        self.enabled = enabled
        # The model's own maximum, never Ollama's default.
        self.num_ctx = model.context_length or ollama.resolve_num_ctx(model.name)
        self.ctx_is_fallback = not model.context_length
        # Small models get the short prompt first; capable ones get the rich one.
        self.primary = SHORT_CAPTION_PROMPT if model.is_small else LONG_CAPTION_PROMPT
        self.fallback = LONG_CAPTION_PROMPT if model.is_small else SHORT_CAPTION_PROMPT
        self.empty_streak = 0
        self.ok_count = 0
        self.attempted = 0
        self.disabled_reason = ""

    def _one(self, jpeg: bytes, prompt: str) -> str:
        return ollama.generate(
            self.model.name,
            prompt,
            images=[jpeg],
            num_ctx=self.num_ctx,
            num_predict=200,
        )

    def caption(self, img: Image.Image) -> CaptionResult:
        if not self.enabled:
            return CaptionResult(status="skipped", detail=self.disabled_reason or "captioning off")

        jpeg = to_jpeg_bytes(img)
        self.attempted += 1
        tried: list[str] = []
        last_err = ""

        for prompt, label in ((self.primary, "primary"), (self.fallback, "fallback")):
            try:
                text = self._one(jpeg, prompt).strip()
            except ollama.OllamaError as e:
                last_err = str(e)
                tried.append(f"{label}:error")
                continue
            except Exception as e:  # network / timeout
                last_err = f"{type(e).__name__}: {e}"
                tried.append(f"{label}:error")
                continue

            if is_prose(text):
                self.empty_streak = 0
                self.ok_count += 1
                return CaptionResult(text=text, status="ok", prompt_used=label)
            tried.append(f"{label}:{'empty' if not text else 'non-prose'}")

        self.empty_streak += 1
        self._maybe_disable()
        detail = ", ".join(tried) + (f" ({last_err})" if last_err else "")
        return CaptionResult(status="error" if last_err else "empty", detail=detail)

    def _maybe_disable(self) -> None:
        """If the first several photos all caption empty, stop pretending it works."""
        if self.ok_count == 0 and self.empty_streak >= CAPTION_FAILURE_STREAK:
            self.enabled = False
            self.disabled_reason = (
                f"the first {self.empty_streak} photos all produced an empty or "
                f"non-prose caption from '{self.model.name}'"
            )

    def summary(self) -> str:
        if not self.attempted:
            return "captioning: not attempted"
        return (
            f"captioning: {self.ok_count}/{self.attempted} ok via {self.model.name} "
            f"(num_ctx={self.num_ctx}{' [fallback: model declares none]' if self.ctx_is_fallback else ''})"
        )
