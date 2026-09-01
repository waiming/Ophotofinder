"""Thin Ollama client.

Two things here cost real debugging time and are therefore explicit:

1. Capability discovery. ``/api/show`` reports a ``capabilities`` list; only
   models that report ``"vision"`` can caption an image. Guessing from the
   model name is wrong often enough to matter.
2. Context length. Ollama silently defaults ``num_ctx`` to a small value (2048
   or 4096 depending on build), which truncates a base64 image plus prompt and
   yields empty captions. We read the model's own maximum out of
   ``model_info["<arch>.context_length"]`` and send it explicitly.
"""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from typing import Any

import requests

from .config import (BROKEN_VISION_MODELS, FALLBACK_NUM_CTX, OLLAMA_HOST,
                     OLLAMA_TIMEOUT, normalise_host)


class OllamaError(RuntimeError):
    pass


@dataclass
class ModelInfo:
    name: str
    capabilities: list[str] = field(default_factory=list)
    architecture: str = ""
    context_length: int = 0
    parameter_count: int = 0
    size: int = 0
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def has_vision(self) -> bool:
        return "vision" in self.capabilities

    @property
    def is_broken(self) -> bool:
        base = self.name.split(":")[0]
        return self.name in BROKEN_VISION_MODELS or base in BROKEN_VISION_MODELS

    @property
    def is_small(self) -> bool:
        from .config import SMALL_VLM_PARAM_LIMIT
        return 0 < self.parameter_count <= SMALL_VLM_PARAM_LIMIT

    def fits(self, accelerator: dict | None = None) -> dict:
        """How well this model suits the machine it would run on."""
        from . import sysinfo

        return sysinfo.assess_model(
            size_bytes=self.size, parameter_count=self.parameter_count,
            details=self.details, name=self.name, accelerator=accelerator)

    def describe(self) -> str:
        caps = ",".join(self.capabilities) or "-"
        params = f"{self.parameter_count/1e9:.1f}B" if self.parameter_count else "?"
        return f"{self.name}  [{caps}]  {params}  ctx={self.context_length or '?'}"


def host() -> str:
    """Where Ollama is: the saved setting if any, else the environment default.

    Resolved per call rather than at import, so changing it in Settings takes
    effect immediately without a restart.
    """
    from . import settings

    return normalise_host(settings.get("ollama_host", "")) or OLLAMA_HOST


def _url(path: str) -> str:
    return f"{host()}{path}"


def is_up() -> bool:
    try:
        requests.get(_url("/api/tags"), timeout=3).raise_for_status()
        return True
    except Exception:
        return False


def require_up() -> None:
    if not is_up():
        raise OllamaError(
            f"Cannot reach Ollama at {host()}. Start it with `ollama serve`, "
            f"or set the address in Settings."
        )


_sizes: dict[str, int] = {}


def list_model_names() -> list[str]:
    require_up()
    r = requests.get(_url("/api/tags"), timeout=10)
    r.raise_for_status()
    models = r.json().get("models", [])
    _sizes.clear()
    for m in models:
        try:
            _sizes[m["name"]] = int(m.get("size") or 0)
        except (TypeError, ValueError):
            pass
    return sorted(m["name"] for m in models)


def model_size(name: str) -> int:
    """On-disk size in bytes, as Ollama reports it."""
    if name not in _sizes:
        try:
            list_model_names()
        except Exception:
            return 0
    return _sizes.get(name, 0)


_show_cache: dict[str, ModelInfo] = {}
_cache_host: str = ""


def clear_cache() -> None:
    """Forget model metadata -- call when the server address changes."""
    _show_cache.clear()


def show(name: str) -> ModelInfo:
    """Fetch capabilities + context length for one model."""
    global _cache_host
    if _cache_host != host():          # a different server: nothing carries over
        _show_cache.clear()
        _cache_host = host()
    if name in _show_cache:
        return _show_cache[name]
    require_up()
    r = requests.post(_url("/api/show"), json={"model": name}, timeout=30)
    r.raise_for_status()
    d = r.json()

    mi = d.get("model_info") or {}
    arch = str(mi.get("general.architecture") or (d.get("details") or {}).get("family") or "")

    # The context length key is namespaced by architecture, e.g. "phi2.context_length",
    # "qwen2.context_length", "llama.context_length". Prefer the declared arch, then
    # fall back to any *.context_length key present.
    ctx = 0
    if arch:
        ctx = int(mi.get(f"{arch}.context_length") or 0)
    if not ctx:
        for k, v in mi.items():
            if k.endswith(".context_length"):
                try:
                    ctx = max(ctx, int(v))
                except (TypeError, ValueError):
                    continue

    params = int(mi.get("general.parameter_count") or 0)
    if not params:
        params = _parse_param_size((d.get("details") or {}).get("parameter_size", ""))

    info = ModelInfo(
        size=_sizes.get(name, 0),
        name=name,
        capabilities=list(d.get("capabilities") or []),
        architecture=arch,
        context_length=ctx,
        parameter_count=params,
        details=d.get("details") or {},
    )
    _show_cache[name] = info
    return info


def _parse_param_size(text: str) -> int:
    """'1.4B' / '7B' / '494.03M' -> integer parameter count."""
    text = (text or "").strip().upper()
    if not text:
        return 0
    mult = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}.get(text[-1])
    try:
        return int(float(text[:-1]) * mult) if mult else int(float(text))
    except ValueError:
        return 0


def resolve_num_ctx(model: str) -> int:
    """The model's own maximum context, for use as ``num_ctx``.

    Read from ``model_info["<arch>.context_length"]``. Falls back only when the
    model publishes no context length at all, which is reported rather than
    hidden -- Ollama's own default would silently truncate the image payload.
    """
    try:
        ctx = show(model).context_length
    except Exception:
        ctx = 0
    return ctx or FALLBACK_NUM_CTX


def inventory() -> list[ModelInfo]:
    """Every installed model with its capabilities resolved."""
    out = []
    for name in list_model_names():
        try:
            out.append(show(name))
        except Exception:
            out.append(ModelInfo(name=name))
    return out


def vision_models(include_broken: bool = False) -> list[ModelInfo]:
    """Only models that actually report the 'vision' capability."""
    models = [m for m in inventory() if m.has_vision]
    if not include_broken:
        models = [m for m in models if not m.is_broken]
    return models


def text_models() -> list[ModelInfo]:
    return [m for m in inventory() if "completion" in m.capabilities]


def _rank(name: str, preferred: list[str]) -> tuple[int, str]:
    base = name.split(":")[0].lower()
    for i, p in enumerate(preferred):
        if base == p or base.startswith(p):
            return (i, name)
    return (len(preferred), name)


def pick_vision_model(explicit: str | None = None) -> ModelInfo:
    """Resolve the captioning model, verifying capability rather than trusting the name."""
    if explicit:
        info = show(explicit)
        if not info.has_vision:
            raise OllamaError(
                f"Model '{explicit}' does not report the 'vision' capability "
                f"(it reports: {info.capabilities or 'none'}). It cannot caption images."
            )
        if info.is_broken:
            raise OllamaError(
                f"Model '{explicit}' is known not to load on current Ollama "
                f"(architecture '{info.architecture or 'mllama'}'). Try `moondream`."
            )
        return info

    from .config import PREFERRED_VISION_MODELS
    cands = vision_models()
    if not cands:
        broken = vision_models(include_broken=True)
        if broken:
            raise OllamaError(
                "The only vision models installed are known-broken on this Ollama "
                f"version ({', '.join(m.name for m in broken)}). Run `ollama pull moondream`."
            )
        raise OllamaError(
            "No installed model reports the 'vision' capability. Run `ollama pull moondream`."
        )
    cands.sort(key=lambda m: _rank(m.name, PREFERRED_VISION_MODELS))
    return cands[0]


def pick_text_model(explicit: str | None = None) -> ModelInfo | None:
    if explicit:
        return show(explicit)
    from .config import PREFERRED_TEXT_MODELS
    cands = [m for m in text_models() if not m.has_vision] or text_models()
    if not cands:
        return None
    cands.sort(key=lambda m: _rank(m.name, PREFERRED_TEXT_MODELS))
    return cands[0]


def generate(
    model: str,
    prompt: str,
    images: list[bytes] | None = None,
    num_ctx: int | None = None,
    temperature: float = 0.1,
    num_predict: int = 220,
    timeout: float | None = None,
) -> str:
    """One-shot generate.

    ``num_ctx`` is *always* sent, and always defaults to the model's own maximum.
    Callers need not pass it: leaving it unset resolves the maximum from the
    model's metadata here, so no call site can accidentally fall back to
    Ollama's default (which truncates a base64 image plus prompt and yields an
    empty response).
    """
    options: dict[str, Any] = {"temperature": temperature, "num_predict": num_predict}
    options["num_ctx"] = int(num_ctx) if num_ctx else resolve_num_ctx(model)

    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": options,
    }
    if images:
        payload["images"] = [base64.b64encode(b).decode("ascii") for b in images]

    r = requests.post(
        _url("/api/generate"), json=payload, timeout=timeout or OLLAMA_TIMEOUT
    )
    if r.status_code >= 400:
        detail = r.text.strip()
        try:
            detail = json.loads(detail).get("error", detail)
        except Exception:
            pass
        raise OllamaError(f"{model}: {detail}")
    return (r.json().get("response") or "").strip()
