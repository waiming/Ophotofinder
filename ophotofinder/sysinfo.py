"""What this machine can actually run, and how well.

Model *size* is a poor predictor of speed. What matters far more is whether the
model's runtime matches the machine's accelerator:

* An MLX build (``format: safetensors`` with an MLX-style quantisation) runs on
  Apple Silicon's unified-memory GPU through the MLX runtime. On Apple Silicon it
  is usually the fastest option; on any other machine it is the wrong build.
* A GGUF build runs through llama.cpp, which offloads to Metal, CUDA or ROCm when
  present and otherwise falls back to the CPU.

So a 6.5 GB MLX model can comfortably beat a 2 GB GGUF one on a Mac, while the
same MLX model is a poor choice on a CUDA box. Everything here is detected on the
machine it runs on -- nothing is specific to any one computer.
"""
from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess

# Weights plus KV cache and runtime overhead.
OVERHEAD = 1.25
# Memory left for the operating system and other apps.
HEADROOM_FRACTION = 0.20
# Above this many parameters, CPU-only inference is painfully slow.
CPU_PARAM_LIMIT = 4_000_000_000


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


# --------------------------------------------------------------------- memory

def total_memory() -> int:
    system = platform.system()
    try:
        if system == "Darwin":
            return int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                                      capture_output=True, text=True, timeout=5).stdout.strip())
        if system == "Linux":
            with open("/proc/meminfo") as fh:
                for line in fh:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) * 1024
        if system == "Windows":
            import ctypes

            class M(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            st = M(); st.dwLength = ctypes.sizeof(M)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return int(st.ullTotalPhys)
    except Exception:
        pass
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return 0


# ---------------------------------------------------------------- accelerator

def detect_accelerator() -> dict:
    """What this machine will actually compute on. Detected, never assumed."""
    system, machine = platform.system(), platform.machine()

    if system == "Darwin" and machine == "arm64":
        chip = "Apple Silicon"
        try:
            chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True, timeout=5).stdout.strip() or chip
        except Exception:
            pass
        mem = total_memory()
        return {
            "kind": "apple_silicon", "name": chip,
            "memory": mem, "memory_human": human(mem),
            "unified": True,
            "runtimes": ["mlx", "gguf"],
            "preferred_runtime": "mlx",
            "summary": f"{chip} — GPU via Metal, unified {human(mem)} shared with the system",
        }

    # NVIDIA / AMD, discovered through torch when it is present, else the CLI tools.
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = int(torch.cuda.get_device_properties(0).total_memory)
            is_rocm = bool(getattr(torch.version, "hip", None))
            return {
                "kind": "rocm" if is_rocm else "cuda", "name": name,
                "memory": vram, "memory_human": human(vram), "unified": False,
                "runtimes": ["gguf"], "preferred_runtime": "gguf",
                "summary": f"{name} — {human(vram)} of dedicated video memory",
            }
    except Exception:
        pass

    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
            if out:
                name, mem_s = [x.strip() for x in out[0].split(",")]
                vram = int(re.sub(r"[^0-9]", "", mem_s)) * 1024 * 1024
                return {"kind": "cuda", "name": name, "memory": vram,
                        "memory_human": human(vram), "unified": False,
                        "runtimes": ["gguf"], "preferred_runtime": "gguf",
                        "summary": f"{name} — {human(vram)} of dedicated video memory"}
        except Exception:
            pass

    if system == "Darwin":          # Intel Mac: Metal exists but no MLX
        mem = total_memory()
        return {"kind": "intel_mac", "name": platform.processor() or "Intel Mac",
                "memory": mem, "memory_human": human(mem), "unified": False,
                "runtimes": ["gguf"], "preferred_runtime": "gguf",
                "summary": f"Intel Mac — limited Metal acceleration, {human(mem)} memory"}

    mem = total_memory()
    return {"kind": "cpu", "name": platform.processor() or platform.machine() or "CPU",
            "memory": mem, "memory_human": human(mem), "unified": False,
            "runtimes": ["gguf"], "preferred_runtime": "gguf",
            "summary": f"CPU only — no GPU acceleration detected, {human(mem)} memory"}


# -------------------------------------------------------------- model runtime

def classify_runtime(details: dict | None, name: str = "") -> dict:
    """Which runtime a model will use, from what Ollama reports about it."""
    details = details or {}
    fmt = str(details.get("format", "")).lower()
    quant = str(details.get("quantization_level", ""))
    lowered = f"{name} {quant}".lower()

    if fmt == "gguf":
        return {"runtime": "gguf", "format": fmt, "quantization": quant,
                "label": f"GGUF ({quant})" if quant else "GGUF"}
    if fmt in ("safetensors", "mlx") or "mlx" in lowered or quant.lower() in ("nvfp4", "fp4", "int4"):
        return {"runtime": "mlx", "format": fmt or "safetensors", "quantization": quant,
                "label": f"MLX ({quant})" if quant else "MLX"}
    return {"runtime": "unknown", "format": fmt or "?", "quantization": quant,
            "label": fmt or "unknown format"}


def assess_model(*, size_bytes: int = 0, parameter_count: int = 0,
                 details: dict | None = None, name: str = "",
                 accelerator: dict | None = None) -> dict:
    """How well a model suits this machine.

    level: 'good' | 'ok' | 'warn' | 'bad' | 'unknown'
    """
    accel = accelerator or detect_accelerator()
    rt = classify_runtime(details, name)
    label = name or "this model"
    reasons: list[str] = []
    level = "ok"

    # 1. Does the build match the accelerator? This dominates everything else.
    if rt["runtime"] == "mlx":
        if accel["kind"] == "apple_silicon":
            level = "good"
            reasons.append(f"MLX build — native to {accel['name']}, runs on the GPU")
        else:
            level = "bad"
            reasons.append(
                "MLX build — these are compiled for Apple Silicon and will not use "
                f"the GPU on {accel['name']}. Choose a GGUF build instead.")
    elif rt["runtime"] == "gguf":
        if accel["kind"] == "apple_silicon":
            reasons.append("GGUF build — GPU-accelerated through Metal")
        elif accel["kind"] in ("cuda", "rocm"):
            level = "good"
            reasons.append(f"GGUF build — GPU-accelerated on {accel['name']}")
        else:
            level = "warn"
            reasons.append("GGUF build — no GPU detected, so this runs on the CPU")
    else:
        reasons.append(f"unrecognised build ({rt['label']})")

    # 2. Is it heavy for this machine?
    accel_mem = accel.get("memory") or 0
    needed = int(size_bytes * OVERHEAD) if size_bytes else 0
    if needed and accel_mem:
        usable = accel_mem * (1 - HEADROOM_FRACTION)
        share = needed / accel_mem
        if needed > accel_mem:
            level = "bad"
            reasons.append(
                f"needs roughly {human(needed)} but this machine has {human(accel_mem)} "
                f"{'of unified memory' if accel.get('unified') else 'of video memory'}")
        elif needed > usable:
            if level != "bad":
                level = "warn"
            reasons.append(
                f"needs roughly {human(needed)} of {human(accel_mem)} — very little "
                f"room left for anything else")
        elif share > 0.5:
            if level not in ("bad", "warn"):
                level = "ok"
            reasons.append(f"fairly heavy: about {human(needed)} of {human(accel_mem)}")
        else:
            reasons.append(f"light: about {human(needed)} of {human(accel_mem)}")

    # 3. CPU-only machines struggle with large models regardless of memory.
    if accel["kind"] in ("cpu", "intel_mac") and parameter_count > CPU_PARAM_LIMIT:
        level = "bad" if level != "bad" else level
        reasons.append(
            f"{parameter_count/1e9:.1f}B parameters on a CPU will be extremely slow — "
            f"prefer a model under {CPU_PARAM_LIMIT/1e9:.0f}B")

    headline = {
        "good": "well matched to this machine",
        "ok": "should run fine here",
        "warn": "will run, but expect it to be slow",
        "bad": "not a good fit for this machine",
    }.get(level, "")

    return {"level": level, "headline": headline, "reasons": reasons,
            "runtime": rt, "needed": needed, "accelerator": accel["kind"],
            "message": f"{label}: {headline}. " + " · ".join(reasons)}


def running_offload(host_url: str | None = None) -> list[dict]:
    """Ground truth from Ollama: how much of each loaded model is on the GPU."""
    import requests

    if host_url is None:
        from . import ollama
        host_url = ollama.host()
    try:
        r = requests.get(f"{host_url}/api/ps", timeout=5)
        r.raise_for_status()
        out = []
        for m in r.json().get("models", []):
            size, vram = int(m.get("size", 0)), int(m.get("size_vram", 0))
            pct = (vram / size * 100) if size else 0
            out.append({"name": m.get("name", ""), "size": size, "size_vram": vram,
                        "gpu_percent": round(pct),
                        "note": ("fully on the GPU" if pct >= 99 else
                                 "entirely on the CPU" if pct < 1 else
                                 f"{pct:.0f}% on the GPU, the rest on the CPU")})
        return out
    except Exception:
        return []


def summary() -> dict:
    accel = detect_accelerator()
    total = total_memory()
    return {"accelerator": accel, "total": total, "total_human": human(total),
            "platform": platform.platform()}
