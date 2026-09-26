"""Torch/CUDA helpers that degrade gracefully when torch is absent."""

from __future__ import annotations

from typing import Any


def _torch() -> Any | None:
    try:
        import torch  # noqa: PLC0415 - optional at import time (tests run without CUDA)
    except ImportError:
        return None
    return torch


def cuda_available() -> bool:
    torch = _torch()
    return bool(torch and torch.cuda.is_available())


def vram_snapshot(device: str = "cuda") -> dict[str, int | None]:
    """Total/used/free/reserved VRAM in bytes, or ``None`` values when unavailable."""
    empty = {"total_bytes": None, "used_bytes": None, "free_bytes": None, "reserved_bytes": None}
    torch = _torch()
    if torch is None or not device.startswith("cuda") or not torch.cuda.is_available():
        return empty
    free, total = torch.cuda.mem_get_info()
    return {
        "total_bytes": int(total),
        "used_bytes": int(total - free),
        "free_bytes": int(free),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
    }


def reset_peak_vram() -> None:
    torch = _torch()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_vram_bytes() -> int | None:
    torch = _torch()
    if torch is None or not torch.cuda.is_available():
        return None
    return int(torch.cuda.max_memory_allocated())


def empty_cache() -> None:
    torch = _torch()
    if torch is None or not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def is_cuda_oom(exc: BaseException) -> bool:
    """True for CUDA OOM, including the ``torch.cuda.OutOfMemoryError`` subclass."""
    torch = _torch()
    if torch is not None:
        oom_error = getattr(torch.cuda, "OutOfMemoryError", None)
        if oom_error is not None and isinstance(exc, oom_error):
            return True
    message = str(exc).lower()
    return "out of memory" in message or "cuda oom" in message


def resolve_dtype(precision: str) -> Any:
    torch = _torch()
    if torch is None:
        raise RuntimeError("PyTorch is not installed")
    mapping = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    try:
        return mapping[precision]
    except KeyError as exc:
        raise ValueError(f"Unsupported precision: {precision!r}") from exc
