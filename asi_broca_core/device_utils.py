"""Device selection helpers (CUDA, Apple MPS, CPU)."""

from __future__ import annotations

import os

import torch


def normalize_device_arg(raw: str | None) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s in {"", "auto"}:
        return None
    return raw.strip()


def pick_torch_device(pref: str | None = None, *, preferred_order: tuple[str, ...] | None = None) -> torch.device:
    """Resolve execution device.

    ``pref`` is ``None``, ``''``, or ``'auto'`` → use *preferred_order* then fall back sensibly.

    Override with ``ASI_DEVICE`` when ``pref`` is auto-like (useful for CI).
    """

    normalized = normalize_device_arg(pref)
    env = os.environ.get("ASI_DEVICE", "").strip()
    pick = normalized if normalized is not None else (env or None)

    if pick:
        dev = torch.device(pick)
        # Fail fast where unavailable (except CPU)
        if dev.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")
        if dev.type == "mps":
            backend = getattr(torch.backends, "mps", None)
            if backend is None or not backend.is_available():
                raise RuntimeError("MPS device requested but MPS backend is unavailable on this Python/torch")
        return dev

    order = preferred_order or ("mps", "cuda")
    resolved: list[torch.device] = []
    for token in order:
        if token == "mps":
            backend = getattr(torch.backends, "mps", None)
            if backend is not None and backend.is_available():
                resolved.append(torch.device("mps"))
        elif token == "cuda" and torch.cuda.is_available():
            resolved.append(torch.device("cuda"))

    return resolved[0] if resolved else torch.device("cpu")


def inference_dtype(device: torch.device) -> torch.dtype:
    """Heuristic dtype for loading inference models on the given device."""
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


