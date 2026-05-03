"""Device selection helpers (CUDA, Apple MPS, CPU)."""

from __future__ import annotations

from typing import Any

import torch


_DEVICE_ALIASES: dict[str, str] = {
    "auto": "",
    "default": "",
    "gpu": "cuda",
    "metal": "mps",
}


def normalize_device_arg(raw: str | torch.device | None = None) -> str | None:
    """Normalize a user/device argument into a torch device string.

    ``None``, ``""``, ``"auto"``, and ``"default"`` mean auto-select.  Explicit
    device requests fail loudly when the requested backend is unavailable so a
    run cannot silently migrate from CPU to GPU or vice versa.
    """

    if raw is None:
        return None
    if isinstance(raw, torch.device):
        value = str(raw)
    else:
        value = str(raw).strip().lower()
    value = _DEVICE_ALIASES.get(value, value)
    if value == "":
        return None

    try:
        device = torch.device(value)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"Unsupported torch device {raw!r}") from exc

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested, but torch.backends.mps.is_available() is false.")
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError(
            f"Unsupported torch device type {device.type!r}; expected one of cpu, cuda, mps."
        )
    return str(device)


def pick_torch_device(preferred: str | torch.device | None = None) -> torch.device:
    """Return an explicit or automatically selected torch device."""

    normalized = normalize_device_arg(preferred)
    if normalized is not None:
        return torch.device(normalized)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def inference_dtype(device: torch.device | str | None = None) -> torch.dtype:
    """Heuristic dtype for loading inference models on the given device."""

    dev = pick_torch_device(device) if device is not None else pick_torch_device()
    if dev.type == "cuda":
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            return torch.float16
        return torch.float16
    if dev.type == "mps":
        return torch.float16
    return torch.float32
