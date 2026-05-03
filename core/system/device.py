"""Device selection helpers (CUDA, Apple MPS, CPU)."""

from __future__ import annotations

import torch


def pick_torch_device(device: str | None) -> torch.device:
    if device is None or str(device).strip() == "":
        if torch.cuda.is_available():
            return torch.device("cuda")
        
        if torch.backends.mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    return torch.device(device.strip().lower())

def inference_dtype(device: torch.device) -> torch.dtype:
    """Heuristic dtype for loading inference models on the given device."""
    if device.type == "cuda" and torch.cuda.is_bf16_supported(device):
        return torch.bfloat16

    if device.type == "mps" and torch.backends.mps.is_bf16_supported():
        return torch.bfloat16

    return torch.float32
