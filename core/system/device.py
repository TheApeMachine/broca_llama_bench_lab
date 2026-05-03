"""Device selection helpers (CUDA, Apple MPS, CPU)."""

from __future__ import annotations

import torch


def pick_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")

def inference_dtype(device: torch.device) -> torch.dtype:
    """Heuristic dtype for loading inference models on the given device."""
    if device.type == "cuda" and torch.cuda.is_bf16_supported(device):
        return torch.bfloat16

    return torch.float32
