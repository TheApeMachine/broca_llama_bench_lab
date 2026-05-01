from __future__ import annotations

import torch.nn as nn


def freeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def count_parameters(module: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable
