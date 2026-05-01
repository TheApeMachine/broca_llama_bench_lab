from __future__ import annotations

import torch.nn as nn


def freeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def count_parameters(module: nn.Module) -> tuple[int, int]:
    total = 0
    trainable = 0
    for p in module.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return total, trainable
