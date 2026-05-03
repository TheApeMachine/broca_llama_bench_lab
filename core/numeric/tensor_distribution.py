"""Tensor distribution math."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class TensorDistribution:
    """Named distribution calculations for logit tensors."""

    def peakedness(self, logits: torch.Tensor) -> torch.Tensor:
        log_probabilities = F.log_softmax(logits, dim=-1)
        probabilities = log_probabilities.exp()
        entropy = -(probabilities * log_probabilities).sum(dim=-1)
        log_cardinality = math.log(max(2, int(logits.shape[-1])))
        return (1.0 - entropy / log_cardinality).clamp(0.0, 1.0)
