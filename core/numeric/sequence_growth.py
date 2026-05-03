"""Sequence-length growth calculations."""

from __future__ import annotations

import math


class SequenceGrowth:
    """Named growth curves for autoregressive sequence state."""

    def inertia(self, length: int) -> float:
        return math.log1p(float(max(0, int(length))))
