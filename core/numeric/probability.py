"""Probability and confidence math."""

from __future__ import annotations

import math
from collections.abc import Sequence


class Probability:
    """Named probability calculations used across substrate policies."""

    def unit_interval(self, value: float) -> float:
        probability = float(value)

        if not math.isfinite(probability):
            raise ValueError(f"probability is not finite: {probability}")
        
        return max(0.0, min(1.0, probability))

    def inverse_cardinality(self, size: int) -> float:
        if int(size) <= 0:
            return 1.0
        
        return 1.0 / float(size)

    def entropy(self, probabilities: Sequence[float]) -> float:
        total = 0.0
        
        for probability in probabilities:
            p = float(probability)
        
            if not math.isfinite(p):
                raise ValueError(f"probability is not finite: {p}")
        
            if p < 0.0:
                raise ValueError(f"probability is negative: {p}")
        
            if p > 0.0:
                total -= p * math.log(p)
        
        return total

    def normalized_entropy(self, probabilities: Sequence[float]) -> float:
        n = len(probabilities)

        if n < 2:
            return 1.0

        h_max = math.log(float(n))

        if h_max <= 1e-9:
            return 1.0

        return self.unit_interval(self.entropy(probabilities) / h_max)

    def confidence_damping(self, confidence: float) -> float:
        return max(1e-3, 1.0 - 0.6 * self.unit_interval(confidence))

    def temperature_scale(self, *, confidence: float, posterior: Sequence[float]) -> float:
        return max(
            1e-3,
            self.normalized_entropy(posterior) * self.confidence_damping(confidence),
        )
