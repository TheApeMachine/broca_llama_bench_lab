"""Distribution math for categorical active-inference models."""

from __future__ import annotations

import math
from collections.abc import Sequence


class DistributionMath:
    """Reusable operations over finite categorical distributions."""

    epsilon = 1e-12

    def normalize(self, values: Sequence[float]) -> list[float]:
        """Clamp negatives to zero and return a normalized probability vector."""

        total = float(sum(max(0.0, float(value)) for value in values))

        if total <= self.epsilon:
            return [1.0 / len(values) for _ in values]

        return [max(0.0, float(value)) / total for value in values]

    def entropy(self, probabilities: Sequence[float]) -> float:
        """Return Shannon entropy for a categorical probability vector."""

        return -sum(
            float(probability) * math.log(max(float(probability), self.epsilon))
            for probability in probabilities
        )

    def kl(self, p: Sequence[float], q: Sequence[float]) -> float:
        """Return ``KL(p || q)`` over a shared finite support."""

        if len(p) != len(q):
            raise ValueError(
                f"kl: length mismatch len(p)={len(p)} len(q)={len(q)}; distributions must have the same support size"
            )

        return sum(
            float(pi)
            * (
                math.log(max(float(pi), self.epsilon))
                - math.log(max(float(qi), self.epsilon))
            )
            for pi, qi in zip(p, q)
        )

    def softmax_neg(self, values: Sequence[float], precision: float = 1.0) -> list[float]:
        """Softmax over negative energy values."""

        shifted_values = [-float(precision) * float(value) for value in values]
        maximum = max(shifted_values)
        exponentials = [math.exp(value - maximum) for value in shifted_values]
        total = sum(exponentials)

        return [value / total for value in exponentials]

    def unit_clamped(self, value: float) -> float:
        """Clamp a scalar into the open-ish unit interval used by POMDP builders."""

        return float(max(self.epsilon, min(1.0 - self.epsilon, float(value))))
