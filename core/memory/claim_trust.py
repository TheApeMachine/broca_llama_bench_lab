"""ClaimTrust — prediction-error-weighted consolidation weights.

Belief revision in :class:`SymbolicMemory` weighs each candidate claim by a
Gaussian likelihood ratio over the population's surprise distribution. The
two methods here are the entire scoring policy:

* :meth:`population_stats` summarizes ``(mean, std)`` of ``prediction_gap``
  across a claim corpus. Returns ``None`` when there is not enough variance
  to anchor a Z-score.
* :meth:`weight` returns ``exp(-0.5 z^2)`` where ``z`` is the claim's gap
  under the population. Without a population (``stats is None``) the
  formula collapses to a unit-Gaussian baseline.

The class is stateless. It exists so the rule "everything is a method on a
class, no loose functions" applies to a piece of math that two layers
(consolidation and DMN) reach for.
"""

from __future__ import annotations

import math
from typing import Sequence


class ClaimTrust:
    """Stateless wrapper for prediction-error-weighted trust scoring."""

    @classmethod
    def population_stats(cls, claims: Sequence[dict]) -> tuple[float, float] | None:
        """Mean and std of ``prediction_gap`` across ``claims``.

        Filters out non-dict evidence, non-numeric gaps, and zero-or-below
        values. Returns ``None`` when fewer than two valid samples are
        present or the variance is degenerate.
        """

        gaps: list[float] = []
        for claim in claims:
            ev = claim.get("evidence")
            if not isinstance(ev, dict):
                continue
            raw = ev.get("prediction_gap")
            try:
                gap = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(gap) and gap > 0.0:
                gaps.append(gap)
        if len(gaps) < 2:
            return None
        mu = sum(gaps) / len(gaps)
        var = sum((g - mu) ** 2 for g in gaps) / len(gaps)
        sigma = math.sqrt(var)
        if sigma <= 1e-6:
            return None
        return mu, sigma

    @classmethod
    def weight(cls, claim: dict, *, stats: tuple[float, float] | None = None) -> float:
        """Likelihood-ratio weight for one claim under the population's surprise."""

        ev = claim.get("evidence")
        if not isinstance(ev, dict):
            return 1.0
        raw = ev.get("prediction_gap")
        try:
            gap = float(raw)
        except (TypeError, ValueError):
            return 1.0
        if not math.isfinite(gap) or gap <= 0.0:
            return 1.0
        if stats is None:
            mu, sigma = 0.0, 1.0
        else:
            mu, sigma = stats
            sigma = max(1e-3, float(sigma))
        z = max(0.0, (gap - mu) / sigma)
        return math.exp(-0.5 * z * z)
