"""Invariant checks for conformal calibration objects."""

from __future__ import annotations

import math
from typing import Any

from ..contracts import InvariantReport, InvariantViolation


class ConformalInvariants:
    """Validate predictor parameters and stored scores."""

    def validate(self, predictor: Any, *, name: str = "conformal") -> InvariantReport:
        violations: list[InvariantViolation] = []
        alpha = float(getattr(predictor, "alpha", float("nan")))
        if not math.isfinite(alpha) or not (0.0 < alpha < 1.0):
            violations.append(
                InvariantViolation(
                    f"{name}.alpha",
                    "alpha must be in (0, 1)",
                    expected="0 < alpha < 1",
                    observed=alpha,
                )
            )
        method = str(getattr(predictor, "method", ""))
        if method not in {"lac", "aps"}:
            violations.append(
                InvariantViolation(
                    f"{name}.method",
                    "unknown conformal method",
                    expected="lac or aps",
                    observed=method,
                )
            )
        for idx, score in enumerate(getattr(predictor, "scores", [])):
            s = float(score)
            if not math.isfinite(s) or s < 0.0:
                violations.append(
                    InvariantViolation(
                        f"{name}.scores[{idx}]",
                        "nonconformity score is invalid",
                        expected="finite non-negative score",
                        observed=s,
                    )
                )
        return InvariantReport(name, not violations, tuple(violations))
