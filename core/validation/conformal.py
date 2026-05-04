"""Empirical validation helpers for conformal prediction channels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ConformalCoverageReport:
    """Held-out coverage and set-size summary for one predictor."""

    n_examples: int
    target_coverage: float
    empirical_coverage: float
    average_set_size: float
    calibration_size: int
    method: str

    @property
    def coverage_gap(self) -> float:
        return float(self.empirical_coverage - self.target_coverage)

    @property
    def status(self) -> str:
        if self.n_examples <= 0:
            return "empty"
        return "pass" if self.empirical_coverage + 1e-12 >= self.target_coverage else "undercovered"

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "n_examples": self.n_examples,
            "target_coverage": self.target_coverage,
            "empirical_coverage": self.empirical_coverage,
            "coverage_gap": self.coverage_gap,
            "average_set_size": self.average_set_size,
            "calibration_size": self.calibration_size,
            "method": self.method,
            "status": self.status,
        }


class ConformalCoverageEvaluator:
    """Measure conformal behavior on held-out labeled distributions."""

    def evaluate(
        self,
        predictor: object,
        examples: Sequence[tuple[Mapping[str, float], str]],
    ) -> ConformalCoverageReport:
        hits = 0
        total_size = 0
        for distribution, true_label in examples:
            result = predictor.predict_set(distribution)  # type: ignore[attr-defined]
            hits += int(str(true_label) in {str(label) for label in result.labels})
            total_size += int(result.set_size)
        n = len(examples)
        alpha = float(getattr(predictor, "alpha", 0.1))
        return ConformalCoverageReport(
            n_examples=n,
            target_coverage=1.0 - alpha,
            empirical_coverage=hits / max(1, n),
            average_set_size=total_size / max(1, n),
            calibration_size=len(getattr(predictor, "scores", [])),
            method=str(getattr(predictor, "method", "unknown")),
        )
