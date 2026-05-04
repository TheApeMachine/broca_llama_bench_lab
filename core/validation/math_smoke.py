"""Static math validation suite that does not load external models."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..agent.active_inference import build_tiger_pomdp
from ..agent.invariants import POMDPInvariants
from ..calibration.conformal import ConformalPredictor
from ..calibration.invariants import ConformalInvariants
from ..causal import build_simpson_scm
from ..causal.invariants import SCMInvariants
from ..contracts import InvariantReport
from .active_inference import ActiveInferenceValidator


@dataclass(frozen=True)
class StaticMathValidation:
    """Bundle of math checks suitable for CI and CLI smoke runs."""

    invariants: tuple[InvariantReport, ...] = field(default_factory=tuple)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        if any(report.status == "fail" for report in self.invariants):
            return "fail"
        if any(report.status == "warn" for report in self.invariants):
            return "warn"
        metric_statuses = [str(v.get("status")) for v in self.metrics.values() if isinstance(v, dict)]
        if any(status in {"regressed", "undercovered", "invalid_model"} for status in metric_statuses):
            return "warn"
        return "pass"

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "invariants": [report.as_dict() for report in self.invariants],
            "metrics": self.metrics,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent, sort_keys=True, default=str)

    def table_lines(self) -> list[str]:
        lines = [f"Static math validation: {self.status}"]
        for report in self.invariants:
            lines.append(f"  {report.name:<28} {report.status}")
            for violation in report.violations:
                lines.append(f"    - {violation.path}: {violation.message} observed={violation.observed!r}")
        for name, metric in self.metrics.items():
            status = metric.get("status", "unknown") if isinstance(metric, dict) else "unknown"
            lines.append(f"  metric.{name:<21} {status} {metric}")
        return lines

    @classmethod
    def run(cls, *, include_tiger_metric: bool = True) -> "StaticMathValidation":
        reports: list[InvariantReport] = []
        pomdp = build_tiger_pomdp()
        reports.append(POMDPInvariants().validate(pomdp, name="tiger_pomdp"))
        pomdp.expand_state_with_mass("validation_hypothesis", qs=list(pomdp.D), mass=0.08)
        reports.append(POMDPInvariants().validate(pomdp, name="expanded_tiger_pomdp"))
        scm = build_simpson_scm()
        reports.append(SCMInvariants().validate(scm, name="simpson_scm"))
        lac = ConformalPredictor(alpha=0.1, method="lac", min_calibration=8)
        aps = ConformalPredictor(alpha=0.1, method="aps", min_calibration=8)
        reports.append(ConformalInvariants().validate(lac, name="cold_lac"))
        reports.append(ConformalInvariants().validate(aps, name="cold_aps"))
        cold_aps = aps.predict_set({"a": 0.7, "b": 0.2, "c": 0.1})
        metrics: dict[str, Any] = {
            "cold_aps_set": {
                "labels": list(cold_aps.labels),
                "set_size": int(cold_aps.set_size),
                "status": "pass" if cold_aps.set_size == 3 else "undercovered",
            },
        }
        if include_tiger_metric:
            metrics["tiger_active_inference"] = ActiveInferenceValidator().tiger_smoke(episodes=16).as_dict()
        return cls(tuple(reports), metrics)
