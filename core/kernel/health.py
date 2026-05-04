"""Runtime health reports combining capabilities and mathematical invariants."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..agent.invariants import POMDPInvariants
from ..calibration.invariants import ConformalInvariants
from ..causal.invariants import SCMInvariants
from ..contracts import InvariantReport, InvariantViolation
from .capabilities import CapabilityReport
from .manifest import RuntimeManifest
from .profiles import manifest_for_profile


@dataclass(frozen=True)
class SystemHealth:
    """Full health view for a constructed runtime."""

    capabilities: CapabilityReport
    invariants: tuple[InvariantReport, ...] = field(default_factory=tuple)

    @property
    def status(self) -> str:
        if self.capabilities.status == "fail" or any(r.status == "fail" for r in self.invariants):
            return "fail"
        if self.capabilities.status == "warn" or any(r.status == "warn" for r in self.invariants):
            return "warn"
        return "pass"

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "capabilities": self.capabilities.as_dict(),
            "invariants": [report.as_dict() for report in self.invariants],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent, sort_keys=True, default=str)

    def table_lines(self) -> list[str]:
        lines = [f"System health: {self.status}"]
        lines.extend(f"  {line}" for line in self.capabilities.table_lines())
        lines.append("  Invariants:")
        if not self.invariants:
            lines.append("    none recorded")
        for report in self.invariants:
            lines.append(f"    {report.name:<28} {report.status}")
            for violation in report.violations:
                lines.append(f"      - {violation.path}: {violation.message} observed={violation.observed!r}")
        return lines

    @classmethod
    def from_controller(
        cls,
        controller: Any,
        *,
        manifest: RuntimeManifest | None = None,
    ) -> "SystemHealth":
        manifest = manifest or manifest_for_profile("full")
        capabilities = CapabilityReport.from_controller(controller, manifest)
        reports: list[InvariantReport] = []
        reports.extend(_safe_pomdp_reports(controller))
        reports.extend(_safe_scm_reports(controller))
        reports.extend(_safe_conformal_reports(controller))
        return cls(capabilities=capabilities, invariants=tuple(reports))


def _safe_pomdp_reports(controller: Any) -> list[InvariantReport]:
    out: list[InvariantReport] = []
    checker = POMDPInvariants()
    for attr in ("pomdp", "causal_pomdp"):
        if not hasattr(controller, attr):
            continue
        try:
            out.append(checker.validate(getattr(controller, attr), name=attr))
        except Exception as exc:
            out.append(_exception_report(attr, exc))
    return out


def _safe_scm_reports(controller: Any) -> list[InvariantReport]:
    if not hasattr(controller, "scm"):
        return []
    try:
        return [SCMInvariants().validate(getattr(controller, "scm"), name="scm")]
    except Exception as exc:
        return [_exception_report("scm", exc)]


def _safe_conformal_reports(controller: Any) -> list[InvariantReport]:
    out: list[InvariantReport] = []
    checker = ConformalInvariants()
    for attr in ("relation_conformal", "native_tool_conformal"):
        if not hasattr(controller, attr):
            continue
        try:
            out.append(checker.validate(getattr(controller, attr), name=attr))
        except Exception as exc:
            out.append(_exception_report(attr, exc))
    return out


def _exception_report(name: str, exc: Exception) -> InvariantReport:
    return InvariantReport(
        name,
        False,
        (
            InvariantViolation(
                path=name,
                message="invariant checker raised an exception",
                expected="checker completes without exception",
                observed=repr(exc),
            ),
        ),
    )
