"""Shared invariant reporting contracts.

Invariant checks are intentionally small and explicit.  They turn mathematical
assumptions into executable contracts so runtime health reports can say exactly
which part of the substrate is valid, cold, degraded, or broken.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class InvariantViolation:
    """One failed invariant with enough context to fix it."""

    path: str
    message: str
    expected: str = ""
    observed: Any = None
    severity: str = "error"

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "message": self.message,
            "expected": self.expected,
            "observed": self.observed,
            "severity": self.severity,
        }


@dataclass(frozen=True)
class InvariantReport:
    """Result of validating a single object or subsystem."""

    name: str
    passed: bool
    violations: tuple[InvariantViolation, ...] = field(default_factory=tuple)

    @property
    def status(self) -> str:
        if self.passed:
            return "pass"
        if any(v.severity == "error" for v in self.violations):
            return "fail"
        return "warn"

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "passed": self.passed,
            "violations": [v.as_dict() for v in self.violations],
        }


class InvariantFailure(ValueError):
    """Raised when a mathematical contract fails in strict mode."""

    def __init__(self, report: InvariantReport) -> None:
        self.report = report
        joined = "\n".join(
            f"- {v.path}: {v.message} expected={v.expected!r} observed={v.observed!r}"
            for v in report.violations
        )
        super().__init__(f"Invariant report {report.name!r} failed:\n{joined}")
