"""Invariant checks for finite structural causal models."""

from __future__ import annotations

import math
from typing import Any

from ..contracts import InvariantFailure, InvariantReport, InvariantViolation


class SCMInvariants:
    """Validate structural and probability contracts of a ``FiniteSCM``."""

    def __init__(self, *, atol: float = 1e-6) -> None:
        self.atol = float(atol)

    def validate(self, scm: Any, *, name: str = "scm") -> InvariantReport:
        violations: list[InvariantViolation] = []
        domains = getattr(scm, "domains", {})
        exogenous = getattr(scm, "exogenous", {})
        equations = getattr(scm, "equations", {})
        order = list(getattr(scm, "order", ()))

        for var, domain in domains.items():
            if not tuple(domain):
                violations.append(
                    InvariantViolation(
                        f"{name}.domains[{var!r}]",
                        "domain is empty",
                        expected="non-empty finite domain",
                        observed=list(domain),
                    )
                )

        for var, probs in exogenous.items():
            if var not in domains:
                violations.append(
                    InvariantViolation(
                        f"{name}.exogenous[{var!r}]",
                        "exogenous variable missing domain",
                        expected="domain declared in scm.domains",
                    )
                )
            total = 0.0
            for value, p in dict(probs).items():
                pf = float(p)
                total += pf
                if value not in domains.get(var, ()):  # type: ignore[arg-type]
                    violations.append(
                        InvariantViolation(
                            f"{name}.exogenous[{var!r}][{value!r}]",
                            "probability assigned to value outside domain",
                            expected=list(domains.get(var, ())),
                            observed=value,
                        )
                    )
                if not math.isfinite(pf) or pf < -self.atol:
                    violations.append(
                        InvariantViolation(
                            f"{name}.exogenous[{var!r}][{value!r}]",
                            "probability is non-finite or negative",
                            expected="finite non-negative probability",
                            observed=pf,
                        )
                    )
            if not math.isclose(total, 1.0, abs_tol=self.atol, rel_tol=0.0):
                violations.append(
                    InvariantViolation(
                        f"{name}.exogenous[{var!r}]",
                        "exogenous distribution does not sum to one",
                        expected="sum == 1",
                        observed=round(total, 12),
                    )
                )

        seen_order: set[str] = set()
        for var in order:
            if var in seen_order:
                violations.append(
                    InvariantViolation(
                        f"{name}.order",
                        "endogenous variable appears more than once",
                        expected="unique topological order",
                        observed=var,
                    )
                )
            seen_order.add(var)
            if var not in equations:
                violations.append(
                    InvariantViolation(
                        f"{name}.order[{var!r}]",
                        "ordered endogenous variable has no equation",
                        expected="equation exists",
                    )
                )
            if var not in domains:
                violations.append(
                    InvariantViolation(
                        f"{name}.order[{var!r}]",
                        "ordered endogenous variable has no domain",
                        expected="domain exists",
                    )
                )

        for var, eq in equations.items():
            if var not in domains:
                violations.append(
                    InvariantViolation(
                        f"{name}.equations[{var!r}]",
                        "equation output variable has no domain",
                        expected="domain exists",
                    )
                )
            for parent in getattr(eq, "parents", ()):  # EndogenousEquation.parents
                if parent not in domains:
                    violations.append(
                        InvariantViolation(
                            f"{name}.equations[{var!r}].parents",
                            "equation parent has no domain",
                            expected="all parents declared before use",
                            observed=parent,
                        )
                    )
        return InvariantReport(name, not violations, tuple(violations))

    def validate_or_raise(self, scm: Any, *, name: str = "scm") -> InvariantReport:
        report = self.validate(scm, name=name)
        if not report.passed:
            raise InvariantFailure(report)
        return report
