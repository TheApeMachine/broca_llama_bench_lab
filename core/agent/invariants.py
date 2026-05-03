"""Invariant checks for finite categorical POMDPs."""

from __future__ import annotations

import math
from typing import Any, Sequence

from ..contracts import InvariantFailure, InvariantReport, InvariantViolation


class POMDPInvariants:
    """Validate the probability-simplex contracts of a categorical POMDP.

    The project stores observation and transition models in column-major form:
    ``A[action][observation][state] = P(o | s, action)`` and
    ``B[action][next_state][state] = P(s' | s, action)``.  Therefore every
    action/state column must be a non-negative distribution summing to one.
    """

    def __init__(self, *, atol: float = 1e-6) -> None:
        self.atol = float(atol)

    def validate(self, pomdp: Any, *, name: str = "pomdp") -> InvariantReport:
        violations: list[InvariantViolation] = []
        n_actions = int(getattr(pomdp, "n_actions"))
        n_states = int(getattr(pomdp, "n_states"))
        n_obs = int(getattr(pomdp, "n_observations"))
        violations.extend(self._shape_checks(pomdp, n_actions, n_states, n_obs, name=name))
        if violations:
            return InvariantReport(name, False, tuple(violations))

        for a in range(n_actions):
            for s in range(n_states):
                column = [float(pomdp.A[a][o][s]) for o in range(n_obs)]
                violations.extend(
                    self._distribution_checks(
                        column,
                        path=f"{name}.A[action={a}][state={s}]",
                        expected="sum_o P(o | state, action) == 1 and all probabilities finite/non-negative",
                    )
                )

        for a in range(n_actions):
            for s in range(n_states):
                column = [float(pomdp.B[a][sp][s]) for sp in range(n_states)]
                violations.extend(
                    self._distribution_checks(
                        column,
                        path=f"{name}.B[action={a}][state={s}]",
                        expected="sum_next_state P(s' | state, action) == 1 and all probabilities finite/non-negative",
                    )
                )

        violations.extend(
            self._distribution_checks(
                [float(x) for x in pomdp.C],
                path=f"{name}.C",
                expected="preferences are a finite probability distribution",
            )
        )
        violations.extend(
            self._distribution_checks(
                [float(x) for x in pomdp.D],
                path=f"{name}.D",
                expected="prior is a finite probability distribution",
            )
        )
        return InvariantReport(name, not violations, tuple(violations))

    def validate_or_raise(self, pomdp: Any, *, name: str = "pomdp") -> InvariantReport:
        report = self.validate(pomdp, name=name)
        if not report.passed:
            raise InvariantFailure(report)
        return report

    def _shape_checks(
        self,
        pomdp: Any,
        n_actions: int,
        n_states: int,
        n_obs: int,
        *,
        name: str,
    ) -> list[InvariantViolation]:
        out: list[InvariantViolation] = []
        if len(getattr(pomdp, "A")) != n_actions:
            out.append(
                InvariantViolation(
                    f"{name}.A",
                    "observation model action dimension mismatch",
                    expected=str(n_actions),
                    observed=len(getattr(pomdp, "A")),
                )
            )
        if len(getattr(pomdp, "B")) != n_actions:
            out.append(
                InvariantViolation(
                    f"{name}.B",
                    "transition model action dimension mismatch",
                    expected=str(n_actions),
                    observed=len(getattr(pomdp, "B")),
                )
            )
        for a, rows in enumerate(getattr(pomdp, "A")):
            if len(rows) != n_obs:
                out.append(
                    InvariantViolation(
                        f"{name}.A[action={a}]",
                        "observation row count mismatch",
                        expected=str(n_obs),
                        observed=len(rows),
                    )
                )
                continue
            for o, row in enumerate(rows):
                if len(row) != n_states:
                    out.append(
                        InvariantViolation(
                            f"{name}.A[action={a}][observation={o}]",
                            "state column count mismatch",
                            expected=str(n_states),
                            observed=len(row),
                        )
                    )
        for a, rows in enumerate(getattr(pomdp, "B")):
            if len(rows) != n_states:
                out.append(
                    InvariantViolation(
                        f"{name}.B[action={a}]",
                        "next-state row count mismatch",
                        expected=str(n_states),
                        observed=len(rows),
                    )
                )
                continue
            for sp, row in enumerate(rows):
                if len(row) != n_states:
                    out.append(
                        InvariantViolation(
                            f"{name}.B[action={a}][next_state={sp}]",
                            "state column count mismatch",
                            expected=str(n_states),
                            observed=len(row),
                        )
                    )
        if len(getattr(pomdp, "C")) != n_obs:
            out.append(
                InvariantViolation(
                    f"{name}.C",
                    "preference vector length mismatch",
                    expected=str(n_obs),
                    observed=len(getattr(pomdp, "C")),
                )
            )
        if len(getattr(pomdp, "D")) != n_states:
            out.append(
                InvariantViolation(
                    f"{name}.D",
                    "prior vector length mismatch",
                    expected=str(n_states),
                    observed=len(getattr(pomdp, "D")),
                )
            )
        return out

    def _distribution_checks(
        self,
        values: Sequence[float],
        *,
        path: str,
        expected: str,
    ) -> list[InvariantViolation]:
        out: list[InvariantViolation] = []
        if not values:
            return [InvariantViolation(path, "empty distribution", expected=expected, observed=[])]
        bad = [x for x in values if not math.isfinite(float(x)) or float(x) < -self.atol]
        if bad:
            out.append(
                InvariantViolation(
                    path,
                    "distribution contains non-finite or negative values",
                    expected=expected,
                    observed=[float(x) for x in values],
                )
            )
        total = sum(max(0.0, float(x)) for x in values)
        if not math.isclose(total, 1.0, abs_tol=self.atol, rel_tol=0.0):
            out.append(
                InvariantViolation(
                    path,
                    "distribution does not sum to one",
                    expected=expected,
                    observed=round(total, 12),
                )
            )
        return out
