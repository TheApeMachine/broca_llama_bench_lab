"""Active-inference decision record."""

from __future__ import annotations

from dataclasses import dataclass

from .policy_evaluation import PolicyEvaluation


@dataclass
class Decision:
    """Chosen action plus belief and policy posterior diagnostics."""

    action: int | None
    action_name: str
    qs: list[float]
    policies: list[PolicyEvaluation]
    posterior_over_policies: list[float]
