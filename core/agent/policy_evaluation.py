"""Policy evaluation record."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PolicyEvaluation:
    """Expected free energy decomposition for one action sequence."""

    policy: tuple[int, ...]
    expected_free_energy: float
    risk: float
    ambiguity: float
    epistemic_value: float
