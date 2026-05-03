"""Coupled faculty decision record."""

from __future__ import annotations

from dataclasses import dataclass

from .decision import Decision


@dataclass
class CoupledDecision:
    """Decision made by comparing two EFE-driven faculties."""

    faculty: str
    action_name: str
    spatial_decision: Decision
    causal_decision: Decision
    spatial_min_G: float
    causal_min_G: float
