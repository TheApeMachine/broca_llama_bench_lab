"""Composition root for coupled active-inference faculties."""

from __future__ import annotations

import logging

from .active_agent import ActiveInferenceAgent
from .coupled_decision import CoupledDecision

logger = logging.getLogger(__name__)


class CoupledEFEAgent:
    """Pick the faculty whose minimal one-step Expected Free Energy is lower."""

    def __init__(self, spatial: ActiveInferenceAgent, causal: ActiveInferenceAgent) -> None:
        self.spatial = spatial
        self.causal = causal

    def decide(self) -> CoupledDecision:
        spatial_decision = self.spatial.decide()
        causal_decision = self.causal.decide()
        spatial_min_g = min(evaluation.expected_free_energy for evaluation in spatial_decision.policies)
        causal_min_g = min(evaluation.expected_free_energy for evaluation in causal_decision.policies)

        if spatial_min_g <= causal_min_g:
            logger.debug(
                "CoupledEFEAgent.decide: faculty=spatial action=%s G_spa=%.4f G_cau=%.4f (spatial wins tie)",
                spatial_decision.action_name,
                spatial_min_g,
                causal_min_g,
            )

            return CoupledDecision(
                "spatial",
                spatial_decision.action_name,
                spatial_decision,
                causal_decision,
                spatial_min_g,
                causal_min_g,
            )

        logger.debug(
            "CoupledEFEAgent.decide: faculty=causal action=%s G_spa=%.4f G_cau=%.4f",
            causal_decision.action_name,
            spatial_min_g,
            causal_min_g,
        )

        return CoupledDecision(
            "causal",
            causal_decision.action_name,
            spatial_decision,
            causal_decision,
            spatial_min_g,
            causal_min_g,
        )
