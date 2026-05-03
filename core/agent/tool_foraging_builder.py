"""POMDP builder for native-tool foraging decisions."""

from __future__ import annotations

import logging

from .categorical_pomdp import CategoricalPOMDP
from .distribution_math import DistributionMath
from .pomdp_builder import POMDPBuilder

logger = logging.getLogger(__name__)


class ToolForagingPOMDPBuilder:
    """Build the synthesize-vs-use-existing-tool POMDP."""

    states: tuple[str, ...] = ("knowledge_sufficient", "knowledge_insufficient")
    actions: tuple[str, ...] = (
        "use_existing_tool",
        "explore_memory",
        "synthesize_tool",
    )
    observations: tuple[str, ...] = ("info_gained", "info_stagnant")

    def __init__(
        self,
        *,
        math: DistributionMath | None = None,
        transitions: POMDPBuilder | None = None,
    ) -> None:
        self.math = math if math is not None else DistributionMath()
        self.transitions = transitions if transitions is not None else POMDPBuilder(math=self.math)

    def build(
        self,
        *,
        n_existing_tools: int = 0,
        insufficient_prior: float = 0.5,
    ) -> CategoricalPOMDP:
        """Build a POMDP whose minimal-EFE action can synthesize a new tool."""

        insufficient = self.math.unit_clamped(insufficient_prior)
        likelihoods = self.likelihoods(n_existing_tools=int(n_existing_tools))
        transitions = self.transitions.identity_transition(len(self.actions), len(self.states))
        preferences = self.math.normalize([0.85, 0.15])
        priors = self.math.normalize([1.0 - insufficient, insufficient])
        pomdp = CategoricalPOMDP(
            list(likelihoods),
            list(transitions),
            list(preferences),
            list(priors),
            list(self.states),
            list(self.actions),
            list(self.observations),
        )

        logger.debug(
            "build_tool_foraging_pomdp: n_tools=%d insufficient_prior=%.4f coverage_signal=%.4f",
            int(n_existing_tools),
            insufficient,
            self.coverage_signal(n_existing_tools=n_existing_tools),
        )

        return pomdp

    def likelihoods(self, *, n_existing_tools: int) -> list[list[list[float]]]:
        """Construct ``A[action][observation][state]`` for the foraging POMDP."""

        coverage = self.coverage_signal(n_existing_tools=n_existing_tools)
        use_gain_sufficient = max(0.5 + self.math.epsilon, 0.5 + 0.45 * coverage)
        use_gain_insufficient = 0.20
        explore_gain_sufficient = 0.55
        explore_gain_insufficient = 0.40
        synthesize_gain_sufficient = 0.30
        synthesize_gain_insufficient = 0.85

        return [
            [
                [use_gain_sufficient, use_gain_insufficient],
                [1.0 - use_gain_sufficient, 1.0 - use_gain_insufficient],
            ],
            [
                [explore_gain_sufficient, explore_gain_insufficient],
                [1.0 - explore_gain_sufficient, 1.0 - explore_gain_insufficient],
            ],
            [
                [synthesize_gain_sufficient, synthesize_gain_insufficient],
                [1.0 - synthesize_gain_sufficient, 1.0 - synthesize_gain_insufficient],
            ],
        ]

    def coverage_signal(self, *, n_existing_tools: int) -> float:
        n = max(0, int(n_existing_tools))

        return 1.0 - 1.0 / (1.0 + n)
