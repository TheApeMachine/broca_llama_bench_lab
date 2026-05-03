"""Compatibility facade for active-inference agents and POMDP builders.

The implementation is split into small composable objects in this package:
``DistributionMath``, ``CategoricalPOMDP``, active/coupled agents, and POMDP
builder classes.  This module keeps the historical import surface stable by
exporting bound methods from :class:`ActiveInferenceFacade` instead of keeping
logic in module-level functions.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .active_agent import ActiveInferenceAgent
from .categorical_pomdp import CategoricalPOMDP
from .coupled_decision import CoupledDecision
from .coupled_efe_agent import CoupledEFEAgent
from .decision import Decision
from .distribution_math import DistributionMath
from .policy_evaluation import PolicyEvaluation
from .pomdp_builder import POMDPBuilder
from .tiger_door_env import TigerDoorEnv
from .tiger_episode_runner import TigerEpisodeRunner
from .tool_foraging_agent import ToolForagingAgent
from .tool_foraging_builder import ToolForagingPOMDPBuilder


class ActiveInferenceFacade:
    """Import-preserving surface over the active-inference object graph."""

    def __init__(self, math: DistributionMath | None = None) -> None:
        self.math = math or DistributionMath()
        self.pomdp_builder = POMDPBuilder(math=self.math)
        self.tool_foraging_builder = ToolForagingPOMDPBuilder(
            math=self.math,
            transitions=self.pomdp_builder,
        )
        self.tiger_runner = TigerEpisodeRunner()

    @property
    def epsilon(self) -> float:
        return self.math.epsilon

    @property
    def max_policy_enumeration(self) -> int:
        return CategoricalPOMDP.max_policy_enumeration

    def normalize(self, xs: Sequence[float]) -> list[float]:
        return self.math.normalize(xs)

    def entropy(self, p: Sequence[float]) -> float:
        return self.math.entropy(p)

    def kl(self, p: Sequence[float], q: Sequence[float]) -> float:
        return self.math.kl(p, q)

    def softmax_neg(self, xs: Sequence[float], precision: float = 1.0) -> list[float]:
        return self.math.softmax_neg(xs, precision)

    def build_causal_epistemic_pomdp(
        self,
        scm: Any,
        *,
        treatment: str = "T",
        outcome: str = "Y",
        outcome_hit: object = 1,
    ) -> CategoricalPOMDP:
        return self.pomdp_builder.build_causal_epistemic(
            scm,
            treatment=treatment,
            outcome=outcome,
            outcome_hit=outcome_hit,
        )

    def identity_transition(self, n_actions: int, n_states: int) -> list[list[list[float]]]:
        return self.pomdp_builder.identity_transition(n_actions, n_states)

    def derived_listen_channel_reliability(self, *, n_hidden_states: int) -> float:
        return self.pomdp_builder.listen_channel_reliability(n_hidden_states=n_hidden_states)

    def build_tiger_pomdp(self) -> CategoricalPOMDP:
        return self.pomdp_builder.build_tiger()

    def run_episode(
        self,
        agent: ActiveInferenceAgent,
        env: TigerDoorEnv,
        *,
        max_steps: int = 3,
    ) -> tuple[bool, float, list[dict]]:
        return self.tiger_runner.run(agent, env, max_steps=max_steps)

    def random_episode(self, env: TigerDoorEnv, *, max_steps: int = 3) -> tuple[bool, float]:
        return self.tiger_runner.random(env, max_steps=max_steps)

    def tool_foraging_likelihoods(self, *, n_existing_tools: int) -> list[list[list[float]]]:
        return self.tool_foraging_builder.likelihoods(n_existing_tools=n_existing_tools)

    def build_tool_foraging_pomdp(
        self,
        *,
        n_existing_tools: int = 0,
        insufficient_prior: float = 0.5,
    ) -> CategoricalPOMDP:
        return self.tool_foraging_builder.build(
            n_existing_tools=n_existing_tools,
            insufficient_prior=insufficient_prior,
        )

    def extend_pomdp_with_synthesize_tool(
        self,
        pomdp: CategoricalPOMDP,
        *,
        n_existing_tools: int = 0,
    ) -> CategoricalPOMDP:
        return self.pomdp_builder.with_synthesize_tool(
            pomdp,
            n_existing_tools=n_existing_tools,
        )


_FACADE = ActiveInferenceFacade()
_EPS = _FACADE.epsilon
MAX_POLICY_ENUMERATION = _FACADE.max_policy_enumeration
TOOL_FORAGING_STATES = ToolForagingPOMDPBuilder.states
TOOL_FORAGING_ACTIONS = ToolForagingPOMDPBuilder.actions
TOOL_FORAGING_OBSERVATIONS = ToolForagingPOMDPBuilder.observations

normalize = _FACADE.normalize
entropy = _FACADE.entropy
kl = _FACADE.kl
softmax_neg = _FACADE.softmax_neg
build_causal_epistemic_pomdp = _FACADE.build_causal_epistemic_pomdp
identity_transition = _FACADE.identity_transition
derived_listen_channel_reliability = _FACADE.derived_listen_channel_reliability
build_tiger_pomdp = _FACADE.build_tiger_pomdp
run_episode = _FACADE.run_episode
random_episode = _FACADE.random_episode
_tool_foraging_likelihoods = _FACADE.tool_foraging_likelihoods
build_tool_foraging_pomdp = _FACADE.build_tool_foraging_pomdp
extend_pomdp_with_synthesize_tool = _FACADE.extend_pomdp_with_synthesize_tool

__all__ = [
    "ActiveInferenceAgent",
    "ActiveInferenceFacade",
    "CategoricalPOMDP",
    "CoupledDecision",
    "CoupledEFEAgent",
    "Decision",
    "DistributionMath",
    "MAX_POLICY_ENUMERATION",
    "POMDPBuilder",
    "PolicyEvaluation",
    "TOOL_FORAGING_ACTIONS",
    "TOOL_FORAGING_OBSERVATIONS",
    "TOOL_FORAGING_STATES",
    "TigerDoorEnv",
    "TigerEpisodeRunner",
    "ToolForagingAgent",
    "ToolForagingPOMDPBuilder",
    "build_causal_epistemic_pomdp",
    "build_tiger_pomdp",
    "build_tool_foraging_pomdp",
    "derived_listen_channel_reliability",
    "entropy",
    "extend_pomdp_with_synthesize_tool",
    "identity_transition",
    "kl",
    "normalize",
    "random_episode",
    "run_episode",
    "softmax_neg",
]
