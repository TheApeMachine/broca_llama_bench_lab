"""Active-inference agent for native-tool foraging."""

from __future__ import annotations

from dataclasses import dataclass, field

from .active_agent import ActiveInferenceAgent
from .categorical_pomdp import CategoricalPOMDP
from .decision import Decision
from .distribution_math import DistributionMath
from .tool_foraging_builder import ToolForagingPOMDPBuilder


@dataclass
class ToolForagingAgent:
    """Agent specialised for the synthesize_tool decision."""

    pomdp: CategoricalPOMDP
    agent: ActiveInferenceAgent = field(init=False)
    math: DistributionMath = field(default_factory=DistributionMath, init=False, repr=False)

    def __post_init__(self) -> None:
        self.agent = ActiveInferenceAgent(self.pomdp, horizon=1, learn=False)

    @classmethod
    def build(
        cls,
        *,
        n_existing_tools: int = 0,
        insufficient_prior: float = 0.5,
    ) -> "ToolForagingAgent":
        builder = ToolForagingPOMDPBuilder()

        return cls(
            pomdp=builder.build(
                n_existing_tools=int(n_existing_tools),
                insufficient_prior=float(insufficient_prior),
            )
        )

    def update_belief(self, *, insufficient_prior: float) -> None:
        """Set prior over knowledge_insufficient ahead of the next decision."""

        insufficient = self.math.unit_clamped(insufficient_prior)
        self.pomdp.D = self.math.normalize([1.0 - insufficient, insufficient])
        self.agent.qs = list(self.pomdp.D)

    def decide(self) -> Decision:
        return self.agent.decide()

    def should_synthesize(self) -> bool:
        decision = self.decide()

        return decision.action_name == "synthesize_tool"

    def observe(self, action_name: str, observation_name: str, *, lr: float = 1.0) -> list[float]:
        """Update belief after seeing a real-world foraging observation."""

        normalized_action = str(action_name)
        normalized_observation = str(observation_name)

        if normalized_action not in self.pomdp.action_names:
            raise ValueError(
                f"observe: unknown action_name {normalized_action!r}; valid actions: {list(self.pomdp.action_names)}"
            )

        if normalized_observation not in self.pomdp.observation_names:
            raise ValueError(
                f"observe: unknown observation_name {normalized_observation!r}; valid observations: {list(self.pomdp.observation_names)}"
            )

        action = self.pomdp.action_names.index(normalized_action)
        observation = self.pomdp.observation_names.index(normalized_observation)

        return self.agent.update(action, observation, lr=lr)
