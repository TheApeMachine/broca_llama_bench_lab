"""Finite active-inference controller."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .categorical_pomdp import CategoricalPOMDP
from .decision import Decision
from .distribution_math import DistributionMath

logger = logging.getLogger(__name__)


@dataclass
class ActiveInferenceAgent:
    """Controller that chooses actions by minimizing expected free energy."""

    pomdp: CategoricalPOMDP
    horizon: int = 1
    learn: bool = True
    qs: list[float] | None = None
    expand_on_surprise: bool = False
    _expand_serial: int = field(default=0, repr=False)
    _math: DistributionMath = field(default_factory=DistributionMath, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.qs is None:
            self.qs = list(self.pomdp.D)

    def reset_belief(self) -> None:
        self.qs = list(self.pomdp.D)

    def decide(self) -> Decision:
        if self.qs is None:
            raise RuntimeError(
                "ActiveInferenceAgent.qs is not initialized; cannot run decide() "
                "(evaluate_policy / enumerate_policies require beliefs)."
            )

        evaluations = [
            self.pomdp.evaluate_policy(policy, self.qs)
            for policy in self.pomdp.enumerate_policies(self.horizon)
        ]
        g_values = [evaluation.expected_free_energy for evaluation in evaluations]
        spread = float(max(g_values) - min(g_values))
        precision = (1.0 / max(spread, self._math.epsilon)) if spread > self._math.epsilon else float(len(evaluations))
        posterior = self._math.softmax_neg(g_values, precision)
        best_index = max(range(len(evaluations)), key=lambda index: posterior[index])
        chosen_policy = evaluations[best_index].policy

        if not chosen_policy:
            action: int | None = None
            action_name = ""
        else:
            action = chosen_policy[0]
            action_name = self.pomdp.action_names[action]

        logger.debug(
            "ActiveInferenceAgent.decide: action=%s min_G=%.4f n_policies=%d horizon=%d qs=%s",
            f"{action_name!s}({action})" if action is not None else "none",
            min(g_values),
            len(evaluations),
            self.horizon,
            [round(q, 4) for q in self.qs],
        )

        return Decision(action, action_name, list(self.qs), evaluations, posterior)

    def update(self, action: int, obs: int, lr: float = 1.0) -> list[float]:
        if self.qs is None:
            raise RuntimeError("ActiveInferenceAgent.qs is not initialized; cannot run update().")

        before = list(self.qs)
        prediction = self.pomdp.predict_state(before, action)
        observation_probabilities = self.pomdp.observation_distribution(prediction, action)
        expanded = False
        uniform_floor = 1.0 / float(max(1, self.pomdp.n_observations))

        if self.expand_on_surprise and observation_probabilities[obs] < uniform_floor:
            label = f"hyp_{self.pomdp.n_states}_{self._expand_serial}"
            self._expand_serial += 1
            self.qs = self.pomdp.expand_state(
                label,
                qs=before,
                predictive_mass_obs=float(observation_probabilities[obs]),
            )
            prediction = self.pomdp.predict_state(self.qs, action)
            expanded = True

        posterior = self.pomdp.posterior_after_observation(prediction, action, obs)

        if self.learn:
            self.pomdp.learn_A(action, obs, posterior, lr=lr)

            if not expanded:
                self.pomdp.learn_B(action, before, posterior, lr=0.25 * lr)

        self.qs = posterior
        logger.debug(
            "ActiveInferenceAgent.update: action=%s obs=%d expanded=%s post=%s",
            self.pomdp.action_names[action],
            obs,
            expanded,
            [round(float(probability), 4) for probability in posterior],
        )

        return posterior
