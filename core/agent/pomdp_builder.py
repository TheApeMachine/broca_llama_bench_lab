"""Builders for reusable active-inference POMDP structures."""

from __future__ import annotations

import logging
from typing import Any

from .categorical_pomdp import CategoricalPOMDP
from .distribution_math import DistributionMath

logger = logging.getLogger(__name__)


class POMDPBuilder:
    """Factory for common finite categorical POMDP layouts."""

    def __init__(self, *, math: DistributionMath | None = None) -> None:
        self.math = math if math is not None else DistributionMath()

    def identity_transition(self, n_actions: int, n_states: int) -> list[list[list[float]]]:
        return [
            [[1.0 if sp == s else 0.0 for s in range(n_states)] for sp in range(n_states)]
            for _ in range(n_actions)
        ]

    def listen_channel_reliability(self, *, n_hidden_states: int) -> float:
        """Listen diagonal slack tied to latent cardinality."""

        denominator = float(max(1, 2 * int(n_hidden_states) * int(n_hidden_states)))

        return float(
            max(
                0.5 + self.math.epsilon,
                min(1.0 - self.math.epsilon, 1.0 - 1.0 / denominator),
            )
        )

    def build_tiger(self) -> CategoricalPOMDP:
        states = ["left", "right"]
        actions = ["listen", "open_left", "open_right"]
        observations = ["hear_left", "hear_right", "reward", "punish"]
        reliability = self.listen_channel_reliability(n_hidden_states=len(states))
        listen = [
            [reliability, 1.0 - reliability],
            [1.0 - reliability, reliability],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
        open_left = [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
        open_right = [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ]
        transitions = self.identity_transition(3, 2)
        preferences = [0.30, 0.30, 0.68, 0.02]
        priors = [0.5, 0.5]

        return CategoricalPOMDP(
            [listen, open_left, open_right],
            transitions,
            preferences,
            priors,
            states,
            actions,
            observations,
        )

    def build_causal_epistemic(
        self,
        scm: Any,
        *,
        treatment: str = "T",
        outcome: str = "Y",
        outcome_hit: object = 1,
    ) -> CategoricalPOMDP:
        """Build a POMDP where actions distinguish association from intervention."""

        from ..causal import FiniteSCM

        if not isinstance(scm, FiniteSCM):
            raise TypeError("scm must be a FiniteSCM")

        p_y_do_t1 = scm.probability({outcome: outcome_hit}, given={}, interventions={treatment: 1})
        p_y_do_t0 = scm.probability({outcome: outcome_hit}, given={}, interventions={treatment: 0})
        p_y_t1 = scm.probability({outcome: outcome_hit}, given={treatment: 1}, interventions={})
        p_y_t0 = scm.probability({outcome: outcome_hit}, given={treatment: 0}, interventions={})
        obs_positive_gradient = (p_y_t1 - p_y_t0) >= 0.0
        delta_observed = abs(p_y_t1 - p_y_t0)
        delta_intervened = abs(p_y_do_t1 - p_y_do_t0)
        association_strength = 0.5 + 0.5 * min(1.0, abs(delta_observed - delta_intervened))
        trial_strength = max(association_strength, 0.5 + 0.5 * min(1.0, delta_intervened))
        observation_rows = self._causal_observation_rows(
            obs_positive_gradient=obs_positive_gradient,
            association_strength=association_strength,
        )
        trial_rows = self._causal_trial_rows(trial_strength=trial_strength)
        states = ["ate_non_negative", "ate_negative"]
        actions = ["observe_association", "run_intervention_readout"]
        observations = ["signal_matches_intervention", "signal_mismatch_intervention"]
        transitions = self.identity_transition(2, 2)
        intervention_margin = abs(p_y_do_t1 - p_y_do_t0)
        match_preference = 0.5 + 0.5 * min(1.0, intervention_margin)
        preferences = self.math.normalize([match_preference, max(self.math.epsilon, 1.0 - match_preference)])
        priors = [0.5, 0.5]
        pomdp = CategoricalPOMDP(
            [observation_rows, trial_rows],
            transitions,
            preferences,
            priors,
            states,
            actions,
            observations,
        )

        logger.debug(
            "build_causal_epistemic_pomdp: p_y|do(T=1)=%.4f p_y|do(T=0)=%.4f p_y|T=1=%.4f p_y|T=0=%.4f obs_grad_pos=%s margin_do=%.4f",
            p_y_do_t1,
            p_y_do_t0,
            p_y_t1,
            p_y_t0,
            obs_positive_gradient,
            intervention_margin,
        )

        return pomdp

    def with_synthesize_tool(
        self,
        pomdp: CategoricalPOMDP,
        *,
        n_existing_tools: int = 0,
    ) -> CategoricalPOMDP:
        """Return a new POMDP with the original action space plus synthesize_tool."""

        n_states = pomdp.n_states
        n_observations = pomdp.n_observations
        new_action = self._synthesize_likelihood_action(pomdp, n_existing_tools=n_existing_tools)
        new_A = [
            [[pomdp.A[a][o][s] for s in range(n_states)] for o in range(n_observations)]
            for a in range(pomdp.n_actions)
        ]
        new_A.append(new_action)
        new_B = [
            [[pomdp.B[a][sp][s] for s in range(n_states)] for sp in range(n_states)]
            for a in range(pomdp.n_actions)
        ]
        new_B.append(
            [[1.0 if sp == s else 0.0 for s in range(n_states)] for sp in range(n_states)]
        )

        return CategoricalPOMDP(
            new_A,
            new_B,
            list(pomdp.C),
            list(pomdp.D),
            list(pomdp.state_names),
            list(pomdp.action_names) + ["synthesize_tool"],
            list(pomdp.observation_names),
        )

    def _causal_observation_rows(
        self,
        *,
        obs_positive_gradient: bool,
        association_strength: float,
    ) -> list[list[float]]:
        rows: list[list[float]] = []

        for obs_index in range(2):
            column_over_states: list[float] = []

            for state_index in range(2):
                world_positive = state_index == 0
                aligned_read = obs_positive_gradient == world_positive
                p_match = association_strength if aligned_read else (1.0 - association_strength)
                column_over_states.append(p_match if obs_index == 0 else (1.0 - p_match))

            rows.append(column_over_states)

        return rows

    def _causal_trial_rows(self, *, trial_strength: float) -> list[list[float]]:
        rows: list[list[float]] = []

        for obs_index in range(2):
            row: list[float] = []

            for state_index in range(2):
                if obs_index == 0:
                    row.append(trial_strength if state_index == 0 else (1.0 - trial_strength))
                else:
                    row.append((1.0 - trial_strength) if state_index == 0 else trial_strength)

            rows.append(row)

        return rows

    def _synthesize_likelihood_action(
        self,
        pomdp: CategoricalPOMDP,
        *,
        n_existing_tools: int,
    ) -> list[list[float]]:
        n_states = pomdp.n_states
        n_observations = pomdp.n_observations
        coverage = 1.0 - 1.0 / (1.0 + int(max(0, n_existing_tools)))
        coverage_weight = 0.5 + 0.5 * coverage
        by_state: list[list[float]] = []

        for s in range(n_states):
            per_observation = [
                max(pomdp.A[a][o][s] for a in range(pomdp.n_actions))
                for o in range(n_observations)
            ]
            boosted = [
                coverage_weight * value + (1.0 - coverage_weight) * (1.0 / n_observations)
                for value in per_observation
            ]
            by_state.append(boosted)

        return [[by_state[s][o] for s in range(n_states)] for o in range(n_observations)]
