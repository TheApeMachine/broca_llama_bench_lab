"""Finite categorical POMDP used by active inference agents."""

from __future__ import annotations

import itertools
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from .distribution_math import DistributionMath
from .policy_evaluation import PolicyEvaluation

logger = logging.getLogger(__name__)


@dataclass
class CategoricalPOMDP:
    """Finite categorical active-inference model.

    ``A[action][observation][state]`` stores ``P(o | s, action)``.
    ``B[action][next_state][state]`` stores ``P(s' | s, action)``.
    ``C[observation]`` stores preferred observations.
    ``D[state]`` stores the prior over hidden states.
    """

    A: list[list[list[float]]]
    B: list[list[list[float]]]
    C: list[float]
    D: list[float]
    state_names: list[str]
    action_names: list[str]
    observation_names: list[str]
    a_counts: list[list[list[float]]] = field(default_factory=list)
    b_counts: list[list[list[float]]] = field(default_factory=list)

    math: DistributionMath = field(default_factory=DistributionMath, init=False, repr=False)
    default_count_strength: float = field(default=20.0, init=False, repr=False)
    count_epsilon: float = field(default=1e-3, init=False, repr=False)
    max_policy_enumeration: int = field(default=500_000, init=False, repr=False)

    def __post_init__(self) -> None:
        self.D = self.math.normalize(self.D)
        self.C = self.math.normalize(self.C)
        self._normalize_observation_likelihoods()
        self._normalize_transition_likelihoods()
        self._initialize_counts()
        from .invariants import POMDPInvariants

        POMDPInvariants().validate_or_raise(self, name="categorical_pomdp")

    @property
    def n_states(self) -> int:
        return len(self.state_names)

    @property
    def n_actions(self) -> int:
        return len(self.action_names)

    @property
    def n_observations(self) -> int:
        return len(self.observation_names)

    def predict_state(self, qs: Sequence[float], action: int) -> list[float]:
        out = []

        for sp in range(self.n_states):
            out.append(sum(self.B[action][sp][s] * qs[s] for s in range(self.n_states)))

        return self.math.normalize(out)

    def observation_distribution(self, qs: Sequence[float], action: int) -> list[float]:
        out = []

        for o in range(self.n_observations):
            out.append(sum(self.A[action][o][s] * qs[s] for s in range(self.n_states)))

        return self.math.normalize(out)

    def posterior_after_observation(
        self,
        qs_pred: Sequence[float],
        action: int,
        obs: int,
    ) -> list[float]:
        numerators = [self.A[action][obs][s] * qs_pred[s] for s in range(self.n_states)]

        return self.math.normalize(numerators)

    def ambiguity(self, qs: Sequence[float], action: int) -> float:
        total = 0.0

        for s in range(self.n_states):
            column = [self.A[action][o][s] for o in range(self.n_observations)]
            total += qs[s] * self.math.entropy(column)

        return total

    def epistemic_value(self, qs_pred: Sequence[float], action: int) -> float:
        observation_probabilities = self.observation_distribution(qs_pred, action)
        information = 0.0

        for o in range(self.n_observations):
            if observation_probabilities[o] <= self.math.epsilon:
                continue

            posterior = self.posterior_after_observation(qs_pred, action, o)
            information += observation_probabilities[o] * self.math.kl(posterior, qs_pred)

        return information

    def evaluate_policy(
        self,
        policy: Sequence[int],
        qs: Sequence[float] | None = None,
    ) -> PolicyEvaluation:
        q = list(self.D if qs is None else qs)
        risk = 0.0
        ambiguity_value = 0.0
        epistemic = 0.0

        for action in policy:
            q_pred = self.predict_state(q, action)
            po = self.observation_distribution(q_pred, action)
            risk += self.math.kl(po, self.C)
            ambiguity_value += self.ambiguity(q_pred, action)
            epistemic += self.epistemic_value(q_pred, action)
            q = q_pred

        expected_free_energy = risk + ambiguity_value - epistemic

        return PolicyEvaluation(
            tuple(policy),
            expected_free_energy,
            risk,
            ambiguity_value,
            epistemic,
        )

    def enumerate_policies(
        self,
        horizon: int,
        *,
        max_policies: int | None = None,
    ) -> Iterable[tuple[int, ...]]:
        """Yield all action sequences of length ``horizon``."""

        h = int(horizon)

        if h < 0:
            raise ValueError("horizon must be non-negative")

        n_actions = int(self.n_actions)
        cap = int(self.max_policy_enumeration if max_policies is None else max_policies)

        if cap < 1:
            raise ValueError("max_policies must be >= 1")

        total = n_actions**h if h > 0 else 1
        product = itertools.product(range(n_actions), repeat=h)

        if total > cap:
            logger.warning(
                "enumerate_policies: n_actions=%d horizon=%d yields %d policies (>%d); returning iterator not list",
                n_actions,
                h,
                total,
                cap,
            )

            return product

        return list(product)

    def learn_A(self, action: int, obs: int, qs_post: Sequence[float], lr: float = 1.0) -> None:
        for s, mass in enumerate(qs_post):
            self.a_counts[action][obs][s] += lr * float(mass)

        for s in range(self.n_states):
            column = self.math.normalize(
                [self.a_counts[action][o][s] for o in range(self.n_observations)]
            )

            for o, value in enumerate(column):
                self.A[action][o][s] = value

    def learn_B(
        self,
        action: int,
        qs_prev: Sequence[float],
        qs_post: Sequence[float],
        lr: float = 1.0,
    ) -> None:
        for s, before in enumerate(qs_prev):
            for sp, after in enumerate(qs_post):
                self.b_counts[action][sp][s] += lr * float(before) * float(after)

        for s in range(self.n_states):
            column = self.math.normalize(
                [self.b_counts[action][sp][s] for sp in range(self.n_states)]
            )

            for sp, value in enumerate(column):
                self.B[action][sp][s] = value

    def expand_state(
        self,
        new_state_name: str,
        *,
        qs: Sequence[float],
        predictive_mass_obs: float,
    ) -> list[float]:
        """Append a state; belief-steal mass scales with surprise."""

        uniform_observer = 1.0 / float(max(1, self.n_observations))
        mass_raw = (uniform_observer - float(predictive_mass_obs)) / max(
            uniform_observer,
            self.math.epsilon,
        )
        mass = float(min(0.45, max(self.math.epsilon * 1000.0, mass_raw)))

        return self.expand_state_with_mass(new_state_name, qs=qs, mass=mass)

    def expand_state_with_mass(
        self,
        new_state_name: str,
        *,
        qs: Sequence[float],
        mass: float = 0.08,
    ) -> list[float]:
        """Low-level grow operator used after expansion mass has been derived."""

        old_state_count = self.n_states
        action_count = self.n_actions
        observation_count = self.n_observations
        mass = float(max(self.math.epsilon * 100, min(0.45, mass)))
        new_qs = self.math.normalize([float(q) * (1.0 - mass) for q in qs] + [mass])

        self.state_names.append(new_state_name)
        self._expand_observation_model(old_state_count, action_count, observation_count)
        self._expand_transition_model(old_state_count, action_count)
        self.D = self.math.normalize([d * (1.0 - mass) for d in self.D] + [mass])
        self._expand_counts(old_state_count, action_count, observation_count)

        return new_qs

    def _normalize_observation_likelihoods(self) -> None:
        for a in range(len(self.A)):
            for s in range(len(self.state_names)):
                column = self.math.normalize(
                    [self.A[a][o][s] for o in range(len(self.observation_names))]
                )

                for o, value in enumerate(column):
                    self.A[a][o][s] = value

    def _normalize_transition_likelihoods(self) -> None:
        for a in range(len(self.B)):
            for s in range(len(self.state_names)):
                column = self.math.normalize(
                    [self.B[a][sp][s] for sp in range(len(self.state_names))]
                )

                for sp, value in enumerate(column):
                    self.B[a][sp][s] = value

    def _initialize_counts(self) -> None:
        if not self.a_counts:
            self.a_counts = [
                [
                    [
                        self.default_count_strength * self.A[a][o][s] + self.count_epsilon
                        for s in range(len(self.state_names))
                    ]
                    for o in range(len(self.observation_names))
                ]
                for a in range(len(self.action_names))
            ]

        if not self.b_counts:
            self.b_counts = [
                [
                    [
                        self.default_count_strength * self.B[a][sp][s] + self.count_epsilon
                        for s in range(len(self.state_names))
                    ]
                    for sp in range(len(self.state_names))
                ]
                for a in range(len(self.action_names))
            ]

    def _expand_observation_model(
        self,
        old_state_count: int,
        action_count: int,
        observation_count: int,
    ) -> None:
        for a in range(action_count):
            for o in range(observation_count):
                average = sum(self.A[a][o]) / max(old_state_count, 1)
                duplicate = self.A[a][o][old_state_count - 1]
                self.A[a][o].append(0.6 * duplicate + 0.4 * average)

            # A[action][observation][state] stores P(o | state, action), so
            # the new model must normalize over observations for each state.
            for s in range(old_state_count + 1):
                column = self.math.normalize(
                    [self.A[a][o][s] for o in range(observation_count)]
                )
                for o, value in enumerate(column):
                    self.A[a][o][s] = value

    def _expand_transition_model(self, old_state_count: int, action_count: int) -> None:
        for a in range(action_count):
            for sp in range(old_state_count):
                row = list(self.B[a][sp])
                row.append(0.5 * row[-1] + 0.5 / (old_state_count + 1))
                self.B[a][sp] = row

            new_row = self.math.normalize([1.0 / (old_state_count + 1)] * (old_state_count + 1))
            self.B[a].append(list(new_row))

            for s in range(old_state_count + 1):
                column = [self.B[a][sp][s] for sp in range(old_state_count + 1)]
                normalized = self.math.normalize(column)

                for sp in range(old_state_count + 1):
                    self.B[a][sp][s] = normalized[sp]

    def _expand_counts(
        self,
        old_state_count: int,
        action_count: int,
        observation_count: int,
    ) -> None:
        for a in range(action_count):
            for o in range(observation_count):
                self.a_counts[a][o].append(
                    self.default_count_strength * self.A[a][o][old_state_count] + self.count_epsilon
                )

            for sp in range(old_state_count):
                self.b_counts[a][sp].append(
                    self.default_count_strength * self.B[a][sp][old_state_count] + self.count_epsilon
                )

            self.b_counts[a].append(
                [
                    self.default_count_strength * self.B[a][old_state_count][s] + self.count_epsilon
                    for s in range(old_state_count + 1)
                ]
            )
