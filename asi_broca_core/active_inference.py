from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass, field
from typing import Sequence


_EPS = 1e-12


def normalize(xs: Sequence[float]) -> list[float]:
    s = float(sum(max(0.0, x) for x in xs))
    if s <= _EPS:
        return [1.0 / len(xs) for _ in xs]
    return [max(0.0, float(x)) / s for x in xs]


def entropy(p: Sequence[float]) -> float:
    return -sum(float(x) * math.log(max(float(x), _EPS)) for x in p)


def kl(p: Sequence[float], q: Sequence[float]) -> float:
    return sum(float(pi) * (math.log(max(float(pi), _EPS)) - math.log(max(float(qi), _EPS))) for pi, qi in zip(p, q))


def softmax_neg(xs: Sequence[float], precision: float = 1.0) -> list[float]:
    vals = [-precision * float(x) for x in xs]
    m = max(vals)
    exps = [math.exp(v - m) for v in vals]
    z = sum(exps)
    return [e / z for e in exps]


@dataclass
class PolicyEvaluation:
    policy: tuple[int, ...]
    expected_free_energy: float
    risk: float
    ambiguity: float
    epistemic_value: float


@dataclass
class Decision:
    action: int
    action_name: str
    qs: list[float]
    policies: list[PolicyEvaluation]
    posterior_over_policies: list[float]


@dataclass
class CategoricalPOMDP:
    """Finite categorical active-inference model.

    A[action][observation][state] = P(o | s, action)
    B[action][next_state][state] = P(s' | s, action)
    C[observation] = preferred observation distribution
    D[state] = prior over hidden states
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

    def __post_init__(self) -> None:
        self.D = normalize(self.D)
        self.C = normalize(self.C)
        for a in range(len(self.A)):
            for s in range(len(self.state_names)):
                col = normalize([self.A[a][o][s] for o in range(len(self.observation_names))])
                for o, val in enumerate(col):
                    self.A[a][o][s] = val
        for a in range(len(self.B)):
            for s in range(len(self.state_names)):
                col = normalize([self.B[a][sp][s] for sp in range(len(self.state_names))])
                for sp, val in enumerate(col):
                    self.B[a][sp][s] = val
        if not self.a_counts:
            self.a_counts = [[[20.0 * self.A[a][o][s] + 1e-3 for s in range(len(self.state_names))] for o in range(len(self.observation_names))] for a in range(len(self.action_names))]
        if not self.b_counts:
            self.b_counts = [[[20.0 * self.B[a][sp][s] + 1e-3 for s in range(len(self.state_names))] for sp in range(len(self.state_names))] for a in range(len(self.action_names))]

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
        return normalize(out)

    def observation_distribution(self, qs: Sequence[float], action: int) -> list[float]:
        out = []
        for o in range(self.n_observations):
            out.append(sum(self.A[action][o][s] * qs[s] for s in range(self.n_states)))
        return normalize(out)

    def posterior_after_observation(self, qs_pred: Sequence[float], action: int, obs: int) -> list[float]:
        numer = [self.A[action][obs][s] * qs_pred[s] for s in range(self.n_states)]
        return normalize(numer)

    def ambiguity(self, qs: Sequence[float], action: int) -> float:
        total = 0.0
        for s in range(self.n_states):
            col = [self.A[action][o][s] for o in range(self.n_observations)]
            total += qs[s] * entropy(col)
        return total

    def epistemic_value(self, qs_pred: Sequence[float], action: int) -> float:
        po = self.observation_distribution(qs_pred, action)
        info = 0.0
        for o in range(self.n_observations):
            if po[o] <= _EPS:
                continue
            post = self.posterior_after_observation(qs_pred, action, o)
            info += po[o] * kl(post, qs_pred)
        return info

    def evaluate_policy(self, policy: Sequence[int], qs: Sequence[float] | None = None) -> PolicyEvaluation:
        q = list(self.D if qs is None else qs)
        risk = 0.0
        ambiguity_value = 0.0
        epistemic = 0.0
        for action in policy:
            q_pred = self.predict_state(q, action)
            po = self.observation_distribution(q_pred, action)
            risk += kl(po, self.C)
            ambiguity_value += self.ambiguity(q_pred, action)
            epistemic += self.epistemic_value(q_pred, action)
            q = q_pred
        G = risk + ambiguity_value - epistemic
        return PolicyEvaluation(tuple(policy), G, risk, ambiguity_value, epistemic)

    def enumerate_policies(self, horizon: int) -> list[tuple[int, ...]]:
        return list(itertools.product(range(self.n_actions), repeat=int(horizon)))

    def learn_A(self, action: int, obs: int, qs_post: Sequence[float], lr: float = 1.0) -> None:
        for s, mass in enumerate(qs_post):
            self.a_counts[action][obs][s] += lr * float(mass)
        for s in range(self.n_states):
            col = normalize([self.a_counts[action][o][s] for o in range(self.n_observations)])
            for o, val in enumerate(col):
                self.A[action][o][s] = val

    def learn_B(self, action: int, qs_prev: Sequence[float], qs_post: Sequence[float], lr: float = 1.0) -> None:
        for s, before in enumerate(qs_prev):
            for sp, after in enumerate(qs_post):
                self.b_counts[action][sp][s] += lr * float(before) * float(after)
        for s in range(self.n_states):
            col = normalize([self.b_counts[action][sp][s] for sp in range(self.n_states)])
            for sp, val in enumerate(col):
                self.B[action][sp][s] = val


@dataclass
class ActiveInferenceAgent:
    pomdp: CategoricalPOMDP
    horizon: int = 1
    precision: float = 8.0
    learn: bool = True
    qs: list[float] | None = None

    def __post_init__(self) -> None:
        if self.qs is None:
            self.qs = list(self.pomdp.D)

    def reset_belief(self) -> None:
        self.qs = list(self.pomdp.D)

    def decide(self) -> Decision:
        assert self.qs is not None
        evals = [self.pomdp.evaluate_policy(pol, self.qs) for pol in self.pomdp.enumerate_policies(self.horizon)]
        posterior = softmax_neg([e.expected_free_energy for e in evals], self.precision)
        best_index = max(range(len(evals)), key=lambda i: posterior[i])
        action = evals[best_index].policy[0]
        return Decision(action, self.pomdp.action_names[action], list(self.qs), evals, posterior)

    def update(self, action: int, obs: int, lr: float = 1.0) -> list[float]:
        assert self.qs is not None
        before = list(self.qs)
        pred = self.pomdp.predict_state(before, action)
        post = self.pomdp.posterior_after_observation(pred, action, obs)
        if self.learn:
            self.pomdp.learn_A(action, obs, post, lr=lr)
            self.pomdp.learn_B(action, before, post, lr=0.25 * lr)
        self.qs = post
        return post


def identity_transition(n_actions: int, n_states: int) -> list[list[list[float]]]:
    return [
        [[1.0 if sp == s else 0.0 for s in range(n_states)] for sp in range(n_states)]
        for _ in range(n_actions)
    ]


def build_tiger_pomdp(reliability: float = 0.85) -> CategoricalPOMDP:
    states = ["left", "right"]
    actions = ["listen", "open_left", "open_right"]
    observations = ["hear_left", "hear_right", "reward", "punish"]
    r = float(reliability)
    # A[action][obs][state]
    listen = [
        [r, 1.0 - r],       # hear_left
        [1.0 - r, r],       # hear_right
        [0.0, 0.0],         # reward
        [0.0, 0.0],         # punish
    ]
    open_left = [
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, 0.0],         # reward if treasure/tiger state is left
        [0.0, 1.0],
    ]
    open_right = [
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
    ]
    B = identity_transition(3, 2)
    # Preferences: reward is strongly preferred; punishment is strongly avoided;
    # hearing observations are acceptable but not goals.
    C = [0.30, 0.30, 0.68, 0.02]
    D = [0.5, 0.5]
    return CategoricalPOMDP([listen, open_left, open_right], B, C, D, states, actions, observations)


class TigerDoorEnv:
    """Noisy listen/open environment for active-inference experiments."""

    def __init__(self, reliability: float = 0.85, seed: int = 0):
        self.reliability = float(reliability)
        self.rng = random.Random(seed)
        self.hidden_state = 0

    def reset(self) -> int:
        self.hidden_state = 0 if self.rng.random() < 0.5 else 1
        return self.hidden_state

    def step(self, action_name: str) -> tuple[str, float, bool]:
        s = self.hidden_state
        if action_name == "listen":
            if self.rng.random() < self.reliability:
                obs = "hear_left" if s == 0 else "hear_right"
            else:
                obs = "hear_right" if s == 0 else "hear_left"
            return obs, -0.01, False
        if action_name == "open_left":
            return ("reward", 1.0, True) if s == 0 else ("punish", -2.0, True)
        if action_name == "open_right":
            return ("reward", 1.0, True) if s == 1 else ("punish", -2.0, True)
        raise KeyError(action_name)


def run_episode(agent: ActiveInferenceAgent, env: TigerDoorEnv, *, max_steps: int = 3) -> tuple[bool, float, list[dict]]:
    pomdp = agent.pomdp
    env.reset()
    agent.reset_belief()
    trace = []
    total = 0.0
    success = False
    for _ in range(max_steps):
        d = agent.decide()
        obs_name, reward, done = env.step(d.action_name)
        obs = pomdp.observation_names.index(obs_name)
        post = agent.update(d.action, obs)
        total += reward
        if obs_name == "reward":
            success = True
        trace.append(
            {
                "action": d.action_name,
                "observation": obs_name,
                "reward": reward,
                "posterior": {name: round(float(p), 3) for name, p in zip(pomdp.state_names, post)},
            }
        )
        if done:
            break
    return success, total, trace


def random_episode(env: TigerDoorEnv, *, max_steps: int = 3) -> tuple[bool, float]:
    env.reset()
    total = 0.0
    success = False
    for _ in range(max_steps):
        # Let random act in the same action space, including the possibility of wasting a listen.
        action = env.rng.choice(["listen", "open_left", "open_right"])
        obs, reward, done = env.step(action)
        total += reward
        success = success or obs == "reward"
        if done:
            break
    return success, total


