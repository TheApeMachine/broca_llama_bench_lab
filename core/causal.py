from __future__ import annotations

import itertools
import logging
import math
import random
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Sequence


_EPS = 1e-12

logger = logging.getLogger(__name__)


@dataclass
class EndogenousEquation:
    name: str
    parents: tuple[str, ...]
    fn: Callable[[dict], object]


@dataclass
class FiniteSCM:
    """Finite structural causal model with exact enumeration.

    Exogenous variables are independent roots. Endogenous variables are
    deterministic functions of their parents and exogenous noise. Interventions
    replace structural equations with constants; counterfactuals are computed by
    abduction over exogenous worlds, action, then prediction.

    Exact queries iterate over **all** exogenous worlds via ``itertools.product``.
    That is tractable only for toy cardinalities (see ``exogenous_world_volume``).
    For larger discrete models use ``probability_monte_carlo`` / sampled counterfactuals.
    """

    domains: dict[str, tuple]
    exogenous: dict[str, dict[object, float]] = field(default_factory=dict)
    equations: dict[str, EndogenousEquation] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    labels: dict[str, str] = field(default_factory=dict)

    def add_exogenous(self, name: str, domain: Sequence, probs: Mapping[object, float] | None = None) -> None:
        dom = tuple(domain)
        if probs is None:
            p = {x: 1.0 / len(dom) for x in dom}
        else:
            z = float(sum(probs.values()))
            p = {x: float(probs.get(x, 0.0)) / z for x in dom}
        self.domains[name] = dom
        self.exogenous[name] = p

    def add_endogenous(self, name: str, domain: Sequence, parents: Sequence[str], fn: Callable[[dict], object]) -> None:
        self.domains[name] = tuple(domain)
        self.equations[name] = EndogenousEquation(name, tuple(parents), fn)
        self.order.append(name)

    @property
    def endogenous_names(self) -> tuple[str, ...]:
        return tuple(self.order)

    @property
    def observed_names(self) -> tuple[str, ...]:
        return tuple(self.order)

    def graph_parents(self, *, include_exogenous: bool = False) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        if include_exogenous:
            for u in self.exogenous:
                out[u] = []
        for name in self.order:
            parents = list(self.equations[name].parents)
            if not include_exogenous:
                parents = [p for p in parents if p not in self.exogenous]
            out[name] = parents
        return out

    def _exogenous_worlds(self):
        names = list(self.exogenous)
        domains = [list(self.exogenous[n]) for n in names]
        for vals in itertools.product(*domains):
            p = 1.0
            world = {}
            for n, v in zip(names, vals):
                p *= self.exogenous[n][v]
                world[n] = v
            yield world, p

    def _sample_exogenous_world(self, rng: random.Random) -> dict[str, object]:
        world: dict[str, object] = {}
        for name in self.exogenous:
            dom = list(self.exogenous[name].keys())
            weights = [float(self.exogenous[name][x]) for x in dom]
            world[name] = rng.choices(dom, weights=weights, k=1)[0]
        return world

    def evaluate_world(self, exo: Mapping[str, object], interventions: Mapping[str, object] | None = None) -> dict:
        interventions = dict(interventions or {})
        values = dict(exo)
        for name in self.order:
            if name in interventions:
                values[name] = interventions[name]
            else:
                values[name] = self.equations[name].fn(values)
            if values[name] not in self.domains[name]:
                raise ValueError(f"{name} returned value {values[name]!r}, outside domain {self.domains[name]!r}")
        return values

    def probability(
        self,
        event: Mapping[str, object],
        *,
        given: Mapping[str, object] | None = None,
        interventions: Mapping[str, object] | None = None,
    ) -> float:
        given = dict(given or {})
        num = 0.0
        den = 0.0
        for exo, p in self._exogenous_worlds():
            vals = self.evaluate_world(exo, interventions)
            if all(vals.get(k) == v for k, v in given.items()):
                den += p
                if all(vals.get(k) == v for k, v in event.items()):
                    num += p
        if den <= _EPS:
            logger.debug(
                "FiniteSCM.probability: p=0.0 event=%s given=%s do=%s (empty conditioning mass)",
                dict(event),
                given,
                dict(interventions or {}),
            )
            return 0.0
        prob = num / den
        logger.debug(
            "FiniteSCM.probability: p=%.8f event=%s given=%s do=%s worlds=%d",
            prob,
            dict(event),
            given,
            dict(interventions or {}),
            self.exogenous_world_volume,
        )
        return prob

    @property
    def exogenous_world_volume(self) -> int:
        """Product of exogenous domain sizes — enumeration cost scales with this."""

        vol = 1
        for name in self.exogenous:
            vol *= len(self.exogenous[name])
        return vol

    def probability_monte_carlo(
        self,
        event: Mapping[str, object],
        *,
        given: Mapping[str, object] | None = None,
        interventions: Mapping[str, object] | None = None,
        n_samples: int = 10_000,
        seed: int = 0,
    ) -> float:
        """Monte Carlo estimate of conditional ``P(event | given, do(interventions))``.

        Samples exogenous worlds from the factorized prior; keeps draws that satisfy
        ``given``. Rare conditioning sets inflate variance — increase ``n_samples``.
        """

        rng = random.Random(int(seed))
        given_d = dict(given or {})
        num = 0
        den = 0
        for _ in range(max(1, int(n_samples))):
            world = self._sample_exogenous_world(rng)
            vals = self.evaluate_world(world, interventions)
            if all(vals.get(k) == v for k, v in given_d.items()):
                den += 1
                if all(vals.get(k) == v for k, v in event.items()):
                    num += 1
        if den <= 0:
            return 0.0
        return num / den

    def distribution(self, variables: Sequence[str], *, given: Mapping[str, object] | None = None, interventions: Mapping[str, object] | None = None) -> dict[tuple, float]:
        out: dict[tuple, float] = {}
        den = 0.0
        given = dict(given or {})
        for exo, p in self._exogenous_worlds():
            vals = self.evaluate_world(exo, interventions)
            if all(vals.get(k) == v for k, v in given.items()):
                key = tuple(vals[v] for v in variables)
                out[key] = out.get(key, 0.0) + p
                den += p
        if den <= _EPS:
            return {}
        return {k: v / den for k, v in out.items()}

    def counterfactual_probability(
        self,
        query_event: Mapping[str, object],
        *,
        evidence: Mapping[str, object],
        interventions: Mapping[str, object],
        n_samples: int = 4_000,
        seed: int = 0,
        burn_in_sweeps: int = 4,
        init_max_draws: int | None = None,
    ) -> float:
        """Sampled abduction-action-prediction counterfactual probability.

        Runtime Rung-3 queries deliberately use sampled abduction, regardless of
        graph size, so the default path never depends on Cartesian enumeration
        over exogenous worlds. Use ``counterfactual_probability_exact`` only for
        tests or hand-checking tiny SCMs.
        """

        return self.counterfactual_probability_monte_carlo(
            query_event,
            evidence=evidence,
            interventions=interventions,
            n_samples=int(n_samples),
            seed=seed,
            burn_in_sweeps=int(burn_in_sweeps),
            init_max_draws=init_max_draws,
        )

    def counterfactual_probability_exact(
        self,
        query_event: Mapping[str, object],
        *,
        evidence: Mapping[str, object],
        interventions: Mapping[str, object],
    ) -> float:
        """Exact counterfactual probability; enumerates all exogenous worlds."""

        num = 0.0
        den = 0.0
        for exo, p in self._exogenous_worlds():
            actual = self.evaluate_world(exo, interventions=None)
            if all(actual.get(k) == v for k, v in evidence.items()):
                den += p
                cf = self.evaluate_world(exo, interventions=interventions)
                if all(cf.get(k) == v for k, v in query_event.items()):
                    num += p
        if den <= _EPS:
            return 0.0
        return num / den

    def counterfactual_probability_monte_carlo(
        self,
        query_event: Mapping[str, object],
        *,
        evidence: Mapping[str, object],
        interventions: Mapping[str, object],
        n_samples: int = 4_000,
        seed: int = 0,
        burn_in_sweeps: int = 4,
        init_max_draws: int | None = None,
    ) -> float:
        """Exact-conditional Gibbs abduction for counterfactual probability.

        Each Gibbs step resamples one exogenous variable ``U_i`` from its
        *exact* conditional posterior::

            P(U_i = v | U_{-i}, E) ∝ P(U_i = v) · 1[evaluate(U_i = v, U_{-i}) ⊨ E]

        Because every endogenous equation is deterministic, the indicator is
        computed by enumerating values of ``U_i`` (one ``evaluate_world`` call
        per value) instead of rejecting samples drawn from the prior. The chain
        therefore lives entirely on the evidence support: every retained
        sample is already evidence-consistent, so we never throw a draw away
        and the estimator scales gracefully to evidence with prior probability
        as low as the initialiser can locate.

        Initialisation finds a seed assignment in ``U`` that satisfies ``E``
        via two strategies. Strategy 1 is prior rejection (fast for
        ``P(E) >= ~1e-3``). Strategy 2 is a WalkSAT-style stochastic local
        search over ``U`` that reduces evidence violations greedily with
        random kicks; this exploits the structural equations to *propose*
        evidence-consistent worlds when no prior draw lands on the support.

        After ``burn_in_sweeps`` full sweeps over the exogenous variables we
        take ``n_samples`` random-scan Gibbs samples (one ``U_i`` resampled
        per sample) and apply ``interventions`` to each for the
        counterfactual readout. The conditional support shrinks to the
        current value when no other value preserves evidence, so a stuck
        chain is a no-op rather than a crash.
        """

        rng = random.Random(int(seed))
        evidence_d = dict(evidence)
        query_event_d = dict(query_event)
        exo_names = list(self.exogenous)

        if not exo_names:
            actual = self.evaluate_world({}, interventions=None)
            if not all(actual.get(k) == v for k, v in evidence_d.items()):
                return 0.0
            cf = self.evaluate_world({}, interventions=interventions)
            return 1.0 if all(cf.get(k) == v for k, v in query_event_d.items()) else 0.0

        state = self._initialize_evidence_consistent_state(rng, evidence_d, init_max_draws)
        if state is None:
            logger.warning(
                "counterfactual_probability_monte_carlo: failed to seed the Gibbs chain on evidence; "
                "evidence has effectively zero structural support, returning 0.0",
            )
            return 0.0

        burn = max(0, int(burn_in_sweeps))
        for _ in range(burn):
            for name in exo_names:
                state = self._gibbs_resample(rng, name, state, evidence_d)

        n = max(1, int(n_samples))
        num = 0
        den = 0
        for _ in range(n):
            name = rng.choice(exo_names)
            state = self._gibbs_resample(rng, name, state, evidence_d)
            cf = self.evaluate_world(state, interventions=interventions)
            den += 1
            if all(cf.get(k) == v for k, v in query_event_d.items()):
                num += 1

        if den <= 0:
            return 0.0
        return num / den

    def _gibbs_resample(
        self,
        rng: random.Random,
        name: str,
        state: Mapping[str, object],
        evidence_d: Mapping[str, object],
    ) -> dict[str, object]:
        """Sample ``U_name`` from ``P(U_name | U_{-name}, E)`` by domain enumeration."""

        prior = self.exogenous[name]
        candidates: list[object] = []
        weights: list[float] = []
        trial = dict(state)
        for value, prior_p in prior.items():
            p = float(prior_p)
            if p <= 0.0:
                continue
            trial[name] = value
            actual = self.evaluate_world(trial, interventions=None)
            if not all(actual.get(k) == v for k, v in evidence_d.items()):
                continue
            candidates.append(value)
            weights.append(p)
        new_state = dict(state)
        if candidates:
            new_state[name] = rng.choices(candidates, weights=weights, k=1)[0]
        return new_state

    def _evidence_violations(
        self,
        state: Mapping[str, object],
        evidence_d: Mapping[str, object],
    ) -> int:
        actual = self.evaluate_world(dict(state), interventions=None)
        return sum(1 for k, v in evidence_d.items() if actual.get(k) != v)

    def _initialize_evidence_consistent_state(
        self,
        rng: random.Random,
        evidence_d: Mapping[str, object],
        init_max_draws: int | None,
    ) -> dict[str, object] | None:
        """Locate any ``U`` satisfying evidence; rejection first, then stochastic local search."""

        exo_names = list(self.exogenous)
        domain_total = sum(len(self.exogenous[n]) for n in exo_names) or 1
        cap = int(init_max_draws) if init_max_draws is not None else max(2048, 32 * domain_total)
        rejection_budget = max(64, cap // 4)

        for _ in range(rejection_budget):
            candidate = self._sample_exogenous_world(rng)
            if self._evidence_violations(candidate, evidence_d) == 0:
                return candidate

        sls_budget = max(0, cap - rejection_budget)
        if sls_budget == 0 or not exo_names:
            return None
        restart_every = max(8, sls_budget // 16)
        noise = 0.3
        state = self._sample_exogenous_world(rng)
        violations = self._evidence_violations(state, evidence_d)

        for step in range(sls_budget):
            if violations == 0:
                return dict(state)
            if step > 0 and step % restart_every == 0:
                candidate = self._sample_exogenous_world(rng)
                cand_violations = self._evidence_violations(candidate, evidence_d)
                if cand_violations < violations:
                    state, violations = candidate, cand_violations
            name = rng.choice(exo_names)
            prior = self.exogenous[name]
            domain_values = [(v, float(p)) for v, p in prior.items() if float(p) > 0.0]
            if not domain_values:
                continue
            if rng.random() < noise:
                value = rng.choices([v for v, _ in domain_values], weights=[p for _, p in domain_values], k=1)[0]
                trial = dict(state)
                trial[name] = value
                trial_violations = self._evidence_violations(trial, evidence_d)
            else:
                best_value = state[name]
                best_violations = violations
                for value, _ in domain_values:
                    trial = dict(state)
                    trial[name] = value
                    v_count = self._evidence_violations(trial, evidence_d)
                    if v_count < best_violations:
                        best_violations = v_count
                        best_value = value
                trial = dict(state)
                trial[name] = best_value
                trial_violations = best_violations
            state, violations = trial, trial_violations

        return dict(state) if violations == 0 else None

    def descendants(self, node: str, *, parents: Mapping[str, Sequence[str]] | None = None) -> set[str]:
        parents = {k: list(v) for k, v in (parents or self.graph_parents(include_exogenous=True)).items()}
        children: dict[str, list[str]] = {n: [] for n in parents}
        for child, ps in parents.items():
            for p in ps:
                children.setdefault(p, []).append(child)
        out: set[str] = set()
        stack = list(children.get(node, []))
        while stack:
            cur = stack.pop()
            if cur in out:
                continue
            out.add(cur)
            stack.extend(children.get(cur, []))
        return out

    def _neighbors(self, parents: Mapping[str, Sequence[str]]) -> dict[str, set[str]]:
        nodes = set(parents)
        for ps in parents.values():
            nodes.update(ps)
        nb = {n: set() for n in nodes}
        for child, ps in parents.items():
            for p in ps:
                nb[p].add(child)
                nb[child].add(p)
        return nb

    def _all_simple_paths(self, start: str, end: str, parents: Mapping[str, Sequence[str]], max_len: int | None = None) -> list[list[str]]:
        nb = self._neighbors(parents)
        max_len = max_len or len(nb) + 1
        paths: list[list[str]] = []
        stack = [(start, [start])]
        while stack:
            cur, path = stack.pop()
            if len(path) > max_len:
                continue
            if cur == end:
                paths.append(path)
                continue
            for nxt in nb.get(cur, ()):
                if nxt not in path:
                    stack.append((nxt, path + [nxt]))
        return paths

    @staticmethod
    def _has_arrow(parents: Mapping[str, Sequence[str]], src: str, dst: str) -> bool:
        return src in parents.get(dst, ())

    def _path_active(self, path: Sequence[str], conditioned: set[str], parents: Mapping[str, Sequence[str]]) -> bool:
        conditioned_or_desc = set(conditioned)
        for z in conditioned:
            conditioned_or_desc.update(self.descendants(z, parents=parents))
        for i in range(1, len(path) - 1):
            a, b, c = path[i - 1], path[i], path[i + 1]
            collider = self._has_arrow(parents, a, b) and self._has_arrow(parents, c, b)
            if collider:
                if b not in conditioned_or_desc:
                    return False
            else:
                if b in conditioned:
                    return False
        return True

    def d_separated(
        self,
        x: str | Iterable[str],
        y: str | Iterable[str],
        z: Iterable[str] = (),
        *,
        parents: Mapping[str, Sequence[str]] | None = None,
    ) -> bool:
        parents = {k: list(v) for k, v in (parents or self.graph_parents(include_exogenous=True)).items()}
        xs = {x} if isinstance(x, str) else set(x)
        ys = {y} if isinstance(y, str) else set(y)
        conditioned = set(z)
        for a in xs:
            for b in ys:
                for path in self._all_simple_paths(a, b, parents):
                    if len(path) > 1 and self._path_active(path, conditioned, parents):
                        return False
        return True

    def _remove_outgoing(self, nodes: Iterable[str], parents: Mapping[str, Sequence[str]] | None = None) -> dict[str, list[str]]:
        nodes = set(nodes)
        parents = {k: list(v) for k, v in (parents or self.graph_parents(include_exogenous=True)).items()}
        for child, ps in list(parents.items()):
            parents[child] = [p for p in ps if p not in nodes]
        return parents

    def _directed_paths(self, start: str, end: str, parents: Mapping[str, Sequence[str]] | None = None) -> list[list[str]]:
        parents = {k: list(v) for k, v in (parents or self.graph_parents(include_exogenous=True)).items()}
        children: dict[str, list[str]] = {n: [] for n in parents}
        for child, ps in parents.items():
            for p in ps:
                children.setdefault(p, []).append(child)
        paths: list[list[str]] = []
        stack = [(start, [start])]
        while stack:
            cur, path = stack.pop()
            if cur == end:
                paths.append(path)
                continue
            for nxt in children.get(cur, []):
                if nxt not in path:
                    stack.append((nxt, path + [nxt]))
        return paths

    def backdoor_sets(self, treatment: str, outcome: str, *, observed_only: bool = True, max_size: int | None = None) -> list[tuple[str, ...]]:
        observed = set(self.observed_names if observed_only else self.domains)
        forbidden = {treatment, outcome} | self.descendants(treatment)
        candidates = sorted(observed - forbidden)
        max_size = len(candidates) if max_size is None else min(max_size, len(candidates))
        parents_cut = self._remove_outgoing([treatment])
        good: list[tuple[str, ...]] = []
        for r in range(max_size + 1):
            for combo in itertools.combinations(candidates, r):
                if self.d_separated(treatment, outcome, combo, parents=parents_cut):
                    good.append(combo)
            if good:
                return good
        return good

    def backdoor_adjustment(self, *, treatment: str, treatment_value, outcome: str, outcome_value, adjustment_set: Sequence[str]) -> float:
        zvars = tuple(adjustment_set)
        if not zvars:
            return self.probability({outcome: outcome_value}, given={treatment: treatment_value})
        total = 0.0
        for zvals in itertools.product(*(self.domains[z] for z in zvars)):
            z = dict(zip(zvars, zvals))
            p_z = self.probability(z)
            p_y = self.probability({outcome: outcome_value}, given={treatment: treatment_value, **z})
            total += p_y * p_z
        return total

    def frontdoor_sets(self, treatment: str, outcome: str, *, observed_only: bool = True, max_size: int | None = None) -> list[tuple[str, ...]]:
        observed = set(self.observed_names if observed_only else self.domains)
        candidates = sorted(observed - {treatment, outcome})
        max_size = len(candidates) if max_size is None else min(max_size, len(candidates))
        parents_full = self.graph_parents(include_exogenous=True)
        directed = self._directed_paths(treatment, outcome, parents_full)
        good: list[tuple[str, ...]] = []
        for r in range(1, max_size + 1):
            for combo in itertools.combinations(candidates, r):
                zset = set(combo)
                # 1) Z intercepts every directed path from X to Y.
                if directed and not all(any(n in zset for n in path[1:-1]) for path in directed):
                    continue
                # 2) No unblocked back-door path from X to Z.
                g_no_x_out = self._remove_outgoing([treatment], parents_full)
                if not self.d_separated(treatment, zset, (), parents=g_no_x_out):
                    continue
                # 3) All back-door paths from Z to Y are blocked by X.
                g_no_z_out = self._remove_outgoing(zset, parents_full)
                if not self.d_separated(zset, outcome, {treatment}, parents=g_no_z_out):
                    continue
                good.append(combo)
            if good:
                return good
        return good

    def frontdoor_adjustment(self, *, treatment: str, treatment_value, outcome: str, outcome_value, mediator_set: Sequence[str]) -> float:
        zvars = tuple(mediator_set)
        if not zvars:
            raise ValueError("front-door adjustment requires at least one mediator")
        total = 0.0
        for zvals in itertools.product(*(self.domains[z] for z in zvars)):
            z = dict(zip(zvars, zvals))
            p_z_given_x = self.probability(z, given={treatment: treatment_value})
            inner = 0.0
            for xprime in self.domains[treatment]:
                p_y = self.probability({outcome: outcome_value}, given={treatment: xprime, **z})
                p_xprime = self.probability({treatment: xprime})
                inner += p_y * p_xprime
            total += p_z_given_x * inner
        return total


def build_simpson_scm() -> FiniteSCM:
    scm = FiniteSCM(domains={})
    scm.labels.update(
        {
            "T": "treatment",
            "Y": "outcome",
            "positive_effect": "helps",
            "negative_effect": "hurts",
        }
    )
    scm.add_exogenous("U_S", [0, 1], {0: 0.5, 1: 0.5})
    scm.add_exogenous("U_T", range(100))
    scm.add_exogenous("U_Y", range(100))

    scm.add_endogenous("S", [0, 1], ["U_S"], lambda v: v["U_S"])

    def t_fn(v):
        # High-baseline stratum S=1 is rarely treated; low-baseline stratum S=0 is usually treated.
        threshold = 10 if v["S"] == 1 else 90
        return 1 if v["U_T"] < threshold else 0

    def y_fn(v):
        # Treatment helps in both strata, but confounding makes naive association look negative.
        table = {
            (1, 1): 80,  # S=1,T=1
            (1, 0): 70,
            (0, 1): 30,
            (0, 0): 20,
        }
        return 1 if v["U_Y"] < table[(v["S"], v["T"])] else 0

    scm.add_endogenous("T", [0, 1], ["S", "U_T"], t_fn)
    scm.add_endogenous("Y", [0, 1], ["S", "T", "U_Y"], y_fn)
    logger.debug("build_simpson_scm: enumerate_worlds=%d vars=%s", scm.exogenous_world_volume, scm.order)
    return scm


def build_frontdoor_scm() -> FiniteSCM:
    scm = FiniteSCM(domains={})
    scm.add_exogenous("U", [0, 1], {0: 0.5, 1: 0.5})       # unobserved confounder of X and Y
    scm.add_exogenous("U_X", range(10))
    scm.add_exogenous("U_M", range(10))
    scm.add_exogenous("U_Y", range(10))

    def x_fn(v):
        threshold = 8 if v["U"] == 1 else 2
        return 1 if v["U_X"] < threshold else 0

    def m_fn(v):
        threshold = 8 if v["X"] == 1 else 2
        return 1 if v["U_M"] < threshold else 0

    def y_fn(v):
        # Y depends on mediator M and hidden U. There is no direct X -> Y arrow.
        threshold = {
            (0, 0): 1,
            (0, 1): 5,
            (1, 0): 5,
            (1, 1): 8,
        }[(v["M"], v["U"])]
        return 1 if v["U_Y"] < threshold else 0

    scm.add_endogenous("X", [0, 1], ["U", "U_X"], x_fn)
    scm.add_endogenous("M", [0, 1], ["X", "U_M"], m_fn)
    scm.add_endogenous("Y", [0, 1], ["M", "U", "U_Y"], y_fn)
    logger.debug("build_frontdoor_scm: enumerate_worlds=%d vars=%s", scm.exogenous_world_volume, scm.order)
    return scm
