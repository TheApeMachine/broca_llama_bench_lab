from __future__ import annotations

import itertools
import logging
import random
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

from .dag import CausalDAG
from .equation import EndogenousEquation


_EPS = 1e-12

# Initialization budgets for evidence-consistent exogenous state search (rejection + local search).
_INIT_CAP_DOMAIN_MULTIPLIER = 32  # Extra headroom on top of total_mass * exo_n so wide domains get enough tries.
_INIT_REJECTION_EXO_DIVISOR_FALLBACK = 4  # Lower bound for dividing cap by exo_n when carving out the rejection slice.
_INIT_RESTART_SLS_DIVISOR_BASE = 16  # WalkSAT restart cadence scales as sls_budget / max(this, exo_n * scale).
_INIT_RESTART_EXO_SCALE = 2  # Per-exogenous factor in restart denominator so more roots restart slightly more often.

logger = logging.getLogger(__name__)


@dataclass
class FiniteSCM:
    """Finite structural causal model with exact enumeration over exogenous worlds.

    Exogenous variables are independent roots. Endogenous variables are deterministic
    functions of parents and exogenous noise. Interventions replace structural equations;
    counterfactuals follow abduction, action, prediction (exact or Gibbs-sampled).
    """

    domains: dict[str, tuple]
    exogenous: dict[str, dict[object, float]] = field(default_factory=dict)
    equations: dict[str, EndogenousEquation] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    labels: dict[str, str] = field(default_factory=dict)

    @classmethod
    def simpson_paradox_demo(cls) -> FiniteSCM:
        scm = cls(domains={})
        scm.labels.update(
            {
                "T": "treatment",
                "Y": "outcome",
                "positive_effect": "helps",
                "negative_effect": "hurts",
            }
        )
        scm.add_exogenous("U_S", [0, 1], {0: 0.5, 1: 0.5})
        scm.add_exogenous_uniform("U_T", range(100))
        scm.add_exogenous_uniform("U_Y", range(100))
        scm.add_endogenous("S", [0, 1], ["U_S"], lambda v: v["U_S"])

        def t_fn(v: dict) -> int:
            threshold = 10 if v["S"] == 1 else 90
            return 1 if v["U_T"] < threshold else 0

        def y_fn(v: dict) -> int:
            table = {
                (1, 1): 80,
                (1, 0): 70,
                (0, 1): 30,
                (0, 0): 20,
            }

            return 1 if v["U_Y"] < table[(v["S"], v["T"])] else 0

        scm.add_endogenous("T", [0, 1], ["S", "U_T"], t_fn)
        scm.add_endogenous("Y", [0, 1], ["S", "T", "U_Y"], y_fn)

        logger.debug(
            "FiniteSCM.simpson_paradox_demo: enumerate_worlds=%d vars=%s",
            scm.exogenous_world_volume,
            scm.order,
        )

        return scm

    @classmethod
    def frontdoor_demo(cls) -> FiniteSCM:
        scm = cls(domains={})
        scm.add_exogenous("U", [0, 1], {0: 0.5, 1: 0.5})
        scm.add_exogenous_uniform("U_X", range(10))
        scm.add_exogenous_uniform("U_M", range(10))
        scm.add_exogenous_uniform("U_Y", range(10))

        def x_fn(v: dict) -> int:
            threshold = 8 if v["U"] == 1 else 2
            return 1 if v["U_X"] < threshold else 0

        def m_fn(v: dict) -> int:
            threshold = 8 if v["X"] == 1 else 2
            return 1 if v["U_M"] < threshold else 0

        def y_fn(v: dict) -> int:
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

        logger.debug(
            "FiniteSCM.frontdoor_demo: enumerate_worlds=%d vars=%s",
            scm.exogenous_world_volume,
            scm.order,
        )

        return scm

    def add_exogenous_uniform(self, name: str, domain: Sequence[object]) -> None:
        dom = tuple(domain)

        if len(dom) == 0:
            raise ValueError(f"FiniteSCM.add_exogenous_uniform: empty domain for {name!r}")

        if len(set(dom)) != len(dom):
            raise ValueError(f"FiniteSCM.add_exogenous_uniform: domain for {name!r} contains duplicates")

        dom_unique = tuple(dict.fromkeys(dom))
        probs = {x: 1.0 / len(dom_unique) for x in dom_unique}
        self._install_exogenous(name, dom_unique, probs)

    def add_exogenous(self, name: str, domain: Sequence[object], probs: Mapping[object, float]) -> None:
        dom = tuple(domain)

        if len(set(dom)) != len(dom):
            raise ValueError(f"FiniteSCM.add_exogenous: domain for {name!r} contains duplicates")

        dom_set = set(dom)
        keys = set(probs.keys())

        if keys != dom_set:
            raise ValueError(f"FiniteSCM.add_exogenous: probs keys must equal domain for {name!r}")

        total = sum(float(probs[x]) for x in dom)

        if total <= 0.0:
            raise ValueError(f"FiniteSCM.add_exogenous: probabilities for {name!r} sum to zero")

        normalized = {x: float(probs[x]) / total for x in dom}
        self._install_exogenous(name, dom, normalized)

    def _install_exogenous(self, name: str, dom: tuple, probs: dict[object, float]) -> None:
        self.domains[name] = dom
        self.exogenous[name] = probs

    def add_endogenous(
        self, 
        name: str, 
        domain: Sequence, 
        parents: Sequence[str], 
        fn: Callable[[dict], object]
    ) -> None:
        missing = [str(p) for p in parents if str(p) not in self.domains]

        if missing:
            raise ValueError(
                f"FiniteSCM.add_endogenous: unknown parent variable(s) {missing} for endogenous {name!r}; "
                "define each parent with add_exogenous / add_endogenous before adding this variable."
            )
        
        self.domains[name] = tuple(domain)
        self.equations[name] = EndogenousEquation(name, tuple(parents), fn)
        self.order.append(name)

    def update_endogenous(
        self,
        name: str,
        *,
        fn: Callable[[dict], object],
        domain: Sequence | None = None,
        parents: Sequence[str] | None = None,
    ) -> None:
        if name not in self.equations:
            raise ValueError(
                f"FiniteSCM.update_endogenous: unknown endogenous variable {name!r}"
            )

        cur = self.equations[name]
        new_parents = tuple(parents) if parents is not None else cur.parents

        if domain is not None:
            self.domains[name] = tuple(domain)

        self.equations[name] = EndogenousEquation(name=name, parents=new_parents, fn=fn)

    @property
    def endogenous_names(self) -> tuple[str, ...]:
        return tuple(self.order)

    @property
    def observed_names(self) -> tuple[str, ...]:
        return tuple(self.order)

    def graph_parents_observed(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}

        for name in self.order:
            parents = [p for p in self.equations[name].parents if p not in self.exogenous]
            out[name] = parents

        return out

    def graph_parents_full(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}

        for u in self.exogenous:
            out[u] = []

        for name in self.order:
            out[name] = list(self.equations[name].parents)

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

    @staticmethod
    def _valuation_matches(
        vals: Mapping[str, object], assignment: Mapping[str, object]
    ) -> bool:
        return all(vals.get(k) == v for k, v in assignment.items())

    def evaluate_world(
        self, exo: Mapping[str, object], interventions: Mapping[str, object]
    ) -> dict[str, object]:
        values = dict(exo)

        for name in self.order:
            if name in interventions:
                values[name] = interventions[name]

            else:
                values[name] = self.equations[name].fn(values)

            if values[name] not in self.domains[name]:
                raise ValueError(
                    f"{name} returned value {values[name]!r}, outside domain {self.domains[name]!r}"
                )

        return values

    def probability(
        self,
        event: Mapping[str, object],
        *,
        given: Mapping[str, object],
        interventions: Mapping[str, object],
    ) -> float:
        given_d = dict(given)
        num = 0.0
        den = 0.0

        for exo, p in self._exogenous_worlds():
            vals = self.evaluate_world(exo, interventions)

            if self._valuation_matches(vals, given_d):
                den += p

                if self._valuation_matches(vals, event):
                    num += p

        if den <= _EPS:
            raise ValueError(
                "FiniteSCM.probability: conditioning event has zero probability under this model "
                f"(given={given_d}, interventions={dict(interventions)})",
            )

        prob = num / den

        logger.debug(
            "FiniteSCM.probability: p=%.8f event=%s given=%s do=%s worlds=%d",
            prob,
            dict(event),
            given_d,
            dict(interventions),
            self.exogenous_world_volume,
        )

        return prob

    @property
    def exogenous_world_volume(self) -> int:
        vol = 1

        for name in self.exogenous:
            vol *= len(self.exogenous[name])

        return vol

    def probability_monte_carlo(
        self,
        event: Mapping[str, object],
        *,
        given: Mapping[str, object],
        interventions: Mapping[str, object],
        n_samples: int,
        seed: int,
    ) -> float:
        rng = random.Random(int(seed))
        given_d = dict(given)
        num = 0
        den = 0

        if n_samples <= 0:
            raise ValueError("FiniteSCM.probability_monte_carlo: n_samples must be positive")

        for _ in range(int(n_samples)):
            world = self._sample_exogenous_world(rng)
            vals = self.evaluate_world(world, interventions)

            if self._valuation_matches(vals, given_d):
                den += 1

                if self._valuation_matches(vals, event):
                    num += 1

        if den <= 0:
            raise ValueError(
                "FiniteSCM.probability_monte_carlo: conditioning event never occurred in sampled worlds "
                f"(given={given_d}, interventions={dict(interventions)})",
            )

        return num / den

    def distribution(
        self,
        variables: Sequence[str],
        *,
        given: Mapping[str, object],
        interventions: Mapping[str, object],
    ) -> dict[tuple, float]:
        out: dict[tuple, float] = {}
        den = 0.0
        given_d = dict(given)

        for exo, p in self._exogenous_worlds():
            vals = self.evaluate_world(exo, interventions)

            if self._valuation_matches(vals, given_d):
                key = tuple(vals[v] for v in variables)
                out[key] = out.get(key, 0.0) + p
                den += p

        if den <= _EPS:
            raise ValueError(
                "FiniteSCM.distribution: conditioning event has zero probability "
                f"(given={given_d}, interventions={dict(interventions)})",
            )

        return {k: v / den for k, v in out.items()}

    def counterfactual_probability(
        self,
        query_event: Mapping[str, object],
        *,
        evidence: Mapping[str, object],
        interventions: Mapping[str, object],
        n_samples: int,
        seed: int,
        gibbs_thin: int = 1,
    ) -> float:
        return self.counterfactual_probability_monte_carlo(
            query_event,
            evidence=evidence,
            interventions=interventions,
            n_samples=int(n_samples),
            seed=int(seed),
            gibbs_thin=int(gibbs_thin),
        )

    def counterfactual_probability_exact(
        self,
        query_event: Mapping[str, object],
        *,
        evidence: Mapping[str, object],
        interventions: Mapping[str, object],
    ) -> float:
        evidence_d = dict(evidence)
        query_event_d = dict(query_event)
        num = 0.0
        den = 0.0

        for exo, p in self._exogenous_worlds():
            actual = self.evaluate_world(exo, {})

            if self._valuation_matches(actual, evidence_d):
                den += p
                cf = self.evaluate_world(exo, interventions)

                if self._valuation_matches(cf, query_event_d):
                    num += p

        if den <= _EPS:
            raise ValueError(
                "FiniteSCM.counterfactual_probability_exact: evidence has zero structural probability mass "
                f"(evidence={evidence_d})",
            )

        return num / den

    def counterfactual_probability_monte_carlo(
        self,
        query_event: Mapping[str, object],
        *,
        evidence: Mapping[str, object],
        interventions: Mapping[str, object],
        n_samples: int,
        seed: int,
        gibbs_thin: int = 1,
    ) -> float:
        rng = random.Random(int(seed))
        evidence_d = dict(evidence)
        query_event_d = dict(query_event)
        exo_names = list(self.exogenous)

        if n_samples <= 0:
            raise ValueError("FiniteSCM.counterfactual_probability_monte_carlo: n_samples must be positive")

        if gibbs_thin < 1:
            raise ValueError("FiniteSCM.counterfactual_probability_monte_carlo: gibbs_thin must be >= 1")

        if not exo_names:
            actual = self.evaluate_world({}, {})

            if not self._valuation_matches(actual, evidence_d):
                raise ValueError(
                    "FiniteSCM.counterfactual_probability_monte_carlo: evidence contradicts deterministic SCM "
                    f"(evidence={evidence_d})",
                )

            cf = self.evaluate_world({}, interventions)

            return 1.0 if self._valuation_matches(cf, query_event_d) else 0.0

        state = self._initialize_evidence_consistent_state(rng, evidence_d)

        if state is None:
            raise RuntimeError(
                "FiniteSCM.counterfactual_probability_monte_carlo: could not initialise any exogenous assignment "
                "consistent with evidence under this SCM",
            )

        burn_in_sweeps = max(len(exo_names), 1)

        for _ in range(burn_in_sweeps):
            for name in exo_names:
                state = self._gibbs_resample(rng, name, state, evidence_d)

        num = 0
        thin = int(gibbs_thin)

        for _ in range(int(n_samples)):
            for _ in range(thin):
                name = rng.choice(exo_names)
                state = self._gibbs_resample(rng, name, state, evidence_d)
            cf = self.evaluate_world(state, interventions)

            if self._valuation_matches(cf, query_event_d):
                num += 1

        return num / float(n_samples)

    def _gibbs_resample(
        self,
        rng: random.Random,
        name: str,
        state: Mapping[str, object],
        evidence_d: Mapping[str, object],
    ) -> dict[str, object]:
        prior = self.exogenous[name]
        candidates: list[object] = []
        weights: list[float] = []
        trial = dict(state)

        for value, prior_p in prior.items():
            p = float(prior_p)

            if p <= 0.0:
                continue

            trial[name] = value
            actual = self.evaluate_world(trial, {})

            if not self._valuation_matches(actual, evidence_d):
                continue

            candidates.append(value)
            weights.append(p)

        new_state = dict(state)

        if candidates:
            new_state[name] = rng.choices(candidates, weights=weights, k=1)[0]

        return new_state

    def _evidence_violations(
        self, state: Mapping[str, object], evidence_d: Mapping[str, object]
    ) -> int:
        actual = self.evaluate_world(dict(state), {})
        return sum(1 for k, v in evidence_d.items() if actual.get(k) != v)

    def _initialization_budgets(self) -> tuple[int, int, int, float]:
        """Return rejection draws, WalkSAT draws, restart cadence, random-flip probability."""

        exo_names = list(self.exogenous)
        exo_n = len(exo_names)
        domain_total = sum(len(self.exogenous[n]) for n in exo_names) or 1
        total_mass = domain_total * max(exo_n, 1)
        cap = max(total_mass * max(exo_n, 1), domain_total * _INIT_CAP_DOMAIN_MULTIPLIER)
        rejection_budget = max(domain_total, cap // max(exo_n, _INIT_REJECTION_EXO_DIVISOR_FALLBACK))
        sls_budget = max(0, cap - rejection_budget)
        restart_every = max(1, sls_budget // max(_INIT_RESTART_SLS_DIVISOR_BASE, exo_n * _INIT_RESTART_EXO_SCALE))
        noise = 1.0 / (1 + exo_n)

        return rejection_budget, sls_budget, restart_every, noise

    def _initialize_evidence_consistent_state(
        self,
        rng: random.Random,
        evidence_d: Mapping[str, object],
    ) -> dict[str, object] | None:
        exo_names = list(self.exogenous)
        rejection_budget, sls_budget, restart_every, noise = self._initialization_budgets()

        for _ in range(rejection_budget):
            candidate = self._sample_exogenous_world(rng)

            if self._evidence_violations(candidate, evidence_d) == 0:
                return candidate

        if sls_budget == 0 or not exo_names:
            return None

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

    def descendants(self, node: str) -> set[str]:
        return CausalDAG(self.graph_parents_full()).descendants(node)

    def d_separated(
        self,
        x: str | Sequence[str],
        y: str | Sequence[str],
        z: Sequence[str],
        *,
        parents: Mapping[str, Sequence[str]] | None = None,
        max_simple_paths: int | None = None,
    ) -> bool:
        dag = CausalDAG(dict(parents)) if parents is not None else CausalDAG(self.graph_parents_full())

        return dag.d_separated(x, y, z, max_simple_paths=max_simple_paths)

    def backdoor_sets(self, treatment: str, outcome: str) -> list[tuple[str, ...]]:
        observed = set(self.observed_names)
        forbidden = {treatment, outcome} | self.descendants(treatment)
        candidates = sorted(observed - forbidden)
        dag_full = CausalDAG(self.graph_parents_full())
        parents_cut = dag_full.remove_outgoing_from([treatment])
        good: list[tuple[str, ...]] = []

        for r in range(len(candidates) + 1):
            for combo in itertools.combinations(candidates, r):
                if parents_cut.d_separated(treatment, outcome, combo):
                    good.append(combo)

            if good:
                return good

        return good

    def backdoor_adjustment(
        self, 
        *, 
        treatment: str, 
        treatment_value, 
        outcome: str, 
        outcome_value, 
        adjustment_set: Sequence[str]
    ) -> float:
        zvars = tuple(adjustment_set)

        if not zvars:
            return self.probability(
                {outcome: outcome_value},
                given={treatment: treatment_value},
                interventions={},
            )

        total = 0.0

        for zvals in itertools.product(*(self.domains[z] for z in zvars)):
            z = dict(zip(zvars, zvals))
            p_z = self.probability(z, given={}, interventions={})
            p_y = self.probability(
                {outcome: outcome_value},
                given={treatment: treatment_value, **z},
                interventions={},
            )
            total += p_y * p_z

        return total

    def frontdoor_sets(
        self, treatment: str, outcome: str
    ) -> list[tuple[str, ...]]:
        observed = set(self.observed_names)
        candidates = sorted(observed - {treatment, outcome})
        dag_full = CausalDAG(self.graph_parents_full())
        directed = dag_full.directed_paths(treatment, outcome)
        good: list[tuple[str, ...]] = []

        for r in range(1, len(candidates) + 1):
            for combo in itertools.combinations(candidates, r):
                zset = set(combo)

                if directed and not all(any(n in zset for n in path[1:-1]) for path in directed):
                    continue

                g_no_x_out = dag_full.remove_outgoing_from([treatment])

                if not g_no_x_out.d_separated(treatment, zset, ()):
                    continue

                g_no_z_out = dag_full.remove_outgoing_from(zset)

                if not g_no_z_out.d_separated(zset, outcome, {treatment}):
                    continue

                good.append(combo)

            if good:
                return good

        return good

    def frontdoor_adjustment(
        self,
        *,
        treatment: str,
        treatment_value,
        outcome: str,
        outcome_value,
        mediator_set: Sequence[str],
    ) -> float:
        zvars = tuple(mediator_set)

        if not zvars:
            raise ValueError("FiniteSCM.frontdoor_adjustment: mediator_set must be non-empty")

        total = 0.0

        for zvals in itertools.product(*(self.domains[z] for z in zvars)):
            z = dict(zip(zvars, zvals))
            p_z_given_x = self.probability(z, given={treatment: treatment_value}, interventions={})
            inner = 0.0

            for xprime in self.domains[treatment]:
                p_y = self.probability({outcome: outcome_value}, given={treatment: xprime, **z}, interventions={})
                p_xprime = self.probability({treatment: xprime}, given={}, interventions={})
                inner += p_y * p_xprime

            total += p_z_given_x * inner

        return total


build_simpson_scm = FiniteSCM.simpson_paradox_demo
build_frontdoor_scm = FiniteSCM.frontdoor_demo
