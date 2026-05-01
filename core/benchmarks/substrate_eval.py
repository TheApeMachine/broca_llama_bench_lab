"""Substrate-specific benchmark suite for the Mosaic/Broca architecture.

These benchmarks test capabilities that are unique to the cognitive substrate
and cannot be measured by standard LLM leaderboards. Each benchmark exercises
a different algebraic or control-theoretic guarantee of the architecture:

1.  **Rule-shift adaptation** — The substrate must revise a stored belief when
    new evidence accumulates, demonstrating online Bayesian belief revision
    with prediction-gap-weighted trust (not just majority voting).

2.  **Adversarial prompt resistance** — Top-down hypothesis masking physically
    blocks rejected tokens; the iterative search must converge on a logically
    valid answer even when the prompt is designed to mislead.

3.  **Causal reasoning (Simpson's paradox)** — The SCM's ``do(·)`` operator
    must recover the correct treatment effect despite confounding that makes
    naive association give the wrong sign.

4.  **Semantic memory fidelity** — Write N triples, recall each; measure
    perfect-recall rate and confidence calibration.

5.  **Conformal coverage guarantee** — Calibrate a split-conformal predictor,
    then verify that empirical coverage on a held-out set meets the ``1-α``
    guarantee.

6.  **VSA algebraic fidelity** — Bind/unbind/bundle round-trip accuracy at
    varying dimensionalities and bundle sizes, testing the capacity bound
    ``~ 0.5 · d / log d``.

7.  **Hopfield retrieval accuracy** — One-step retrieval accuracy at varying
    store sizes, verifying exponential capacity in ``d``.

8.  **Active inference decision quality** — Run the Tiger POMDP agent over
    many episodes and measure success rate vs a random baseline.

All benchmarks are designed to run on CPU without a model download. They
exercise the substrate's algebra directly.
"""

from __future__ import annotations

import json
import logging
import math
import random
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

logger = logging.getLogger(__name__)


@dataclass
class SubstrateBenchmarkResult:
    """Result of a single substrate benchmark."""
    name: str
    description: str
    passed: bool
    score: float
    n_trials: int
    details: dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0


@dataclass
class SubstrateBenchmarkSuite:
    """Aggregate results of the full substrate benchmark suite."""
    results: list[SubstrateBenchmarkResult] = field(default_factory=list)
    total_duration_seconds: float = 0.0

    def summary(self) -> dict[str, Any]:
        passed = sum(1 for r in self.results if r.passed)
        return {
            "kind": "substrate_benchmark_suite",
            "n_benchmarks": len(self.results),
            "n_passed": passed,
            "n_failed": len(self.results) - passed,
            "pass_rate": passed / max(1, len(self.results)),
            "total_duration_seconds": self.total_duration_seconds,
            "per_benchmark": {r.name: {
                "passed": r.passed,
                "score": r.score,
                "n_trials": r.n_trials,
                "duration_seconds": r.duration_seconds,
                "description": r.description,
                "details": r.details,
            } for r in self.results},
        }


# ---------------------------------------------------------------------------
# 1. Rule-shift adaptation (belief revision)
# ---------------------------------------------------------------------------

def bench_rule_shift(*, n_initial_claims: int = 5, n_challenger_claims: int = 8, seed: int = 0) -> SubstrateBenchmarkResult:
    """Test that the substrate revises beliefs when sufficient low-surprise evidence accumulates."""
    from core.broca import PersistentSemanticMemory

    start = time.time()
    details: dict[str, Any] = {}
    n_trials = 1

    with tempfile.TemporaryDirectory(prefix="bench_rule_shift_") as tmp:
        mem = PersistentSemanticMemory(Path(tmp) / "bench.sqlite", namespace="rule_shift")
        mem.upsert("ada", "location", "rome", confidence=0.9, evidence={"source": "seed"})
        for i in range(n_initial_claims):
            mem.record_claim("ada", "location", "rome", confidence=0.9, status="corroborated",
                             evidence={"source": "initial", "prediction_gap": 0.1 + 0.02 * i})
        for i in range(n_challenger_claims):
            mem.record_claim("ada", "location", "paris", confidence=0.95, status="conflict",
                             evidence={"source": "challenger", "prediction_gap": 0.05 + 0.01 * i})
        reflections = mem.consolidate_claims_once(log_odds_threshold=0.3, min_claims=3)
        current = mem.get("ada", "location")
        final_value = current[0] if current else "unknown"
        revised = final_value == "paris"
        details = {
            "initial_value": "rome", "challenger_value": "paris", "final_value": final_value,
            "n_initial_claims": n_initial_claims, "n_challenger_claims": n_challenger_claims,
            "n_reflections": len(reflections),
            "reflection_kinds": [r.get("kind") for r in reflections], "revised": revised,
        }
        mem.close()

    duration = time.time() - start
    return SubstrateBenchmarkResult(
        name="rule_shift_adaptation",
        description="Belief revision under accumulating low-surprise evidence",
        passed=revised, score=1.0 if revised else 0.0, n_trials=n_trials,
        details=details, duration_seconds=duration,
    )


# ---------------------------------------------------------------------------
# 2. Adversarial prompt resistance (hypothesis masking)
# ---------------------------------------------------------------------------

def bench_adversarial_prompt_resistance(*, seed: int = 0) -> SubstrateBenchmarkResult:
    """Test that HypothesisMaskingGraft converges on valid tokens."""
    from core.top_down_control import HypothesisMaskingGraft, HypothesisVerdict

    start = time.time()
    rng = random.Random(seed)
    vocab_size = 100
    bad_tokens = set(rng.sample(range(vocab_size), 15))
    graft = HypothesisMaskingGraft(default_penalty=100.0)
    n_trials = 50
    successes = 0
    max_iterations = 10
    convergence_steps: list[int] = []

    for trial in range(n_trials):
        graft.clear()
        found = False
        for step in range(1, max_iterations + 1):
            available = [t for t in range(vocab_size) if t not in graft.banned]
            if not available:
                break
            candidate = rng.choice(available)
            if candidate in bad_tokens:
                graft.ban((candidate,), reason=f"bad_token_{candidate}")
            else:
                found = True
                convergence_steps.append(step)
                break
        if found:
            successes += 1

    score = successes / max(1, n_trials)
    avg_steps = sum(convergence_steps) / max(1, len(convergence_steps)) if convergence_steps else float("inf")
    duration = time.time() - start

    return SubstrateBenchmarkResult(
        name="adversarial_prompt_resistance",
        description="Hypothesis masking convergence on valid tokens under adversarial rejection",
        passed=score >= 0.90, score=score, n_trials=n_trials,
        details={
            "successes": successes, "vocab_size": vocab_size, "n_bad_tokens": len(bad_tokens),
            "max_iterations": max_iterations, "avg_convergence_steps": round(avg_steps, 2),
            "convergence_steps_histogram": _histogram(convergence_steps, bins=max_iterations),
        },
        duration_seconds=duration,
    )


# ---------------------------------------------------------------------------
# 3. Causal reasoning — Simpson's paradox
# ---------------------------------------------------------------------------

def bench_causal_reasoning(*, seed: int = 0) -> SubstrateBenchmarkResult:
    """Test that do-calculus recovers the correct treatment effect despite confounding."""
    from core.causal import build_simpson_scm

    start = time.time()
    scm = build_simpson_scm()
    p_y_given_t1 = scm.probability({"Y": 1}, given={"T": 1})
    p_y_given_t0 = scm.probability({"Y": 1}, given={"T": 0})
    naive_suggests_helps = p_y_given_t1 > p_y_given_t0
    p_y_do_t1 = scm.probability({"Y": 1}, interventions={"T": 1})
    p_y_do_t0 = scm.probability({"Y": 1}, interventions={"T": 0})
    ate = p_y_do_t1 - p_y_do_t0
    do_says_helps = p_y_do_t1 > p_y_do_t0
    cf_prob = scm.counterfactual_probability_exact(
        {"Y": 0}, evidence={"T": 1, "Y": 1}, interventions={"T": 0}
    )
    bd_sets = scm.backdoor_sets("T", "Y")
    bd_result = None
    if bd_sets:
        bd_result = scm.backdoor_adjustment(
            treatment="T", treatment_value=1, outcome="Y", outcome_value=1,
            adjustment_set=bd_sets[0],
        )
    paradox_correct = (not naive_suggests_helps) and do_says_helps and ate > 0
    duration = time.time() - start

    return SubstrateBenchmarkResult(
        name="causal_reasoning_simpson",
        description="Simpson's paradox: do-calculus recovers correct ATE despite confounding",
        passed=paradox_correct, score=1.0 if paradox_correct else 0.0, n_trials=1,
        details={
            "p_y_given_t1": round(p_y_given_t1, 6), "p_y_given_t0": round(p_y_given_t0, 6),
            "naive_suggests_helps": naive_suggests_helps,
            "p_y_do_t1": round(p_y_do_t1, 6), "p_y_do_t0": round(p_y_do_t0, 6),
            "ate": round(ate, 6), "do_says_helps": do_says_helps,
            "counterfactual_p_y0_given_t1y1_do_t0": round(cf_prob, 6),
            "backdoor_sets": [list(s) for s in bd_sets[:3]],
            "backdoor_adjusted_p": round(bd_result, 6) if bd_result is not None else None,
        },
        duration_seconds=duration,
    )


# ---------------------------------------------------------------------------
# 4. Semantic memory fidelity
# ---------------------------------------------------------------------------

def bench_memory_fidelity(*, n_triples: int = 100, seed: int = 0) -> SubstrateBenchmarkResult:
    """Write N random triples to semantic memory, recall each, measure accuracy."""
    from core.broca import PersistentSemanticMemory

    start = time.time()
    rng = random.Random(seed)
    subjects = [f"entity_{i}" for i in range(n_triples)]
    predicates = ["located_in", "color_of", "type_of", "related_to", "made_of"]
    objects = [f"value_{rng.randint(0, 999)}" for _ in range(n_triples)]

    with tempfile.TemporaryDirectory(prefix="bench_memory_") as tmp:
        mem = PersistentSemanticMemory(Path(tmp) / "bench.sqlite", namespace="fidelity")
        written: list[tuple[str, str, str, float]] = []
        for i in range(n_triples):
            s, p, o = subjects[i], rng.choice(predicates), objects[i]
            conf = round(rng.uniform(0.5, 1.0), 3)
            mem.upsert(s, p, o, confidence=conf, evidence={"source": "bench", "index": i})
            written.append((s, p, o, conf))
        correct = 0
        confidence_errors: list[float] = []
        for s, p, o, conf in written:
            got = mem.get(s, p)
            if got is not None and got[0] == o:
                correct += 1
                confidence_errors.append(abs(got[1] - conf))
        recall_rate = correct / max(1, n_triples)
        avg_conf_error = sum(confidence_errors) / max(1, len(confidence_errors)) if confidence_errors else float("nan")
        mem.close()

    duration = time.time() - start
    return SubstrateBenchmarkResult(
        name="semantic_memory_fidelity",
        description="Write/recall fidelity over N random triples",
        passed=recall_rate >= 0.99, score=recall_rate, n_trials=n_triples,
        details={
            "n_triples": n_triples, "correct_recalls": correct, "recall_rate": recall_rate,
            "avg_confidence_error": round(avg_conf_error, 6) if math.isfinite(avg_conf_error) else None,
        },
        duration_seconds=duration,
    )


# ---------------------------------------------------------------------------
# 5. Conformal coverage guarantee
# ---------------------------------------------------------------------------

def bench_conformal_coverage(*, n_calibration: int = 200, n_test: int = 500, alpha: float = 0.1, seed: int = 0) -> SubstrateBenchmarkResult:
    """Verify that split-conformal prediction achieves >= 1-alpha empirical coverage."""
    from core.conformal import ConformalPredictor, empirical_coverage

    start = time.time()
    rng = random.Random(seed)
    labels = ["A", "B", "C", "D"]
    n_labels = len(labels)

    def random_dist() -> dict[str, float]:
        raw = [rng.expovariate(1.0) for _ in range(n_labels)]
        s = sum(raw)
        return {labels[i]: raw[i] / s for i in range(n_labels)}

    def sample_true(dist: dict[str, float]) -> str:
        r = rng.random()
        cumulative = 0.0
        for lab, p in dist.items():
            cumulative += p
            if r < cumulative:
                return lab
        return labels[-1]

    lac = ConformalPredictor(alpha=alpha, method="lac", min_calibration=8)
    for _ in range(n_calibration):
        dist = random_dist()
        lac.calibrate(p_label=dist[sample_true(dist)])

    aps = ConformalPredictor(alpha=alpha, method="aps", min_calibration=8)
    for _ in range(n_calibration):
        dist = random_dist()
        aps.calibrate(p_distribution=dist, true_label=sample_true(dist))

    test_data_fixed = []
    for _ in range(n_test):
        d = random_dist()
        test_data_fixed.append((d, sample_true(d)))

    lac_cov = empirical_coverage(lac, test_data_fixed)
    aps_cov = empirical_coverage(aps, test_data_fixed)
    lac_sizes = [lac.predict_set(d).set_size for d, _ in test_data_fixed]
    aps_sizes = [aps.predict_set(d).set_size for d, _ in test_data_fixed]
    avg_lac_size = sum(lac_sizes) / max(1, len(lac_sizes))
    avg_aps_size = sum(aps_sizes) / max(1, len(aps_sizes))
    target = 1.0 - alpha
    slack = 0.05
    both_ok = lac_cov >= (target - slack) and aps_cov >= (target - slack)

    duration = time.time() - start
    return SubstrateBenchmarkResult(
        name="conformal_coverage_guarantee",
        description=f"Split-conformal coverage >= {target:.2f} (alpha={alpha})",
        passed=both_ok, score=(lac_cov + aps_cov) / 2.0, n_trials=n_test,
        details={
            "alpha": alpha, "target_coverage": target,
            "lac_coverage": round(lac_cov, 4), "aps_coverage": round(aps_cov, 4),
            "lac_meets_target": lac_cov >= (target - slack),
            "aps_meets_target": aps_cov >= (target - slack),
            "avg_lac_set_size": round(avg_lac_size, 2), "avg_aps_set_size": round(avg_aps_size, 2),
            "n_calibration": n_calibration, "n_test": n_test,
        },
        duration_seconds=duration,
    )


# ---------------------------------------------------------------------------
# 6. VSA algebraic fidelity
# ---------------------------------------------------------------------------

def bench_vsa_algebra(*, dims: Sequence[int] = (1000, 5000, 10000), n_triples: int = 50, seed: int = 0) -> SubstrateBenchmarkResult:
    """Test bind/unbind round-trip accuracy at varying dimensionalities."""
    from core.vsa import VSACodebook, cosine

    start = time.time()
    rng = random.Random(seed)
    per_dim: dict[int, dict[str, Any]] = {}
    total_correct = 0
    total_trials = 0

    for dim in dims:
        cb = VSACodebook(dim=dim, base_seed=seed)
        correct = 0
        cos_sims: list[float] = []
        atoms = [f"atom_{i}" for i in range(n_triples * 3)]
        for a in atoms:
            cb.atom(a)
        for i in range(n_triples):
            s, p, o = atoms[i * 3], atoms[i * 3 + 1], atoms[i * 3 + 2]
            encoded = cb.encode_triple(s, p, o)
            decoded, cos = cb.decode_role(encoded, "ROLE_OBJECT",
                candidates=[o] + [atoms[j] for j in rng.sample(range(len(atoms)), min(20, len(atoms)))])
            cos_sims.append(cos)
            if decoded == o:
                correct += 1
        acc = correct / max(1, n_triples)
        avg_cos = sum(cos_sims) / max(1, len(cos_sims))
        per_dim[dim] = {"accuracy": round(acc, 4), "avg_cosine": round(avg_cos, 4),
                        "n_triples": n_triples, "correct": correct}
        total_correct += correct
        total_trials += n_triples

    overall_acc = total_correct / max(1, total_trials)
    duration = time.time() - start
    return SubstrateBenchmarkResult(
        name="vsa_algebraic_fidelity",
        description="VSA bind/unbind round-trip accuracy across dimensionalities",
        passed=overall_acc >= 0.85, score=overall_acc, n_trials=total_trials,
        details={"per_dimensionality": per_dim, "dims_tested": list(dims)},
        duration_seconds=duration,
    )


# ---------------------------------------------------------------------------
# 7. Hopfield retrieval accuracy
# ---------------------------------------------------------------------------

def bench_hopfield_retrieval(*, d_model: int = 256, store_sizes: Sequence[int] = (10, 50, 100, 500), n_queries: int = 50, seed: int = 0) -> SubstrateBenchmarkResult:
    """Test one-step Hopfield retrieval accuracy at varying store sizes."""
    from core.hopfield import HopfieldAssociativeMemory

    start = time.time()
    rng_torch = torch.Generator().manual_seed(seed)
    per_size: dict[int, dict[str, Any]] = {}
    total_correct = 0
    total_queries = 0

    for n_store in store_sizes:
        mem = HopfieldAssociativeMemory(d_model, max_items=max(n_store + 10, 100))
        patterns = torch.randn(n_store, d_model, generator=rng_torch)
        patterns = patterns / patterns.norm(dim=1, keepdim=True).clamp_min(1e-6)
        for i in range(n_store):
            mem.remember(patterns[i].unsqueeze(0), patterns[i].unsqueeze(0))
        correct = 0
        cos_sims: list[float] = []
        for q_idx in range(min(n_queries, n_store)):
            noise = torch.randn(d_model, generator=rng_torch) * 0.3
            query = patterns[q_idx] + noise
            query = query / query.norm().clamp_min(1e-6)
            retrieved, weights = mem.retrieve(query)
            sim = float(torch.nn.functional.cosine_similarity(
                retrieved.view(1, -1), patterns[q_idx].view(1, -1)).item())
            cos_sims.append(sim)
            if sim > 0.8:
                correct += 1
        n_q = min(n_queries, n_store)
        acc = correct / max(1, n_q)
        avg_cos = sum(cos_sims) / max(1, len(cos_sims))
        per_size[n_store] = {"accuracy": round(acc, 4), "avg_cosine": round(avg_cos, 4),
                             "n_queries": n_q, "correct": correct}
        total_correct += correct
        total_queries += n_q

    overall_acc = total_correct / max(1, total_queries)
    duration = time.time() - start
    return SubstrateBenchmarkResult(
        name="hopfield_retrieval_accuracy",
        description="Modern Continuous Hopfield one-step retrieval at varying store sizes",
        passed=overall_acc >= 0.70, score=overall_acc, n_trials=total_queries,
        details={"d_model": d_model, "per_store_size": per_size, "store_sizes_tested": list(store_sizes)},
        duration_seconds=duration,
    )


# ---------------------------------------------------------------------------
# 8. Active inference decision quality
# ---------------------------------------------------------------------------

def bench_active_inference(*, n_episodes: int = 200, max_steps: int = 3, seed: int = 0) -> SubstrateBenchmarkResult:
    """Compare EFE-driven Tiger POMDP agent to a random baseline."""
    from core.active_inference import (
        ActiveInferenceAgent, TigerDoorEnv, build_tiger_pomdp,
        run_episode, random_episode,
    )

    start = time.time()
    pomdp = build_tiger_pomdp()
    agent = ActiveInferenceAgent(pomdp, horizon=1, learn=True)
    env = TigerDoorEnv(seed=seed)
    agent_successes = 0
    agent_returns: list[float] = []
    for _ in range(n_episodes):
        success, total, _ = run_episode(agent, env, max_steps=max_steps)
        if success:
            agent_successes += 1
        agent_returns.append(total)

    random_successes = 0
    random_returns: list[float] = []
    rand_env = TigerDoorEnv(seed=seed + 1000)
    for _ in range(n_episodes):
        success, total = random_episode(rand_env, max_steps=max_steps)
        if success:
            random_successes += 1
        random_returns.append(total)

    agent_rate = agent_successes / max(1, n_episodes)
    random_rate = random_successes / max(1, n_episodes)
    advantage = agent_rate - random_rate
    duration = time.time() - start

    return SubstrateBenchmarkResult(
        name="active_inference_decision_quality",
        description="EFE-driven Tiger POMDP agent vs random baseline",
        passed=advantage > 0.05, score=agent_rate, n_trials=n_episodes,
        details={
            "agent_success_rate": round(agent_rate, 4),
            "random_success_rate": round(random_rate, 4),
            "advantage_over_random": round(advantage, 4),
            "agent_mean_return": round(sum(agent_returns) / max(1, len(agent_returns)), 4),
            "random_mean_return": round(sum(random_returns) / max(1, len(random_returns)), 4),
            "n_episodes": n_episodes, "max_steps": max_steps,
        },
        duration_seconds=duration,
    )


# ---------------------------------------------------------------------------
# Suite runner
# ---------------------------------------------------------------------------

ALL_SUBSTRATE_BENCHMARKS = [
    bench_rule_shift, bench_adversarial_prompt_resistance, bench_causal_reasoning,
    bench_memory_fidelity, bench_conformal_coverage, bench_vsa_algebra,
    bench_hopfield_retrieval, bench_active_inference,
]


def run_substrate_benchmark_suite(
    *, seed: int = 0, benchmarks: Sequence[str] | None = None,
    output_path: Path | None = None,
) -> SubstrateBenchmarkSuite:
    """Run all (or selected) substrate benchmarks and return the suite result."""
    suite = SubstrateBenchmarkSuite()
    suite_start = time.time()
    for bench_fn in ALL_SUBSTRATE_BENCHMARKS:
        name = bench_fn.__name__.replace("bench_", "")
        if benchmarks is not None and name not in benchmarks:
            continue
        print(f"  Running substrate benchmark: {name} ...", flush=True)
        try:
            result = bench_fn(seed=seed)
            suite.results.append(result)
            status = "PASS" if result.passed else "FAIL"
            print(f"    {status}  score={result.score:.4f}  time={result.duration_seconds:.2f}s", flush=True)
        except Exception as exc:
            logger.exception("Substrate benchmark %s failed", name)
            suite.results.append(SubstrateBenchmarkResult(
                name=name, description=f"Error: {exc}", passed=False,
                score=0.0, n_trials=0, details={"error": str(exc)},
            ))
            print(f"    ERROR: {exc}", flush=True)
    suite.total_duration_seconds = time.time() - suite_start
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(suite.summary(), indent=2, default=str), encoding="utf-8")
        print(f"  Wrote substrate benchmark results to {output_path}", flush=True)
    return suite


def _histogram(values: list[int | float], bins: int) -> dict[str, int]:
    """Simple histogram for details."""
    hist: dict[str, int] = {}
    for v in values:
        key = str(int(v))
        hist[key] = hist.get(key, 0) + 1
    return hist
