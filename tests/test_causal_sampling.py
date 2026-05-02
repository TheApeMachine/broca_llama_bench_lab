from __future__ import annotations

import math
import random

from core.causal import FiniteSCM, build_simpson_scm


def test_gibbs_counterfactual_matches_exact_on_small_scm():
    scm = build_simpson_scm()
    exact = scm.counterfactual_probability_exact({"Y": 1}, evidence={"S": 1, "T": 1, "Y": 1}, interventions={"T": 0})
    sampled = scm.counterfactual_probability(
        {"Y": 1},
        evidence={"S": 1, "T": 1, "Y": 1},
        interventions={"T": 0},
        n_samples=4_000,
        seed=4,
    )

    assert math.isclose(sampled, exact, abs_tol=0.05)


def test_large_counterfactual_uses_sampling_without_exhaustive_worlds():
    scm = FiniteSCM(domains={})
    scm.add_exogenous_uniform("U_X", range(1000))
    scm.add_exogenous_uniform("U_Y", range(1000))
    scm.add_endogenous("X", [0, 1], ["U_X"], lambda v: 1 if v["U_X"] < 900 else 0)
    scm.add_endogenous("Y", [0, 1], ["X", "U_Y"], lambda v: 1 if (v["X"] == 1 and v["U_Y"] < 800) else 0)

    # The Gibbs sampler must not call exhaustive enumeration; assert it never does.
    def fail_enumeration():
        raise AssertionError("exact exogenous enumeration should not run")

    scm._exogenous_worlds = fail_enumeration  # type: ignore[method-assign]
    got = scm.counterfactual_probability(
        {"Y": 1},
        evidence={"X": 1},
        interventions={"X": 1},
        n_samples=2_000,
        seed=1,
    )

    assert 0.75 <= got <= 0.85


def test_gibbs_counterfactual_handles_rare_evidence_via_local_search():
    """Evidence with prior probability ≈ 5e-4 — pure rejection cannot bootstrap
    a chain at any practical budget; the structural local-search initializer
    plus exact-conditional Gibbs must still recover the correct counterfactual.
    """

    scm = FiniteSCM(domains={})
    scm.add_exogenous_uniform("U_A", range(100))
    scm.add_exogenous_uniform("U_B", range(100))
    scm.add_exogenous_uniform("U_Y", range(100))
    scm.add_endogenous("A", [0, 1], ["U_A"], lambda v: 1 if v["U_A"] < 5 else 0)
    scm.add_endogenous("B", [0, 1], ["U_B"], lambda v: 1 if v["U_B"] < 10 else 0)
    scm.add_endogenous(
        "Y",
        [0, 1],
        ["A", "B", "U_Y"],
        lambda v: 1 if (v["A"] == 1 and v["B"] == 1 and v["U_Y"] < 80) else 0,
    )

    # P(A=1, B=1, Y=1) = 0.05 * 0.10 * 0.80 = 4e-3 — too rare for default rejection seeds with no chain advantage.
    got = scm.counterfactual_probability(
        {"Y": 1},
        evidence={"A": 1, "B": 1, "Y": 1},
        interventions={"A": 1, "B": 1},
        n_samples=2_000,
        seed=7,
    )

    # do(A=1, B=1) is consistent with the factual world; Y must remain 1.
    assert got >= 0.98


def test_gibbs_resample_returns_value_inside_evidence_support():
    """The conditional resampler must only emit values that keep evidence satisfied."""

    scm = FiniteSCM(domains={})
    scm.add_exogenous_uniform("U", range(20))
    scm.add_endogenous("E", [0, 1], ["U"], lambda v: 1 if v["U"] < 5 else 0)

    rng = random.Random(0)
    state = {"U": 2}
    for _ in range(50):
        state = scm._gibbs_resample(rng, "U", state, {"E": 1})
        assert state["U"] < 5, f"resample stepped outside evidence support: U={state['U']}"
