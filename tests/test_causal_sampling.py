from __future__ import annotations

import math

from asi_broca_core.causal import FiniteSCM, build_simpson_scm


def test_sampled_counterfactual_matches_exact_on_small_scm():
    scm = build_simpson_scm()
    exact = scm.counterfactual_probability_exact({"Y": 1}, evidence={"S": 1, "T": 1, "Y": 1}, interventions={"T": 0})
    sampled = scm.counterfactual_probability(
        {"Y": 1},
        evidence={"S": 1, "T": 1, "Y": 1},
        interventions={"T": 0},
        n_samples=60_000,
        seed=4,
    )

    assert math.isclose(sampled, exact, abs_tol=0.05)


def test_large_counterfactual_uses_sampling_without_exhaustive_worlds():
    scm = FiniteSCM(domains={})
    scm.add_exogenous("U_X", range(1000))
    scm.add_exogenous("U_Y", range(1000))
    scm.add_endogenous("X", [0, 1], ["U_X"], lambda v: 1 if v["U_X"] < 900 else 0)
    scm.add_endogenous("Y", [0, 1], ["X", "U_Y"], lambda v: 1 if (v["X"] == 1 and v["U_Y"] < 800) else 0)

    # FiniteSCM.counterfactual_probability falls back to Monte Carlo for large exogenous Cartesian products.
    # There is no public hook to force that path, so we patch the private enumerator to ensure this test never
    # walks 1000×1000 worlds (brittle if _exogenous_worlds is renamed — update this test if so).
    def fail_enumeration():
        raise AssertionError("exact exogenous enumeration should not run")

    scm._exogenous_worlds = fail_enumeration  # type: ignore[method-assign]
    got = scm.counterfactual_probability(
        {"Y": 1},
        evidence={"X": 1},
        interventions={"X": 1},
        seed=1,
    )

    assert 0.75 <= got <= 0.85
