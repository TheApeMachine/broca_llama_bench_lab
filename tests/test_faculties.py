from __future__ import annotations

import math

from core.experiments import (
    run_active_inference_experiment,
    run_causal_experiment,
)


def test_active_inference_prefers_information_before_action():
    result = run_active_inference_experiment(seed=0, episodes=60, verbose=False)
    assert result["first_action"] == "listen"
    assert result["active_avg_reward"] > result["random_avg_reward"]
    assert result["active_success"] >= result["random_success"]


def test_pearl_backdoor_frontdoor_and_counterfactuals():
    result = run_causal_experiment(verbose=False)
    assert result["observational_t1"] < result["observational_t0"]
    assert result["do_t1"] > result["do_t0"]
    assert result["backdoor_sets"][0] == ["S"]
    # Pearl's backdoor / front-door identities are exact theorems; under the
    # adaptive Monte-Carlo path both sides carry independent Wilson-coverage
    # noise of order 1/sqrt(MC_SAMPLE_BUDGET) ≈ 0.01, so the agreement target
    # is the joint coverage band rather than machine epsilon.
    assert math.isclose(result["adjusted_t1"], result["do_t1"], abs_tol=0.03)
    assert math.isclose(result["adjusted_t0"], result["do_t0"], abs_tol=0.03)
    assert result["frontdoor_sets"][0] == ["M"]
    assert math.isclose(result["frontdoor_formula_x1"], result["frontdoor_do_x1"], abs_tol=0.03)
    assert 0.0 <= result["counterfactual_success_if_untreated"] <= 1.0
