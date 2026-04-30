from __future__ import annotations

import math

from asi_broca_core.experiments import (
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
    assert math.isclose(result["adjusted_t1"], result["do_t1"], rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(result["adjusted_t0"], result["do_t0"], rel_tol=1e-9, abs_tol=1e-9)
    assert result["frontdoor_sets"][0] == ["M"]
    assert math.isclose(result["frontdoor_formula_x1"], result["frontdoor_do_x1"], rel_tol=1e-9, abs_tol=1e-9)
    assert 0.0 <= result["counterfactual_success_if_untreated"] <= 1.0
