from __future__ import annotations

import math

from asi_broca_core.experiments import (
    run_active_inference_experiment,
    run_causal_experiment,
    run_memory_experiment,
    run_trainable_bridge_experiment,
    run_unified_stack_experiment,
)


def test_persistent_memory_survives_restart(tmp_path):
    result = run_memory_experiment(seed=0, db_path=tmp_path / "mem.sqlite", verbose=False)
    assert result["persisted_records"] == 8
    assert result["after_write"] == 1.0
    assert result["after_restart"] == 1.0


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


def test_unified_faculty_stack_answers_three_task_types(tmp_path):
    result = run_unified_stack_experiment(seed=0, db_path=tmp_path / "stack.sqlite", verbose=False)
    assert result["after"] == 1.0
    assert result["records"] == 8
    assert result["active_choice"] == "listen"
    assert result["causal_effects"]["ate"] > 0


def test_trainable_faculty_bridge_learns_with_frozen_host():
    result = run_trainable_bridge_experiment(seed=0, steps=160, verbose=False)
    assert result["after"] == 1.0
    assert result["bridge_params"] < result["host_total_params"] / 10

