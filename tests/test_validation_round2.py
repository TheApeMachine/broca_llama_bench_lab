from __future__ import annotations

from core.calibration.conformal import ConformalPredictor
from core.validation import ImplementationAuditor, StaticMathValidation
from core.validation.active_inference import ActiveInferenceValidator
from core.validation.causal_discovery import CausalDiscoveryStability
from core.validation.conformal import ConformalCoverageEvaluator


def test_static_math_validation_passes_model_free_contracts() -> None:
    report = StaticMathValidation.run(include_tiger_metric=False)
    assert report.status == "pass"
    assert {item.name for item in report.invariants} >= {
        "tiger_pomdp",
        "expanded_tiger_pomdp",
        "simpson_scm",
        "cold_aps",
    }
    assert report.metrics["cold_aps_set"]["set_size"] == 3


def test_implementation_audit_surfaces_active_inference_gaps() -> None:
    scorecard = ImplementationAuditor().audit("full")
    active = {score.key: score for score in scorecard.scores}
    assert scorecard.status == "incomplete"
    assert active["reasoning.active_inference"].status == "incomplete"
    kinds = {gap.kind for gap in active["reasoning.active_inference"].gaps}
    assert {"domain", "policy_search", "learning"}.issubset(kinds)


def test_conformal_coverage_evaluator_reports_set_metrics() -> None:
    predictor = ConformalPredictor(alpha=0.2, method="lac", min_calibration=2)
    predictor.calibrate(p_label=0.8)
    predictor.calibrate(p_label=0.7)
    examples = [({"yes": 0.9, "no": 0.1}, "yes"), ({"yes": 0.2, "no": 0.8}, "no")]
    report = ConformalCoverageEvaluator().evaluate(predictor, examples)
    assert report.n_examples == 2
    assert report.empirical_coverage == 1.0
    assert report.average_set_size >= 1.0


def test_active_inference_validator_runs_tiger_smoke() -> None:
    report = ActiveInferenceValidator().tiger_smoke(seed=0, episodes=4)
    assert report.invariant_status == "pass"
    assert report.episodes == 4


def test_causal_discovery_stability_warns_on_tiny_samples() -> None:
    rows = [
        {"x": 0, "y": 0},
        {"x": 1, "y": 1},
        {"x": 1, "y": 1},
        {"x": 0, "y": 0},
    ]
    report = CausalDiscoveryStability().evaluate(rows, n_bootstrap=3, seed=1)
    assert report.n_rows == 4
    assert report.warnings
