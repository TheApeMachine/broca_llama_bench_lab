from pathlib import Path

from asi_broca_core.benchmarks.architecture_eval import run_broca_architecture_eval


def test_broca_architecture_eval_scores_enhanced_above_baseline(tmp_path: Path):
    out = tmp_path / "broca_architecture_eval.json"
    result = run_broca_architecture_eval(
        seed=0,
        db_path=tmp_path / "architecture_eval.sqlite",
        backend="tiny",
        output_path=out,
    )

    metrics = result["metrics"]
    assert metrics["baseline_bare_language_host"]["speech_exact_accuracy"] == 0.0
    assert metrics["enhanced_broca_architecture"]["speech_exact_accuracy"] == 1.0
    assert metrics["delta_enhanced_minus_baseline"]["speech_exact_accuracy"] == 1.0
    assert out.exists()


