from pathlib import Path

from asi_broca_core.benchmarks.architecture_eval import run_broca_architecture_eval


def test_broca_architecture_eval_writes_metrics(tmp_path: Path, llama_broca_loaded: None):
    out = tmp_path / "broca_architecture_eval.json"
    result = run_broca_architecture_eval(
        seed=0,
        db_path=tmp_path / "architecture_eval.sqlite",
        output_path=out,
    )

    assert result["kind"] == "broca_architecture_eval"
    assert result["model_id"]
    metrics = result["metrics"]
    for arm in ("baseline_bare_language_host", "enhanced_broca_architecture"):
        assert 0.0 <= metrics[arm]["speech_exact_accuracy"] <= 1.0
        assert 0.0 <= metrics[arm]["answer_present_accuracy"] <= 1.0
    assert out.exists()


