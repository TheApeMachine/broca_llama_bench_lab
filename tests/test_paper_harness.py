"""Unit tests for LaTeX experiment export (no GPU / no HF downloads)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.paper.harness import (
    refresh_paper_experiments,
    write_broca_architecture_experiment_tex,
    write_comparison_table_tex,
    write_experiment_inputs_manifest,
    write_hf_native_experiment_tex,
    write_vanilla_table_tex,
)


def test_write_vanilla_and_comparison_tables(tmp_path: Path) -> None:
    summary = {
        "per_task": {
            "boolq": {"n": 10, "accuracy": 0.5},
            "piqa": {"n": 8, "accuracy": 0.75},
        },
        "comparison": {
            "llama_broca_shell": {
                "per_task": {
                    "boolq": {"n": 10, "accuracy": 0.55},
                    "piqa": {"n": 8, "accuracy": 0.7},
                }
            }
        },
    }
    v = tmp_path / "vanilla.tex"
    c = tmp_path / "cmp.tex"
    write_vanilla_table_tex(summary, v)
    write_comparison_table_tex(summary, c)
    assert r"\begin{tabular}" in v.read_text(encoding="utf-8")
    assert "boolq" in v.read_text(encoding="utf-8")
    txt = c.read_text(encoding="utf-8")
    assert "Broca shell" in txt
    assert "0.0500" in txt or "-0.0500" in txt or "+0.0500" in txt


def test_experiment_inputs_manifest_sorted(tmp_path: Path) -> None:
    (tmp_path / "exp_zzz.tex").write_text("%\n", encoding="utf-8")
    (tmp_path / "exp_aaa.tex").write_text("%\n", encoding="utf-8")
    write_experiment_inputs_manifest(tmp_path)
    body = (tmp_path / "_inputs.tex").read_text(encoding="utf-8")
    assert body.index("exp_aaa") < body.index("exp_zzz")


def test_hf_native_tex_stub_and_success(tmp_path: Path) -> None:
    write_hf_native_experiment_tex(summary=None, exp_dir=tmp_path, error_message="unit test")
    stub = (tmp_path / "exp_hf_native_benchmark.tex").read_text(encoding="utf-8")
    assert "Status" in stub
    assert "unit test" in stub

    summary = {
        "model_id": "meta-llama/Llama-3.2-1B-Instruct",
        "tasks": ["boolq"],
        "limit_per_task": 3,
        "seed": 0,
        "created_at_utc": "20260101T000000Z",
        "scoring": {
            "multiple_choice": "length-normalized LL",
            "generation": "greedy match",
        },
    }
    write_hf_native_experiment_tex(summary=summary, exp_dir=tmp_path, accuracy_figure="hf_native_accuracy_by_task.pdf")
    full = (tmp_path / "exp_hf_native_benchmark.tex").read_text(encoding="utf-8")
    assert "Llama-3.2-1B-Instruct" in full
    assert "includegraphics" in full


def test_broca_arch_tex_from_fixture(tmp_path: Path) -> None:
    payload = json.loads(
        """
{
  "kind": "broca_architecture_eval",
  "description": "probe suite",
  "model_id": "meta-llama/Llama-3.2-1B-Instruct",
  "metrics": {
    "baseline_bare_language_host": {"speech_exact_accuracy": 0.0, "answer_present_accuracy": 0.5},
    "enhanced_broca_architecture": {"speech_exact_accuracy": 1.0, "answer_present_accuracy": 1.0},
    "delta_enhanced_minus_baseline": {"speech_exact_accuracy": 1.0, "answer_present_accuracy": 0.5}
  }
}
"""
    )
    write_broca_architecture_experiment_tex(payload, tmp_path)
    tex = (tmp_path / "exp_broca_architecture.tex").read_text(encoding="utf-8")
    assert "broca architecture probes" in tex.lower()
    assert "$1.000$" in tex


def test_refresh_skips_bench_with_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PAPER_SKIP_NATIVE", "1")
    monkeypatch.setenv("PAPER_SKIP_ARCH_EVAL", "1")
    report = refresh_paper_experiments(root=tmp_path)
    exp_root = tmp_path / "paper" / "include" / "experiment"
    assert exp_root.is_dir()
    assert (exp_root / "_inputs.tex").is_file()
    assert (report.get("hf_native") or {}).get("ok") is False
    assert (report.get("broca_architecture") or {}).get("skipped") is True
