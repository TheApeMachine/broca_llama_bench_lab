"""Hypothesis conjunction tools must match the native-tool ``values: dict`` contract."""

from __future__ import annotations

from pathlib import Path

from core.calibration.conformal import ConformalPredictor
from core.causal import build_simpson_scm
from core.natives.hypothesis_synthesizer import HypothesisSynthesizer
from core.natives.native_tools import NativeToolRegistry


def test_hypothesis_conjunction_accepts_dict_values(tmp_path: Path) -> None:
    scm = build_simpson_scm()
    reg = NativeToolRegistry(tmp_path / "nt.sqlite", namespace="t")
    cold = ConformalPredictor(alpha=0.1, method="lac", min_calibration=10_000)
    synth = HypothesisSynthesizer(scm=scm, tool_registry=reg)
    tool = synth._synthesize_conjunction("S", "T", "hyp_S_AND_T")
    assert tool.name == "hyp_S_AND_T"
    assert tool.fn is not None
    assert tool.fn({"S": 0, "T": 0}) == 0
    assert tool.fn({"S": 1, "T": 1}) == 1
