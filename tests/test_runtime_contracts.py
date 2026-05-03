from __future__ import annotations

import pytest

from core.agent.active_inference import build_tiger_pomdp
from core.agent.invariants import POMDPInvariants
from core.calibration.conformal import ConformalPredictor
from core.kernel import CapabilityReport, manifest_for_profile
from core.system.device import normalize_device_arg, pick_torch_device


def test_device_normalization_respects_explicit_cpu() -> None:
    assert normalize_device_arg("cpu") == "cpu"
    assert str(pick_torch_device("cpu")) == "cpu"
    assert normalize_device_arg("auto") is None


def test_pomdp_invariants_hold_after_state_expansion() -> None:
    pomdp = build_tiger_pomdp()
    qs = [0.5, 0.5]
    pomdp.expand_state_with_mass("hypothesis", qs=qs, mass=0.12)
    report = POMDPInvariants().validate(pomdp, name="tiger")
    assert report.passed, [v.as_dict() for v in report.violations]


def test_aps_cold_conformal_is_conservative() -> None:
    predictor = ConformalPredictor(alpha=0.1, method="aps", min_calibration=8)
    result = predictor.predict_set({"a": 0.7, "b": 0.2, "c": 0.1})
    assert result.labels == ["a", "b", "c"]


def test_manifest_profiles_are_explicit() -> None:
    full = manifest_for_profile("full")
    no_recursion = manifest_for_profile("no_recursion")
    assert full.get("control.recursion").mode == "required"
    assert no_recursion.get("control.recursion").mode == "disabled"


def test_static_capability_report_surfaces_manifest() -> None:
    report = CapabilityReport.from_manifest(manifest_for_profile("full"), static_only=True)
    assert report.static_only is True
    assert any(record.key == "host.llama" for record in report.records)
    assert any(record.key == "swarm" and record.mode == "disabled" for record in report.records)
