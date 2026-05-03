"""Tests that the derived ``substrate_target_snr_scale`` reaches every graft.

These are the integration tests for the rule "use dynamic/derived values in
favor of tunable parameters". The substrate computes a single scale per
forward pass, ships it through ``state["substrate_target_snr_scale"]``, and
every graft must respect it: at scale=0 the residual stream and logits must
be returned untouched (modulo numerical noise from the cloning), regardless
of how strongly the static SNR cap is set.

The scale plumbing has to work for:

* :func:`snr_magnitude` itself (the shared scaling primitive),
* :class:`FeatureVectorGraft` (residual-stream feature bridge),
* :class:`SubstrateLogitBiasGraft` (logit-level bias).

A failure here means a future graft with bias is silently ignoring the
substrate's "do not push" signal — the regression we want to catch.
"""

from __future__ import annotations

import math

import pytest

import torch

from core.grafting.grafts import (
    DEFAULT_GRAFT_TARGET_SNR,
    FeatureVectorGraft,
    snr_magnitude,
    state_target_snr_scale,
)
from core.grafts import SubstrateLogitBiasGraft


class TestSnrMagnitudeRespectsSubstrateScale:
    def test_scale_zero_zeros_magnitude(self):
        x = torch.randn(2, 4, 16)
        m = snr_magnitude(
            x,
            target_snr=DEFAULT_GRAFT_TARGET_SNR,
            confidence=1.0,
            inertia=1.0,
            substrate_scale=0.0,
        )
        assert torch.all(m == 0.0)

    def test_scale_one_matches_legacy_behavior(self):
        x = torch.randn(2, 4, 16)
        legacy = snr_magnitude(
            x, target_snr=DEFAULT_GRAFT_TARGET_SNR, confidence=0.7, inertia=1.5
        )
        with_scale_one = snr_magnitude(
            x,
            target_snr=DEFAULT_GRAFT_TARGET_SNR,
            confidence=0.7,
            inertia=1.5,
            substrate_scale=1.0,
        )
        assert torch.allclose(legacy, with_scale_one)

    def test_negative_scale_clamps_to_zero(self):
        x = torch.randn(2, 4, 16)
        m = snr_magnitude(
            x,
            target_snr=DEFAULT_GRAFT_TARGET_SNR,
            confidence=1.0,
            inertia=1.0,
            substrate_scale=-2.0,
        )
        assert torch.all(m == 0.0)

    def test_scale_is_multiplicative(self):
        x = torch.randn(2, 4, 16)
        a = snr_magnitude(
            x, target_snr=0.5, confidence=1.0, inertia=1.0, substrate_scale=1.0
        )
        b = snr_magnitude(
            x, target_snr=0.5, confidence=1.0, inertia=1.0, substrate_scale=0.4
        )
        assert torch.allclose(b, 0.4 * a, rtol=1e-6, atol=1e-6)


class TestStateTargetSnrScaleReader:
    def test_default_when_absent(self):
        assert state_target_snr_scale({}) == 1.0

    def test_reads_value_when_present(self):
        assert state_target_snr_scale({"substrate_target_snr_scale": 0.42}) == pytest.approx(0.42)

    def test_none_value_falls_back_to_default(self):
        assert state_target_snr_scale({"substrate_target_snr_scale": None}) == 1.0

    def test_non_numeric_value_falls_back_to_default(self):
        assert state_target_snr_scale({"substrate_target_snr_scale": "oops"}) == 1.0


class TestFeatureVectorGraftRespectsScale:
    """The residual-stream feature bridge collapses to a no-op at scale=0."""

    def _state(self, *, scale: float, x: torch.Tensor) -> dict:
        return {
            "substrate_target_snr_scale": scale,
            "substrate_confidence": 1.0,
            "substrate_inertia": 1.0,
            "faculty_features": torch.randn(x.shape[0], 8),
            "token_ids": torch.zeros(x.shape[0], x.shape[1], dtype=torch.long),
            "attention_mask": torch.ones(x.shape[0], x.shape[1], dtype=torch.bool),
        }

    def test_scale_zero_does_not_modify_hidden_state(self):
        torch.manual_seed(0)
        x = torch.randn(2, 4, 16)
        graft = FeatureVectorGraft(d_features=8, d_model=16)
        out = graft.forward(x, self._state(scale=0.0, x=x))
        assert torch.allclose(out, x, atol=1e-6)

    def test_scale_one_does_modify_hidden_state(self):
        torch.manual_seed(0)
        x = torch.randn(2, 4, 16)
        graft = FeatureVectorGraft(d_features=8, d_model=16)
        out = graft.forward(x, self._state(scale=1.0, x=x))
        assert not torch.allclose(out, x, atol=1e-6)


class TestSubstrateLogitBiasGraftRespectsScale:
    """At scale=0 the graft must leave logits untouched even when bias is set."""

    def _state(self, *, scale: float, seq_len: int) -> dict:
        return {
            "substrate_target_snr_scale": scale,
            "substrate_confidence": 1.0,
            "substrate_inertia": 1.0,
            "broca_logit_bias": {3: 1.0, 7: 1.0},
            "broca_logit_bias_decay": 1.0,
            "last_indices": torch.tensor([seq_len - 1]),
        }

    def test_scale_zero_returns_logits_unchanged(self):
        torch.manual_seed(0)
        seq_len = 5
        vocab = 16
        logits = torch.randn(1, seq_len, vocab)
        graft = SubstrateLogitBiasGraft()
        out = graft.forward(logits, self._state(scale=0.0, seq_len=seq_len))
        assert torch.allclose(out, logits, atol=1e-6)

    def test_scale_one_modifies_logits(self):
        torch.manual_seed(0)
        seq_len = 5
        vocab = 16
        logits = torch.randn(1, seq_len, vocab)
        graft = SubstrateLogitBiasGraft()
        out = graft.forward(logits, self._state(scale=1.0, seq_len=seq_len))
        assert not torch.allclose(out, logits, atol=1e-6)

    def test_intermediate_scale_is_proportional(self):
        torch.manual_seed(0)
        seq_len = 5
        vocab = 16
        logits = torch.randn(1, seq_len, vocab)
        graft = SubstrateLogitBiasGraft()
        full = graft.forward(logits.clone(), self._state(scale=1.0, seq_len=seq_len))
        half = graft.forward(logits.clone(), self._state(scale=0.5, seq_len=seq_len))
        full_delta = (full - logits)[0, seq_len - 1]
        half_delta = (half - logits)[0, seq_len - 1]
        # Where the full delta is non-zero, the half delta should be ~half.
        nonzero = full_delta.abs() > 1e-9
        if nonzero.any():
            ratio = (half_delta[nonzero] / full_delta[nonzero]).mean().item()
            assert math.isclose(ratio, 0.5, rel_tol=1e-3, abs_tol=1e-6)
