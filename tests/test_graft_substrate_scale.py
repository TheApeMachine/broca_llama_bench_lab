"""Tests that the derived ``substrate_target_snr_scale`` reaches every graft.

These are the integration tests for the rule "use dynamic/derived values in
favor of tunable parameters". The substrate computes a single scale per
forward pass, ships it through ``state["substrate_target_snr_scale"]``, and
every graft must respect it: at scale=0 the residual stream must be returned
untouched (modulo numerical noise from the cloning), regardless of how
strongly the static SNR cap is set.

The scale plumbing has to work for:

* :func:`snr_magnitude` itself (the shared scaling primitive),
* :class:`FeatureVectorGraft` (residual-stream feature bridge),
* :class:`SubstrateConceptGraft` (continuous concept attraction / repulsion).

A failure here means a future graft is silently ignoring the substrate's "do
not push" signal — the regression we want to catch.
"""

from __future__ import annotations

import math

import pytest

import torch
import torch.nn as nn

from core.grafting.grafts import (
    DEFAULT_GRAFT_TARGET_SNR,
    FeatureVectorGraft,
    snr_magnitude,
    state_target_snr_scale,
)
from core.grafts import SubstrateConceptGraft


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


class _StubLM(nn.Module):
    """Minimal stand-in for the Broca host: exposes ``lm_head.weight`` only."""

    def __init__(self, *, vocab: int, d_model: int):
        super().__init__()
        self.lm_head = nn.Linear(d_model, vocab, bias=False)


class TestSubstrateConceptGraftRespectsScale:
    """At scale=0 the graft must leave the residual stream untouched."""

    def _state(self, *, scale: float, seq_len: int, d_model: int, vocab: int) -> dict:
        torch.manual_seed(7)
        return {
            "substrate_target_snr_scale": scale,
            "substrate_confidence": 1.0,
            "substrate_inertia": 1.0,
            "broca_concept_token_ids": {"alpha": [3, 5], "beta": [7]},
            "broca_repulsion_token_ids": {"omega": [11]},
            "last_indices": torch.tensor([seq_len - 1]),
            "model": _StubLM(vocab=vocab, d_model=d_model),
        }

    def test_scale_zero_returns_residual_unchanged(self):
        torch.manual_seed(0)
        seq_len = 5
        d_model = 16
        vocab = 32
        x = torch.randn(1, seq_len, d_model)
        graft = SubstrateConceptGraft()
        out = graft.forward(
            x, self._state(scale=0.0, seq_len=seq_len, d_model=d_model, vocab=vocab)
        )
        assert torch.allclose(out, x, atol=1e-6)

    def test_scale_one_modifies_residual(self):
        torch.manual_seed(0)
        seq_len = 5
        d_model = 16
        vocab = 32
        x = torch.randn(1, seq_len, d_model)
        graft = SubstrateConceptGraft()
        out = graft.forward(
            x, self._state(scale=1.0, seq_len=seq_len, d_model=d_model, vocab=vocab)
        )
        assert not torch.allclose(out, x, atol=1e-6)

    def test_intermediate_scale_is_proportional(self):
        torch.manual_seed(0)
        seq_len = 5
        d_model = 16
        vocab = 32
        x = torch.randn(1, seq_len, d_model)
        graft = SubstrateConceptGraft()
        full_state = self._state(scale=1.0, seq_len=seq_len, d_model=d_model, vocab=vocab)
        half_state = self._state(scale=0.5, seq_len=seq_len, d_model=d_model, vocab=vocab)
        # Reuse the same lm_head so the directions are identical between runs.
        half_state["model"] = full_state["model"]
        full = graft.forward(x.clone(), full_state)
        half = graft.forward(x.clone(), half_state)
        full_delta = (full - x)[0, seq_len - 1]
        half_delta = (half - x)[0, seq_len - 1]
        nonzero = full_delta.abs() > 1e-9
        if nonzero.any():
            ratio = (half_delta[nonzero] / full_delta[nonzero]).mean().item()
            assert math.isclose(ratio, 0.5, rel_tol=1e-3, abs_tol=1e-6)
