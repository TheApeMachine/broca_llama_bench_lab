"""Tests for derived graft strength.

These tests are the spec for the rule "use dynamic/derived values in favor of
tunable parameters". When the substrate has nothing useful to contribute, the
strength must collapse to zero so the LLM speaks freely. When every faculty
agrees, the strength approaches one and the grafts are allowed to push at
their full SNR cap. Anything in between must be a *nudge*, not a *hammer*.
"""

from __future__ import annotations

import math

import pytest

from core.chat.derived_strength import DerivedStrength, StrengthInputs


class TestStrengthCollapsesWhenSubstrateIsEmpty:
    """A substrate with nothing legitimate to say must inject zero bias."""

    def test_non_actionable_intent_zeroes_strength(self):
        strength = DerivedStrength.compute(
            StrengthInputs(
                intent_actionability=0.0,
                memory_confidence=0.95,
                conformal_set_size=1,
                affect_certainty=0.8,
            )
        )
        assert strength == 0.0

    def test_zero_memory_confidence_zeroes_strength(self):
        strength = DerivedStrength.compute(
            StrengthInputs(
                intent_actionability=1.0,
                memory_confidence=0.0,
                conformal_set_size=1,
                affect_certainty=1.0,
            )
        )
        assert strength == 0.0

    def test_zero_affect_certainty_zeroes_strength(self):
        strength = DerivedStrength.compute(
            StrengthInputs(
                intent_actionability=1.0,
                memory_confidence=0.9,
                conformal_set_size=1,
                affect_certainty=0.0,
            )
        )
        assert strength == 0.0


class TestStrengthRespondsToConformalSharpness:
    """``|C|=1`` is decisive; larger sets nudge, not hammer."""

    def test_conformal_one_is_decisive(self):
        strength = DerivedStrength.compute(
            StrengthInputs(
                intent_actionability=1.0,
                memory_confidence=0.95,
                conformal_set_size=1,
                affect_certainty=1.0,
            )
        )
        assert strength == pytest.approx(0.95, rel=1e-6)

    def test_conformal_three_decays_strength(self):
        strength = DerivedStrength.compute(
            StrengthInputs(
                intent_actionability=1.0,
                memory_confidence=0.9,
                conformal_set_size=3,
                affect_certainty=1.0,
            )
        )
        # 0.9 * (1/3) * 1.0 * 1.0 ≈ 0.30
        assert strength == pytest.approx(0.3, rel=1e-3)

    def test_conformal_zero_means_no_extra_penalty(self):
        """``|C|=0`` is the absence of a conformal prediction, not ambiguity."""

        with_conformal = DerivedStrength.compute(
            StrengthInputs(
                intent_actionability=1.0,
                memory_confidence=0.7,
                conformal_set_size=1,
                affect_certainty=1.0,
            )
        )
        without_conformal = DerivedStrength.compute(
            StrengthInputs(
                intent_actionability=1.0,
                memory_confidence=0.7,
                conformal_set_size=0,
                affect_certainty=1.0,
            )
        )
        assert with_conformal == pytest.approx(without_conformal, rel=1e-9)


class TestStrengthIsAlwaysClampedToUnitInterval:
    """The composed value cannot escape ``[0, 1]`` even if inputs do."""

    def test_inputs_above_one_clamp(self):
        strength = DerivedStrength.compute(
            StrengthInputs(
                intent_actionability=10.0,
                memory_confidence=10.0,
                conformal_set_size=1,
                affect_certainty=10.0,
            )
        )
        assert 0.0 <= strength <= 1.0

    def test_negative_inputs_clamp_to_zero(self):
        strength = DerivedStrength.compute(
            StrengthInputs(
                intent_actionability=-1.0,
                memory_confidence=0.9,
                conformal_set_size=1,
                affect_certainty=0.9,
            )
        )
        assert strength == 0.0

    def test_nan_input_raises(self):
        with pytest.raises(ValueError, match="not finite"):
            DerivedStrength.compute(
                StrengthInputs(
                    intent_actionability=1.0,
                    memory_confidence=float("nan"),
                    conformal_set_size=1,
                    affect_certainty=1.0,
                )
            )


class TestNudgeVsHammerBehavior:
    """The promised dynamic: weak evidence → nudge, strong evidence → hammer.

    Failure modes surfaced by these tests are the original bug: when the
    substrate had a garbage SVO triple (e.g., from "Tell me a joke") it
    *hammered* the LLM into babbling "joke, joke, joke". With derived
    strength, ambiguous evidence collapses the bias proportionally.
    """

    def test_low_memory_high_affect_is_a_nudge(self):
        strength = DerivedStrength.compute(
            StrengthInputs(
                intent_actionability=1.0,
                memory_confidence=0.4,  # weak retrieval
                conformal_set_size=1,
                affect_certainty=0.95,
            )
        )
        assert 0.2 <= strength <= 0.5, f"expected nudge range, got {strength}"

    def test_high_memory_high_affect_is_a_hammer(self):
        strength = DerivedStrength.compute(
            StrengthInputs(
                intent_actionability=1.0,
                memory_confidence=0.95,
                conformal_set_size=1,
                affect_certainty=0.95,
            )
        )
        assert strength >= 0.85, f"expected hammer range, got {strength}"

    def test_strength_is_monotone_in_memory_confidence(self):
        prev = -1.0
        for mem in (0.1, 0.3, 0.5, 0.7, 0.9):
            s = DerivedStrength.compute(
                StrengthInputs(
                    intent_actionability=1.0,
                    memory_confidence=mem,
                    conformal_set_size=1,
                    affect_certainty=0.9,
                )
            )
            assert s > prev, "strength must rise with memory confidence"
            prev = s
