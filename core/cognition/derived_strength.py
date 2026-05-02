"""Derived graft strength — the substrate's bias is *earned*, not configured.

Until this module existed, every graft injected at a static
``target_snr=0.30`` whenever an utterance produced any non-trivial frame.
That is why the system *hammered* the LLM into babbling "joke, joke, joke,
joke" after parsing "Tell me a joke" as the triple ``(me, tell, joke)``: the
bias magnitude was the same whether the substrate had a high-confidence
memory recall or a garbage SVO triple it had no business storing.

A derived strength replaces the constant. It composes four substrate signals
multiplicatively, so the bias is *only* as strong as the weakest factor:

    strength = intent_actionability * memory_confidence
             * conformal_sharpness * affect_certainty

All four live in ``[0, 1]``. If any one is zero, the bias is zero — the
graft *nudges* when it has weak evidence and *hammers* only when every
faculty agrees. This is the rule "use dynamic/derived values in favor of
tunable parameters" applied to the most load-bearing knob in the host.

This module is intentionally a function on a class — :class:`DerivedStrength`
holds no state, so it is trivial to test and re-use from multiple call sites
(chat reply, plan-forced speak, benchmarks).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from ..system.event_bus import get_default_bus

logger = logging.getLogger(__name__)


def _publish(topic: str, payload: dict) -> None:
    try:
        get_default_bus().publish(topic, payload)
    except Exception:
        pass


@dataclass(frozen=True)
class StrengthInputs:
    """Inputs to derived graft strength.

    Attributes:
        intent_actionability: ``1.0`` when the substrate may contribute to the
            reply (statement / question), ``0.0`` for requests, greetings,
            commands, and acknowledgements.
        memory_confidence:    Frame confidence in ``[0, 1]``. For
            ``memory_lookup`` this is the persisted confidence of the recalled
            triple; for ``memory_write`` it is the extractor's confidence in
            the parsed claim.
        conformal_set_size:   ``|C|`` of the conformal predictor for the
            current question. ``0`` when no calibrated prediction is available
            — in that case the function falls back on memory confidence alone
            (no separate sharpness factor).
        affect_certainty:     ``1 - normalized_uncertainty`` from the affect
            encoder (one minus the entropy of its emotion distribution divided
            by ``log(n_emotions)``). High when one emotion dominates, low when
            the user's affect is ambiguous; multiplied so an ambiguous user
            cools the bias even if memory is confident.
    """

    intent_actionability: float
    memory_confidence: float
    conformal_set_size: int
    affect_certainty: float


class DerivedStrength:
    """Compose substrate signals into a single ``[0, 1]`` graft strength.

    The class holds no state. It exists so the policy is named and testable
    in isolation rather than buried inside ``SubstrateController``.
    """

    @staticmethod
    def conformal_sharpness(set_size: int) -> float:
        """Sharpness factor from a conformal prediction set size.

        ``|C|=1``  → ``1.0`` (decisive prediction, full strength).
        ``|C|=2+`` → ``1/|C|`` (linear decay in label ambiguity).
        ``|C|=0``  → ``1.0`` so the caller can choose to back off via
        ``memory_confidence`` rather than this term double-penalizing the
        absence of a conformal predictor.
        """

        if set_size <= 0:
            return 1.0
        return 1.0 / float(set_size)

    @classmethod
    def compute(cls, inputs: StrengthInputs) -> float:
        """Compose strength signals; clamp to ``[0, 1]``.

        NaN inputs raise — the substrate would rather expose an upstream
        bug than silently turn a corrupt signal into a confident bias.
        """
        _reject_nan(
            actionability=inputs.intent_actionability,
            memory=inputs.memory_confidence,
            affect=inputs.affect_certainty,
        )
        
        actionability = _clamp01(inputs.intent_actionability)
        
        if actionability <= 0.0:
            _publish(
                "cog.derived_strength",
                {
                    "intent_actionability": actionability,
                    "memory_confidence": float(inputs.memory_confidence),
                    "conformal_set_size": int(inputs.conformal_set_size),
                    "conformal_sharpness": 0.0,
                    "affect_certainty": float(inputs.affect_certainty),
                    "strength": 0.0,
                    "gated_by": "intent",
                },
            )
        
            return 0.0
        
        memory = _clamp01(inputs.memory_confidence)
        sharpness = _clamp01(cls.conformal_sharpness(int(inputs.conformal_set_size)))
        affect = _clamp01(inputs.affect_certainty)
        strength = actionability * memory * sharpness * affect

        if not math.isfinite(strength):
            raise ValueError(
                f"derived strength is not finite: actionability={actionability} "
                f"memory={memory} sharpness={sharpness} affect={affect}"
            )
        
        clamped = _clamp01(strength)
        
        _publish(
            "cog.derived_strength",
            {
                "intent_actionability": actionability,
                "memory_confidence": memory,
                "conformal_set_size": int(inputs.conformal_set_size),
                "conformal_sharpness": sharpness,
                "affect_certainty": affect,
                "strength": clamped,
                "gated_by": _weakest_factor(actionability, memory, sharpness, affect),
            },
        )
    
        return clamped


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _weakest_factor(actionability: float, memory: float, sharpness: float, affect: float) -> str:
    factors = (
        ("intent", actionability),
        ("memory", memory),
        ("sharpness", sharpness),
        ("affect", affect),
    )
    
    return min(factors, key=lambda kv: kv[1])[0]


def _reject_nan(**named_values: float) -> None:
    bad = [name for name, value in named_values.items() if math.isnan(float(value))]

    if bad:
        raise ValueError(
            f"derived strength inputs are not finite: {', '.join(sorted(bad))}"
        )
