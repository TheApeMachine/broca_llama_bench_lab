"""Derived graft strength."""

from __future__ import annotations

import math

from ...numeric import Probability
from ...workspace import WorkspacePublisher
from .inputs import StrengthInputs


class DerivedStrength:
    """Compose substrate signals into the graft's earned strength."""

    event_topic = "graft.derived_strength"
    probability = Probability()

    @classmethod
    def conformal_sharpness(cls, set_size: int) -> float:
        return cls.probability.inverse_cardinality(set_size)

    @classmethod
    def compute(cls, inputs: StrengthInputs) -> float:
        cls._reject_nan(
            actionability=inputs.intent_actionability,
            memory=inputs.memory_confidence,
            affect=inputs.affect_certainty,
        )

        actionability = cls.probability.unit_interval(inputs.intent_actionability)

        if actionability <= 0.0:
            WorkspacePublisher.emit(
                cls.event_topic,
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

        memory = cls.probability.unit_interval(inputs.memory_confidence)
        sharpness = cls.probability.unit_interval(
            cls.conformal_sharpness(int(inputs.conformal_set_size))
        )
        affect = cls.probability.unit_interval(inputs.affect_certainty)
        strength = actionability * memory * sharpness * affect

        if not math.isfinite(strength):
            raise ValueError(
                f"derived strength is not finite: actionability={actionability} "
                f"memory={memory} sharpness={sharpness} affect={affect}"
            )

        clamped = cls.probability.unit_interval(strength)

        WorkspacePublisher.emit(
            cls.event_topic,
            {
                "intent_actionability": actionability,
                "memory_confidence": memory,
                "conformal_set_size": int(inputs.conformal_set_size),
                "conformal_sharpness": sharpness,
                "affect_certainty": affect,
                "strength": clamped,
                "gated_by": cls._weakest_factor(actionability, memory, sharpness, affect),
            },
        )

        return clamped

    @classmethod
    def _weakest_factor(
        cls,
        actionability: float,
        memory: float,
        sharpness: float,
        affect: float,
    ) -> str:
        factors = (
            ("intent", actionability),
            ("memory", memory),
            ("sharpness", sharpness),
            ("affect", affect),
        )

        return min(factors, key=lambda kv: kv[1])[0]

    @classmethod
    def _reject_nan(cls, **named_values: float) -> None:
        bad = [name for name, value in named_values.items() if math.isnan(float(value))]

        if bad:
            raise ValueError(
                f"derived strength inputs are not finite: {', '.join(sorted(bad))}"
            )
