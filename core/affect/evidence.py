"""AffectEvidence — substrate-side conversion of AffectState into JSON evidence.

Two stateless transformations the controller used to inline:

* :meth:`as_dict` — compact, JSON-friendly summary of an :class:`AffectState`,
  stored on every frame so derived graft strength, preference learning, and
  intrinsic cues all consume the same numbers.
* :meth:`certainty` — affect-driven certainty in ``[0, 1]`` derived from the
  GoEmotions distribution's peakedness; feeds graft strength derivation.
"""

from __future__ import annotations

from typing import Any

from ..encoders.affect import AffectState
from ..numeric import Probability


class AffectEvidence:
    """Stateless wrapper that turns an :class:`AffectState` into evidence shapes."""

    probability = Probability()

    @classmethod
    def as_dict(cls, affect: AffectState) -> dict[str, Any]:
        return {
            "dominant_emotion": str(affect.dominant_emotion),
            "dominant_score": float(affect.dominant_score),
            "confidences": [
                {"label": item.label, "score": float(item.score), "signal": item.signal}
                for item in affect.confidences
            ],
            "valence": float(affect.valence),
            "arousal": float(affect.arousal),
            "entropy": float(affect.entropy),
            "certainty": float(affect.certainty),
            "preference_signal": str(affect.preference_signal),
            "preference_strength": float(affect.preference_strength),
            "cognitive_states": dict(affect.cognitive_states),
        }

    @classmethod
    def certainty(cls, affect: AffectState | None) -> float:
        if affect is None:
            return 1.0
        if affect.confidences:
            return cls.probability.unit_interval(affect.certainty)
        return cls.probability.unit_interval(affect.dominant_score)
