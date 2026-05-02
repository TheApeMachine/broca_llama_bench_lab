"""Hierarchical semantic analysis feeding the cognition gates."""

from __future__ import annotations

from typing import Any

from ..encoders.classification import SemanticClassificationEncoder


class SemanticCascade:
    """Classify semantic axes, then collapse them into substrate intent."""

    AXES: dict[str, tuple[str, ...]] = {
        "speech_act": ("claim", "question", "request", "command", "greeting", "feedback"),
        "polarity": ("affirmation", "negation", "correction"),
        "content_role": ("self_description", "world_fact", "task_instruction", "social_signal"),
        "storage": ("storable", "non_storable"),
    }
    PROMPT = (
        "Classify this utterance for a cognitive substrate. Separate speech act, "
        "polarity, content role, and whether the utterance contains durable semantic content."
    )
    EXAMPLES: tuple[dict[str, Any], ...] = (
        {
            "text": "Ada lives in Rome.",
            "labels": [
                "speech_act.claim",
                "polarity.affirmation",
                "content_role.world_fact",
                "storage.storable",
            ],
        },
        {
            "text": "Where is Ada?",
            "labels": [
                "speech_act.question",
                "polarity.affirmation",
                "content_role.world_fact",
                "storage.non_storable",
            ],
        },
        {
            "text": "Tell me a joke.",
            "labels": [
                "speech_act.request",
                "polarity.affirmation",
                "content_role.task_instruction",
                "storage.non_storable",
            ],
        },
        {
            "text": "No, I did not say that.",
            "labels": [
                "speech_act.feedback",
                "polarity.correction",
                "content_role.social_signal",
                "storage.non_storable",
            ],
        },
        {
            "text": "I am the Magnificent.",
            "labels": [
                "speech_act.claim",
                "polarity.affirmation",
                "content_role.self_description",
                "storage.storable",
            ],
        },
    )
    INTENT_FROM_SPEECH_ACT: dict[str, str] = {
        "claim": "statement",
        "question": "question",
        "request": "request",
        "command": "command",
        "greeting": "greeting",
        "feedback": "feedback",
    }

    def __init__(
        self,
        *,
        classifier: SemanticClassificationEncoder,
    ):
        self.classifier = classifier
        self._labels = {axis: list(labels) for axis, labels in self.AXES.items()}

    def intent_scores(self, text: str) -> dict[str, Any]:
        if not text.strip():
            raise ValueError("SemanticCascade.intent_scores requires non-empty text")
        axes = self._classify_axes(text)
        speech_scores = self._require_axis(axes, "speech_act")
        semantic_scores = {
            canonical: float(speech_scores[source])
            for source, canonical in self.INTENT_FROM_SPEECH_ACT.items()
            if source in speech_scores
        }
        if set(semantic_scores) != set(self.INTENT_FROM_SPEECH_ACT.values()):
            raise RuntimeError(
                f"SemanticCascade.intent_scores: incomplete intent scores {semantic_scores!r}"
            )
        scores = dict(semantic_scores)

        label, confidence = max(scores.items(), key=lambda item: item[1])

        allows_storage = self._allows_storage(label, axes)
        return {
            "label": label,
            "confidence": float(confidence),
            "scores": scores,
            "allows_storage": allows_storage,
            "evidence": {
                "semantic_axes": axes,
                "semantic_allows_storage": allows_storage,
            },
        }

    def _classify_axes(self, text: str) -> dict[str, dict[str, float]]:
        return self.classifier.classify_axes(
            text,
            self._labels,
            prompt=self.PROMPT,
            examples=self.EXAMPLES,
        )

    def _allows_storage(
        self,
        label: str,
        axes: dict[str, dict[str, float]],
    ) -> bool:
        if label != "statement":
            return False
        storage_scores = self._require_axis(axes, "storage")
        for required in ("storable", "non_storable"):
            if required not in storage_scores:
                raise RuntimeError(f"SemanticCascade requires storage label {required!r}")
        return storage_scores["storable"] >= storage_scores["non_storable"]

    @staticmethod
    def _require_axis(axes: dict[str, dict[str, float]], axis: str) -> dict[str, float]:
        values = axes.get(axis)
        if not isinstance(values, dict) or not values:
            raise RuntimeError(f"SemanticCascade requires axis {axis!r}, got {axes!r}")
        return values
