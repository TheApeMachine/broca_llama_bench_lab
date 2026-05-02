"""Pragmatic intent classification for incoming utterances.

The substrate's relation extractor must not fire on every utterance: a request
("tell me a joke") is not a declarative claim, and forcing it through SVO
extraction stores garbage triples like ``(me, tell, joke)`` and then biases
the LLM toward verbalizing them. The :class:`IntentGate` is the first stop in
``comprehend()`` and decides which faculties downstream are even *allowed* to
run on this utterance.

Categories:

  ``statement`` — a declarative claim the substrate may store.
  ``question``  — an information-seeking utterance the substrate may answer.
  ``request``   — an imperative that asks the agent to do/produce something.
  ``command``   — a directive (close to ``request``; kept distinct so policy
                  can treat hard imperatives differently).
  ``greeting``  — phatic / social opener.
  ``feedback``  — acknowledgement / evaluation of a prior turn.

Only ``statement`` allows storage; ``statement`` and ``question`` are
considered *actionable* (the substrate has something it could legitimately
contribute). Everything else is non-actionable: the cognitive frame should
end up ``intent="unknown"`` with confidence 0 so derived graft strength
collapses to 0 and the LLM speaks freely.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..encoders.extraction import ExtractionEncoder
from ..system.event_bus import get_default_bus
from .lexical_intent import LexicalIntentClassifier

logger = logging.getLogger(__name__)


def _publish(topic: str, payload: dict) -> None:
    try:
        get_default_bus().publish(topic, payload)
    except Exception:
        pass


INTENT_LABELS: tuple[str, ...] = (
    "statement",
    "question",
    "request",
    "command",
    "greeting",
    "feedback",
)

ACTIONABLE_LABELS: frozenset[str] = frozenset({"statement", "question"})

STORABLE_LABELS: frozenset[str] = frozenset({"statement"})


@dataclass(frozen=True)
class UtteranceIntent:
    """Outcome of the intent gate.

    Attributes:
        label:           Top intent label from :data:`INTENT_LABELS`.
        confidence:      Classifier score in ``[0, 1]``.
        is_actionable:   The substrate may contribute (statement / question).
        allows_storage:  The substrate may *store* a triple (statement only).
        scores:          All label → score pairs for diagnostics / logging.
    """

    label: str
    confidence: float
    is_actionable: bool
    allows_storage: bool
    scores: dict[str, float]


class IntentGate:
    """Zero-shot intent classifier backed by :class:`ExtractionEncoder`.

    The gate exposes a single :meth:`classify` method that returns an
    :class:`UtteranceIntent`. Implementation is intentionally thin — the gate
    is responsible for *deciding*, not for owning a model. All ML capacity
    lives in the extraction encoder.
    """

    def __init__(
        self,
        extraction: ExtractionEncoder,
        *,
        labels: tuple[str, ...] = INTENT_LABELS,
        actionable_labels: frozenset[str] = ACTIONABLE_LABELS,
        storable_labels: frozenset[str] = STORABLE_LABELS,
    ):
        if not labels:
            raise ValueError("IntentGate requires at least one label")
        unknown_actionable = actionable_labels - set(labels)
        if unknown_actionable:
            raise ValueError(
                f"actionable_labels {sorted(unknown_actionable)} not in labels {labels}"
            )
        unknown_storable = storable_labels - set(labels)
        if unknown_storable:
            raise ValueError(
                f"storable_labels {sorted(unknown_storable)} not in labels {labels}"
            )
        self._extraction = extraction
        self._lexical = LexicalIntentClassifier()
        self._labels = tuple(labels)
        self._actionable = frozenset(actionable_labels)
        self._storable = frozenset(storable_labels)

    @property
    def labels(self) -> tuple[str, ...]:
        return self._labels

    def classify(self, utterance: str) -> UtteranceIntent:
        text = (utterance or "").strip()
        if not text:
            return UtteranceIntent(
                label="greeting",
                confidence=0.0,
                is_actionable=False,
                allows_storage=False,
                scores={l: 0.0 for l in self._labels},
            )

        lexical = self._lexical.classify(text, labels=self._labels)
        if lexical is not None:
            top_label, top_score, lexical_scores = lexical
            intent = self._intent_from_scores(top_label, top_score, lexical_scores)
            self._record(text, intent, source="lexical")
            return intent

        ranked = self._extraction.classify(
            text,
            labels=self._labels,
            multi_label=False,
            threshold=0.0,
        )
        scores = {label: 0.0 for label in self._labels}
        for label, score in ranked:
            if label in scores:
                scores[label] = float(score)
        if ranked:
            top_label, top_score = ranked[0]
        else:
            top_label, top_score = self._labels[0], 0.0
        intent = self._intent_from_scores(top_label, top_score, scores)
        self._record(text, intent, source="encoder")
        return intent

    def _intent_from_scores(self, top_label: str, top_score: float, raw_scores: dict[str, float]) -> UtteranceIntent:
        scores = {label: 0.0 for label in self._labels}
        for label, score in raw_scores.items():
            if label in scores:
                scores[label] = float(score)
        return UtteranceIntent(
            label=top_label,
            confidence=float(top_score),
            is_actionable=top_label in self._actionable,
            allows_storage=top_label in self._storable,
            scores=scores,
        )

    def _record(self, text: str, intent: UtteranceIntent, *, source: str) -> None:
        logger.debug(
            "IntentGate.classify: source=%s utterance=%r label=%s conf=%.3f actionable=%s",
            source,
            text[:160],
            intent.label,
            intent.confidence,
            intent.is_actionable,
        )
        _publish(
            "cog.intent",
            {
                "utterance": text[:120],
                "label": intent.label,
                "confidence": intent.confidence,
                "is_actionable": intent.is_actionable,
                "allows_storage": intent.allows_storage,
                "scores": dict(intent.scores),
                "source": source,
            },
        )
