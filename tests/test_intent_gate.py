"""Tests for the IntentGate.

The gate is the substrate's first defense against the original failure mode:
"Tell me a joke" — a request — being parsed by the relation extractor as the
declarative triple ``(me, tell, joke)`` and stored in semantic memory at
confidence 0.92. The gate must cleanly partition utterances into actionable
(statement / question) and non-actionable (request / greeting / command /
feedback / acknowledgment) categories so the relation extractor downstream
never even sees the non-actionable ones.

These tests use a stub organ rather than the real GLiNER2 weights so the
suite stays fast and the gate's *policy* — not the classifier's accuracy —
is what is under test. A separate, slower test file exercises the actual
ExtractionOrgan.classify call.
"""

from __future__ import annotations

from typing import Sequence

import pytest

from core.cognition.intent_gate import (
    ACTIONABLE_LABELS,
    INTENT_LABELS,
    IntentGate,
    UtteranceIntent,
)


class StubExtractionOrgan:
    """Pretends to be an :class:`ExtractionOrgan` for the gate's purposes.

    The gate only calls :meth:`classify` on the organ. The stub stores a
    canned response per text fragment so each test can spell out exactly
    what the classifier "sees" without any model weights.
    """

    def __init__(self, responses: dict[str, list[tuple[str, float]]]):
        self._responses = responses
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def classify(
        self,
        text: str,
        *,
        labels: Sequence[str],
        multi_label: bool = True,
        threshold: float = 0.0,
    ) -> list[tuple[str, float]]:
        self.calls.append((text, tuple(labels)))
        for fragment, scores in self._responses.items():
            if fragment in text.lower():
                return list(scores)
        return [(labels[0], 0.0)]


def _gate(responses: dict[str, list[tuple[str, float]]]) -> IntentGate:
    return IntentGate(StubExtractionOrgan(responses))


class TestIntentClassification:
    def test_request_is_not_actionable(self):
        gate = _gate({
            "tell me a joke": [("request", 0.91), ("statement", 0.04), ("question", 0.05)],
        })
        intent = gate.classify("Tell me a joke")
        assert intent.label == "request"
        assert intent.is_actionable is False
        assert intent.allows_storage is False

    def test_statement_is_storable(self):
        gate = _gate({
            "ada lives in rome": [("statement", 0.88), ("question", 0.07), ("request", 0.05)],
        })
        intent = gate.classify("Ada lives in Rome")
        assert intent.label == "statement"
        assert intent.is_actionable is True
        assert intent.allows_storage is True

    def test_question_is_actionable_but_not_storable(self):
        gate = _gate({
            "where is ada": [("question", 0.93), ("statement", 0.05), ("request", 0.02)],
        })
        intent = gate.classify("Where is Ada?")
        assert intent.label == "question"
        assert intent.is_actionable is True
        assert intent.allows_storage is False

    def test_greeting_is_neither_actionable_nor_storable(self):
        gate = _gate({
            "hi": [("greeting", 0.84), ("statement", 0.10), ("question", 0.06)],
        })
        intent = gate.classify("Hi")
        assert intent.label == "greeting"
        assert intent.is_actionable is False
        assert intent.allows_storage is False


class TestEdgeCases:
    def test_empty_utterance_is_safely_non_actionable(self):
        gate = _gate({})
        intent = gate.classify("")
        assert intent.is_actionable is False
        assert intent.allows_storage is False

    def test_whitespace_only_is_safely_non_actionable(self):
        gate = _gate({})
        intent = gate.classify("   \n\t  ")
        assert intent.is_actionable is False

    def test_organ_returning_no_results_falls_to_first_label(self):
        gate = _gate({})  # no fragment matches → stub returns one (label, 0.0)
        intent = gate.classify("totally unmatched text")
        assert intent.label in INTENT_LABELS
        assert intent.confidence == 0.0
        # First label is "statement" by default — that's actionable, but the
        # confidence is zero, which is the relevant signal for downstream
        # derived strength.
        assert intent.confidence == 0.0


class TestScoresAreAlwaysComplete:
    """Every label must appear in ``scores`` so callers can trust the dict."""

    def test_scores_always_contain_all_labels(self):
        gate = _gate({
            "ada lives in rome": [("statement", 0.88)],
        })
        intent = gate.classify("Ada lives in Rome")
        for label in INTENT_LABELS:
            assert label in intent.scores

    def test_unscored_labels_are_zero(self):
        gate = _gate({
            "ada lives in rome": [("statement", 0.88)],
        })
        intent = gate.classify("Ada lives in Rome")
        for label in INTENT_LABELS:
            if label != "statement":
                assert intent.scores[label] == 0.0


class TestConfigurationValidation:
    def test_actionable_labels_must_be_subset_of_labels(self):
        organ = StubExtractionOrgan({})
        with pytest.raises(ValueError, match="actionable_labels"):
            IntentGate(
                organ,
                labels=("statement", "question"),
                actionable_labels=frozenset({"statement", "command"}),
            )

    def test_storable_labels_must_be_subset_of_labels(self):
        organ = StubExtractionOrgan({})
        with pytest.raises(ValueError, match="storable_labels"):
            IntentGate(
                organ,
                labels=("statement", "question"),
                storable_labels=frozenset({"statement", "command"}),
            )

    def test_empty_labels_rejected(self):
        organ = StubExtractionOrgan({})
        with pytest.raises(ValueError, match="at least one label"):
            IntentGate(organ, labels=())


class TestActionableLabelsExportedConstant:
    def test_actionable_set_contains_expected_labels(self):
        # The substrate's contract: only statements (storable) and questions
        # (queryable) are actionable. Any addition here is a deliberate API
        # change that should be visible in the diff.
        assert ACTIONABLE_LABELS == frozenset({"statement", "question"})
