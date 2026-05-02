"""Tests for the IntentGate.

The gate is the substrate's first defense against the original failure mode:
"Tell me a joke" — a request — being parsed by the relation extractor as the
declarative triple ``(me, tell, joke)`` and stored in semantic memory at
confidence 0.92. The gate must cleanly partition utterances into actionable
(statement / question) and non-actionable (request / greeting / command /
feedback / acknowledgment) categories so the relation extractor downstream
never even sees the non-actionable ones.

These tests use a stub semantic cascade rather than real GLiClass/GLiNER2
weights so the suite stays fast and the gate's *policy* — not model accuracy —
is what is under test.
"""

from __future__ import annotations

import pytest

from core.cognition.intent_gate import (
    ACTIONABLE_LABELS,
    INTENT_LABELS,
    IntentGate,
    UtteranceIntent,
)


class StubSemanticCascade:
    """Pretends to be :class:`SemanticCascade` for the gate's purposes.

    The gate consumes already-collapsed semantic cascade output. These tests
    verify gate policy around that output, not model accuracy.
    """

    def __init__(self, responses: dict[str, list[tuple[str, float]]]):
        self._responses = responses
        self.calls: list[str] = []

    def intent_scores(self, text: str) -> dict:
        self.calls.append(text)
        for fragment, scores in self._responses.items():
            if fragment in text.lower():
                return self._payload(scores)
        return self._payload([("statement", 0.0)])

    @staticmethod
    def _payload(scores_in_order: list[tuple[str, float]]) -> dict:
        if not scores_in_order:
            return {
                "label": "",
                "confidence": 0.0,
                "scores": {},
                "allows_storage": False,
                "evidence": {},
            }
        scores = {label: 0.0 for label in INTENT_LABELS}
        for label, score in scores_in_order:
            scores[label] = float(score)
        label, confidence = scores_in_order[0]
        return {
            "label": label,
            "confidence": float(confidence),
            "scores": scores,
            "allows_storage": label == "statement",
            "evidence": {"stub": True},
        }


def _gate(responses: dict[str, list[tuple[str, float]]]) -> IntentGate:
    return IntentGate(StubSemanticCascade(responses))


class TestIntentClassification:
    def test_request_is_not_actionable(self):
        gate = _gate({
            "tell me a joke": [("request", 0.91), ("statement", 0.04), ("question", 0.05)],
        })
        intent = gate.classify("Tell me a joke")
        assert intent.label == "request"
        assert intent.is_actionable is False
        assert intent.allows_storage is False

    def test_request_invokes_semantic_cascade(self):
        cascade = StubSemanticCascade({
            "tell me a joke": [("request", 0.91), ("statement", 0.04), ("question", 0.05)],
        })
        gate = IntentGate(cascade)
        intent = gate.classify("Tell me a joke")
        assert intent.label == "request"
        assert cascade.calls == ["Tell me a joke"]

    def test_declarative_text_still_invokes_extraction_encoder(self):
        cascade = StubSemanticCascade(
            {"ada lives in rome": [("statement", 0.88), ("question", 0.07)]}
        )
        gate = IntentGate(cascade)
        intent = gate.classify("Ada lives in Rome")
        assert intent.label == "statement"
        assert cascade.calls == ["Ada lives in Rome"]

    def test_first_person_identity_is_model_backed_statement(self):
        gate = _gate({"i am the magnificent": [("statement", 1.0), ("greeting", 0.2)]})
        intent = gate.classify("I am the Magnificent")
        assert intent.label == "statement"
        assert intent.is_actionable is True
        assert intent.allows_storage is True

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

    def test_classifier_returning_zero_confidence_keeps_zero_confidence(self):
        gate = _gate({})  # no fragment matches -> stub returns one (label, 0.0)
        intent = gate.classify("totally unmatched text")
        assert intent.label in INTENT_LABELS
        assert intent.confidence == 0.0

    def test_classifier_returning_no_valid_label_raises(self):
        gate = IntentGate(StubSemanticCascade({"totally unmatched text": []}))
        with pytest.raises(RuntimeError, match="unknown top label"):
            gate.classify("totally unmatched text")


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
        cascade = StubSemanticCascade({})
        with pytest.raises(ValueError, match="actionable_labels"):
            IntentGate(
                cascade,
                labels=("statement", "question"),
                actionable_labels=frozenset({"statement", "command"}),
            )

    def test_storable_labels_must_be_subset_of_labels(self):
        cascade = StubSemanticCascade({})
        with pytest.raises(ValueError, match="storable_labels"):
            IntentGate(
                cascade,
                labels=("statement", "question"),
                storable_labels=frozenset({"statement", "command"}),
            )

    def test_empty_labels_rejected(self):
        cascade = StubSemanticCascade({})
        with pytest.raises(ValueError, match="at least one label"):
            IntentGate(cascade, labels=())


class TestActionableLabelsExportedConstant:
    def test_actionable_set_contains_expected_labels(self):
        # The substrate's contract: only statements (storable) and questions
        # (queryable) are actionable. Any addition here is a deliberate API
        # change that should be visible in the diff.
        assert ACTIONABLE_LABELS == frozenset({"statement", "question"})
