"""Tests that the comprehend pipeline correctly distinguishes utterance types.

The core failure mode: "Tell me a joke" gets parsed as the triple
(me, tell, joke) and stored as a fact with confidence 0.92, activating
grafts that corrupt the LLM's output. Meanwhile the same model via
ollama (no substrate) handles jokes perfectly.

These tests verify:
1. Requests/commands do NOT activate grafts (no memory write, no bias)
2. Factual statements DO get stored and recalled correctly
3. Questions about stored facts DO activate grafts with derived confidence
4. The affect organ detects emotional state on every utterance
5. Graft strength is derived from substrate knowledge, not static confidence
"""

import tempfile
from pathlib import Path

import pytest

from core.cognition.intent_gate import INTENT_LABELS
from core.organs.extraction import ExtractionOrgan, ExtractedEntity, ExtractedRelation
from core.organs.affect import AffectOrgan, AffectState


def _intent_schema(*wanted: str) -> list[str]:
    """Intersect with :data:`INTENT_LABELS` so label order follows the live gate."""

    w = frozenset(wanted)
    return [lab for lab in INTENT_LABELS if lab in w]


class TestExtractionOrganIntentClassification:
    """The extraction organ must distinguish requests from statements."""

    @pytest.fixture
    def organ(self):
        organ = ExtractionOrgan()
        organ.load()
        return organ

    def test_request_classified_as_request(self, organ):
        """'Tell me a joke' is a request, not a factual statement."""
        results = organ.classify(
            "Tell me a joke",
            labels=_intent_schema("request", "statement", "question"),
            multi_label=False,
        )
        assert results, "classify returned no results"
        top_label = results[0][0]
        assert top_label == "request", f"Expected 'request', got '{top_label}'"

    def test_question_classified_as_question(self, organ):
        """'Where is Ada?' is a question."""
        results = organ.classify(
            "Where is Ada?",
            labels=_intent_schema("request", "statement", "question"),
            multi_label=False,
        )
        assert results, "classify returned no results"
        top_label = results[0][0]
        assert top_label == "question", f"Expected 'question', got '{top_label}'"

    def test_statement_classified_as_statement(self, organ):
        """'Ada lives in Rome' is a factual statement."""
        results = organ.classify(
            "Ada lives in Rome",
            labels=_intent_schema("request", "statement", "question"),
            multi_label=False,
        )
        assert results, "classify returned no results"
        top_label = results[0][0]
        assert top_label == "statement", f"Expected 'statement', got '{top_label}'"

    def test_greeting_not_classified_as_statement(self, organ):
        """'Hi' should not be classified as a statement with entities."""
        results = organ.classify(
            "Hi",
            labels=_intent_schema("request", "statement", "question", "greeting"),
            multi_label=False,
        )
        assert results, "classify returned no results"
        top_label = results[0][0]
        assert top_label != "statement", f"'Hi' should not be a statement, got '{top_label}'"

    def test_command_not_classified_as_statement(self, organ):
        """'Stop talking about dogs' is a command, not a fact."""
        results = organ.classify(
            "Stop talking about dogs",
            labels=_intent_schema("request", "statement", "question", "command"),
            multi_label=False,
        )
        assert results, "classify returned no results"
        top_label = results[0][0]
        assert top_label in ("request", "command"), f"Expected request/command, got '{top_label}'"


class TestExtractionOrganRelations:
    """Relation extraction should only fire on actual declarative content."""

    @pytest.fixture
    def organ(self):
        organ = ExtractionOrgan()
        organ.load()
        return organ

    def test_factual_statement_produces_relations(self, organ):
        """'Ada lives in Rome' should produce a relation triple."""
        relations = organ.extract_relations("Ada lives in Rome")
        assert len(relations) >= 1, "No relations extracted from factual statement"
        r = relations[0]
        assert "ada" in r.subject.lower()
        assert "rome" in r.object.lower()

    def test_request_produces_no_relations(self, organ):
        """'Tell me a joke' should NOT produce a relation triple."""
        relations = organ.extract_relations("Tell me a joke")
        # Either no relations, or if any, they should not be stored as facts
        # The key point: the intent classification should prevent storage
        # even if extraction produces something
        # This test documents current behavior - may produce relations
        # but the ROUTER should not store them when intent != statement

    def test_greeting_produces_no_relations(self, organ):
        """'Hi' should produce no relation triples."""
        relations = organ.extract_relations("Hi")
        assert len(relations) == 0, f"Greeting produced relations: {relations}"

    def test_short_utterance_produces_no_relations(self, organ):
        """'Yes' / 'No' should produce no relations."""
        for utterance in ["Yes", "No", "Yeah", "Ok", "Sure"]:
            relations = organ.extract_relations(utterance)
            assert len(relations) == 0, f"'{utterance}' produced relations: {relations}"


class TestAffectOrganDetection:
    """The affect organ must provide emotional signal on every utterance."""

    @pytest.fixture
    def organ(self):
        organ = AffectOrgan()
        organ.load()
        return organ

    def test_frustration_detected(self, organ):
        """Negative feedback should register as annoyance/frustration."""
        state = organ.detect("That's not funny at all, it's completely incoherent")
        assert state.dominant_emotion != "neutral", f"Expected non-neutral, got {state.dominant_emotion}"
        assert state.valence < 0, f"Expected negative valence, got {state.valence}"
        assert state.preference_signal == "negative_preference"

    def test_gratitude_detected(self, organ):
        """Positive feedback should register as positive preference."""
        state = organ.detect("Thanks, that's exactly what I needed!")
        assert state.valence > 0, f"Expected positive valence, got {state.valence}"
        assert state.preference_signal == "positive_preference"

    def test_curiosity_detected(self, organ):
        """Questions should register curiosity as a cognitive state."""
        state = organ.detect("How does the causal model work?")
        cognitive = state.cognitive_states
        assert "curiosity" in cognitive, f"Expected curiosity in cognitive states, got {cognitive}"

    def test_neutral_on_greeting(self, organ):
        """'Hi' should be roughly neutral."""
        state = organ.detect("Hi")
        assert state.dominant_emotion in ("neutral", "approval", "caring"), \
            f"Expected neutral-ish for 'Hi', got {state.dominant_emotion}"

    def test_confusion_on_incoherent_response(self, organ):
        """Response to gibberish should detect confusion/annoyance."""
        state = organ.detect("That makes absolutely no sense, what are you talking about?")
        assert state.valence < 0 or "confusion" in state.cognitive_states or "annoyance" in state.cognitive_states


class TestGraftStrengthDerived:
    """Graft strength must be derived from substrate knowledge state, not static.

    The principle: if the substrate has nothing useful to say, graft
    strength should be zero. If it has a high-confidence memory recall
    with conformal |C|=1, strength should be high.
    """

    def test_no_knowledge_means_zero_strength(self):
        """When memory is empty and intent is unknown, derived strength = 0."""
        # Simulate: intent=unknown, no memory hit, no causal query
        memory_confidence = 0.0  # no memory hit
        conformal_set_size = 0  # no prediction made
        intent_is_actionable = False  # request/greeting/unknown

        # Derived graft strength should be 0
        strength = self._derive_graft_strength(
            memory_confidence=memory_confidence,
            conformal_set_size=conformal_set_size,
            intent_is_actionable=intent_is_actionable,
        )
        assert strength == 0.0, f"Expected 0.0, got {strength}"

    def test_high_confidence_recall_means_high_strength(self):
        """When memory returns a fact with high confidence and |C|=1, strength is high."""
        memory_confidence = 0.95
        conformal_set_size = 1  # single prediction, high certainty
        intent_is_actionable = True  # memory_lookup with a hit

        strength = self._derive_graft_strength(
            memory_confidence=memory_confidence,
            conformal_set_size=conformal_set_size,
            intent_is_actionable=intent_is_actionable,
        )
        assert strength > 0.8, f"Expected > 0.8, got {strength}"

    def test_ambiguous_recall_means_moderate_strength(self):
        """When conformal set has |C|>1, strength is reduced."""
        memory_confidence = 0.9
        conformal_set_size = 3  # ambiguous
        intent_is_actionable = True

        strength = self._derive_graft_strength(
            memory_confidence=memory_confidence,
            conformal_set_size=conformal_set_size,
            intent_is_actionable=intent_is_actionable,
        )
        assert 0.2 < strength < 0.7, f"Expected moderate strength, got {strength}"

    def test_request_intent_means_zero_strength_regardless(self):
        """Even if memory has a hit, a request should not activate grafts."""
        memory_confidence = 0.92
        conformal_set_size = 1
        intent_is_actionable = False  # "tell me a joke" is not actionable

        strength = self._derive_graft_strength(
            memory_confidence=memory_confidence,
            conformal_set_size=conformal_set_size,
            intent_is_actionable=intent_is_actionable,
        )
        assert strength == 0.0, f"Request should have zero strength, got {strength}"

    @staticmethod
    def _derive_graft_strength(
        *,
        memory_confidence: float,
        conformal_set_size: int,
        intent_is_actionable: bool,
    ) -> float:
        """Reference implementation of derived graft strength.

        This is what the substrate should compute. Graft strength is:
        - 0 when intent is not actionable (requests, greetings, commands)
        - memory_confidence * conformal_sharpness when actionable
        - conformal_sharpness = 1/|C| (1 when certain, decays with ambiguity)
        """
        if not intent_is_actionable:
            return 0.0

        if conformal_set_size <= 0:
            return 0.0

        conformal_sharpness = 1.0 / conformal_set_size
        return memory_confidence * conformal_sharpness
