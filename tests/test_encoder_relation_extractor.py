"""Tests for the encoder-backed relation extractor.

The extractor sits inside :class:`CognitiveRouter` and decides whether the
substrate writes a triple to memory. The original failure mode — the LLM
extractor parsing "Tell me a joke" as ``(me, tell, joke)`` and storing it
at confidence 0.92 — must be impossible by *construction*: the intent gate
short-circuits non-storable utterances before the extractor ever asks
GLiNER for a relation.

These tests use stubs for the intent gate and extraction encoder so the
extractor's *policy* is what is under test, not GLiNER's accuracy.
"""

from __future__ import annotations

from typing import Sequence

import pytest

from core.cognition.intent_gate import INTENT_LABELS, IntentGate
from core.cognition.encoder_relation_extractor import EncoderRelationExtractor
from core.encoders.extraction import ExtractedRelation


class StubExtractionEncoder:
    """Stub that returns canned entities/relations and intent classifications."""

    def __init__(
        self,
        *,
        intent_responses: dict[str, list[tuple[str, float]]] | None = None,
        relation_responses: dict[str, list[ExtractedRelation]] | None = None,
    ):
        self._intent = intent_responses or {}
        self._relations = relation_responses or {}
        self.classify_calls: list[str] = []
        self.relation_calls: list[str] = []
        self.identity_calls: list[str] = []

    def extract_identity_relations(self, text: str) -> list[ExtractedRelation]:
        self.identity_calls.append(text)
        return []

    def classify(
        self,
        text: str,
        *,
        labels: Sequence[str],
        multi_label: bool = True,
        threshold: float = 0.0,
    ) -> list[tuple[str, float]]:
        self.classify_calls.append(text)
        for fragment, scores in self._intent.items():
            if fragment in text.lower():
                return list(scores)
        return [(labels[0], 0.0)]

    def extract_relations(
        self,
        text: str,
        *,
        entity_labels: Sequence[str] | None = None,
        relation_labels: Sequence[str] | None = None,
    ) -> list[ExtractedRelation]:
        _ = entity_labels, relation_labels
        self.relation_calls.append(text)
        for fragment, rels in self._relations.items():
            if fragment in text.lower():
                return list(rels)
        return []


class StubSemanticCascade:
    def __init__(self, extraction: StubExtractionEncoder):
        self.extraction = extraction

    def intent_scores(self, text: str) -> dict:
        ranked = self.extraction.classify(text, labels=INTENT_LABELS, multi_label=False, threshold=0.0)
        if not ranked:
            return {
                "label": "",
                "confidence": 0.0,
                "scores": {},
                "allows_storage": False,
                "evidence": {},
            }
        scores = {label: 0.0 for label in INTENT_LABELS}
        for label, score in ranked:
            scores[label] = float(score)
        top_label, top_score = ranked[0]
        return {
            "label": top_label,
            "confidence": float(top_score),
            "scores": scores,
            "allows_storage": top_label == "statement",
            "evidence": {"stub": True},
        }


def _build(
    *,
    intent_responses: dict[str, list[tuple[str, float]]] | None = None,
    relation_responses: dict[str, list[ExtractedRelation]] | None = None,
) -> tuple[EncoderRelationExtractor, StubExtractionEncoder]:
    extraction = StubExtractionEncoder(
        intent_responses=intent_responses,
        relation_responses=relation_responses,
    )
    gate = IntentGate(StubSemanticCascade(extraction))
    extractor = EncoderRelationExtractor(intent_gate=gate, extraction=extraction)
    return extractor, extraction


class TestNonActionableUtterancesNeverProduceClaims:
    """The original bug: requests stored as triples. Must be impossible now."""

    def test_request_returns_none(self):
        ext, extraction = _build(
            intent_responses={
                "tell me a joke": [("request", 0.95), ("statement", 0.03)],
            },
            relation_responses={
                "tell me a joke": [
                    ExtractedRelation(
                        subject="me",
                        predicate="tell",
                        object="joke",
                        confidence=0.92,
                    )
                ],
            },
        )
        result = ext.extract_claim("Tell me a joke", ["tell", "me", "a", "joke"])
        assert result is None
        # And the relation extractor must NEVER have been called: the gate
        # must short-circuit *before* GLiNER is consulted.
        assert extraction.relation_calls == []

    def test_greeting_returns_none_and_does_not_invoke_extractor(self):
        ext, extraction = _build(
            intent_responses={"hi": [("greeting", 0.9), ("statement", 0.05)]},
            relation_responses={"hi": []},
        )
        result = ext.extract_claim("Hi", ["hi"])
        assert result is None
        assert extraction.relation_calls == []

    def test_question_returns_none(self):
        ext, extraction = _build(
            intent_responses={"where is ada": [("question", 0.93), ("statement", 0.05)]},
        )
        result = ext.extract_claim("Where is Ada?", ["where", "is", "ada", "?"])
        assert result is None
        assert extraction.relation_calls == []

    def test_command_returns_none(self):
        ext, _extraction = _build(
            intent_responses={
                "stop talking": [("command", 0.88), ("request", 0.10), ("statement", 0.02)],
            },
        )
        result = ext.extract_claim("Stop talking about dogs", ["stop", "talking", "about", "dogs"])
        assert result is None


class TestStatementsProduceClaims:
    def test_statement_with_relation_yields_claim(self):
        ext, extraction = _build(
            intent_responses={
                "ada lives in rome": [("statement", 0.93), ("question", 0.04)],
            },
            relation_responses={
                "ada lives in rome": [
                    ExtractedRelation(
                        subject="Ada",
                        predicate="lives_in",
                        object="Rome",
                        confidence=0.85,
                        subject_label="person",
                        object_label="location",
                    )
                ],
            },
        )
        claim = ext.extract_claim("Ada lives in Rome", ["ada", "lives", "in", "rome"])
        assert claim is not None
        assert claim.subject == "ada"
        assert claim.predicate == "lives_in"
        assert claim.obj == "rome"
        assert extraction.relation_calls == ["Ada lives in Rome"]

    def test_statement_with_no_relation_returns_none(self):
        """If GLiNER finds no relation, the substrate honestly stores nothing."""

        ext, extraction = _build(
            intent_responses={"hello world": [("statement", 0.6), ("greeting", 0.3)]},
            relation_responses={"hello world": []},
        )
        result = ext.extract_claim("Hello world", ["hello", "world"])
        assert result is None
        # The extractor *was* called — the gate let this through, but no
        # relation was found, so no triple is fabricated.
        assert extraction.relation_calls == ["Hello world"]


class TestClaimConfidenceComposesIntentAndExtractor:
    """Both the intent gate and GLiNER must vouch for the claim."""

    def test_confidence_is_intent_times_extractor(self):
        ext, _extraction = _build(
            intent_responses={"ada lives in rome": [("statement", 0.8)]},
            relation_responses={
                "ada lives in rome": [
                    ExtractedRelation(
                        subject="Ada",
                        predicate="lives_in",
                        object="Rome",
                        confidence=0.5,
                    )
                ],
            },
        )
        claim = ext.extract_claim("Ada lives in Rome", ["ada", "lives", "in", "rome"])
        assert claim is not None
        assert claim.confidence == pytest.approx(0.8 * 0.5, rel=1e-6)

    def test_low_intent_confidence_drags_claim_confidence_down(self):
        ext, _extraction = _build(
            intent_responses={"ada lives in rome": [("statement", 0.4)]},
            relation_responses={
                "ada lives in rome": [
                    ExtractedRelation(subject="Ada", predicate="lives_in", object="Rome", confidence=0.95),
                ],
            },
        )
        claim = ext.extract_claim("Ada lives in Rome", ["ada", "lives", "in", "rome"])
        assert claim is not None
        assert claim.confidence == pytest.approx(0.4 * 0.95, rel=1e-6)


class TestEvidenceIncludesIntentTrace:
    """The frame must record which gate decision unlocked the claim."""

    def test_evidence_records_intent_label_and_scores(self):
        ext, _extraction = _build(
            intent_responses={
                "ada lives in rome": [
                    ("statement", 0.88), ("question", 0.07), ("request", 0.05)
                ],
            },
            relation_responses={
                "ada lives in rome": [
                    ExtractedRelation(subject="Ada", predicate="lives_in", object="Rome", confidence=0.9),
                ],
            },
        )
        claim = ext.extract_claim("Ada lives in Rome", ["ada", "lives", "in", "rome"])
        assert claim is not None
        ev = claim.evidence
        assert ev["intent_label"] == "statement"
        assert ev["intent_confidence"] == pytest.approx(0.88, rel=1e-6)
        assert "intent_scores" in ev
        assert ev["parser"] == "encoder_relation_extractor"

    def test_alternative_relations_recorded(self):
        ext, _extraction = _build(
            intent_responses={"alpha is beta": [("statement", 0.9)]},
            relation_responses={
                "alpha is beta": [
                    ExtractedRelation(subject="Alpha", predicate="is_a", object="Beta", confidence=0.8),
                    ExtractedRelation(subject="Alpha", predicate="related_to", object="Beta", confidence=0.6),
                ],
            },
        )
        claim = ext.extract_claim("Alpha is Beta", ["alpha", "is", "beta"])
        assert claim is not None
        # Highest confidence wins; the other is recorded as an alternative.
        assert claim.predicate == "is_a"
        alts = claim.evidence["alternative_relations"]
        assert len(alts) == 1
        assert alts[0]["predicate"] == "related_to"

    def test_prefilled_intent_skips_second_classify(self):
        """Router passes comprehend's UtteranceIntent so GLiNER classifies intent once."""

        ext, extraction = _build(
            intent_responses={"ada lives in rome": [("statement", 0.93)]},
            relation_responses={
                "ada lives in rome": [
                    ExtractedRelation(subject="Ada", predicate="lives_in", object="Rome", confidence=0.9),
                ],
            },
        )
        gate = IntentGate(StubSemanticCascade(extraction))
        cached = gate.classify("Ada lives in Rome")
        extraction.classify_calls.clear()
        claim = ext.extract_claim(
            "Ada lives in Rome",
            ["ada", "lives", "in", "rome"],
            utterance_intent=cached,
        )
        assert claim is not None
        assert extraction.classify_calls == [], "passed UtteranceIntent must not invoke extraction.classify again"
        assert extraction.relation_calls == ["Ada lives in Rome"]
