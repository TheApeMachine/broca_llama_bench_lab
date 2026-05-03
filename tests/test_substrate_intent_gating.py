"""Substrate-level integration tests for intent gating + derived strength.

The original failure mode, end to end:

  User says "Tell me a joke" → relation extraction parses it as the triple
  ``(me, tell, joke)`` → ``CognitiveRouter`` picks a memory write candidate
  above the relevance floor → graft activates with confidence=0.92 → the LLM
  produces "memory write me tell joke".

This test asserts the new behavior, end to end:

  User says "Tell me a joke" → :class:`IntentGate` classifies as ``request``
  (non-actionable) → ``comprehend`` short-circuits to ``unknown`` →
  :func:`SubstrateController._derived_target_snr_scale` returns 0.0 →
  no broca features, no logit bias, the LLM speaks freely.

We stub the actual encoder weights so the test stays fast — the *wiring* is
what's under test, not GLiClass or GLiNER2 model accuracy.
"""

from __future__ import annotations

import types
from pathlib import Path
from typing import Sequence

import pytest

import core.cognition.substrate as substrate_mod
from core.cognition.intent_gate import INTENT_LABELS, IntentGate
from core.cognition.encoder_relation_extractor import EncoderRelationExtractor
from core.cli import build_substrate_controller
from core.cognition.substrate import SubstrateController
from core.encoders.affect import AffectState
from core.encoders.extraction import ExtractedRelation

from conftest import FakeHost, FakeTokenizer, make_stub_llm_pair


@pytest.fixture
def fake_host_loader(monkeypatch: pytest.MonkeyPatch):
    def _make() -> FakeHost:
        host = FakeHost()
        tokenizer = FakeTokenizer(host._stub_tokenizer)
        monkeypatch.setattr(
            substrate_mod,
            "load_llama_broca_host",
            lambda *args, **kwargs: (host, tokenizer),
        )
        return host

    return _make


class StubExtractionEncoder:
    """Same shape as :class:`ExtractionEncoder`, returns canned data."""

    def __init__(
        self,
        *,
        intent_responses: dict[str, list[tuple[str, float]]],
        relation_responses: dict[str, list[ExtractedRelation]] | None = None,
    ):
        self._intent = intent_responses
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


class StubAffectEncoder:
    """Returns a canned :class:`AffectState` so substrate doesn't load weights."""

    def __init__(self, state: AffectState):
        self._state = state
        self.calls: list[str] = []

    def detect(self, text: str, *, threshold: float | None = None) -> AffectState:
        _ = threshold
        self.calls.append(text)
        return self._state


class StubSemanticCascade:
    def __init__(self, extraction: StubExtractionEncoder):
        self.extraction = extraction

    def intent_scores(self, text: str) -> dict:
        ranked = self.extraction.classify(text, labels=INTENT_LABELS, multi_label=False, threshold=0.0)
        if not ranked:
            return {"label": "", "confidence": 0.0, "scores": {}, "evidence": {}}
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


class StubBackgroundWorker:
    running = True

    def __init__(self):
        self.notified = False
        self.marked_active = False

    def notify_work(self) -> None:
        self.notified = True

    def mark_user_active(self) -> None:
        self.marked_active = True


def _wire_stubs(
    mind: SubstrateController,
    *,
    intent_responses: dict[str, list[tuple[str, float]]],
    relation_responses: dict[str, list[ExtractedRelation]] | None = None,
    affect: AffectState | None = None,
) -> StubExtractionEncoder:
    """Replace the substrate's encoders with stubs and rebuild dependent wiring."""

    extraction = StubExtractionEncoder(
        intent_responses=intent_responses,
        relation_responses=relation_responses,
    )
    mind.extraction_encoder = extraction  # type: ignore[assignment]
    mind.affect_encoder = StubAffectEncoder(affect or AffectState())  # type: ignore[assignment]
    mind.semantic_cascade = StubSemanticCascade(extraction)  # type: ignore[assignment]
    mind.intent_gate = IntentGate(mind.semantic_cascade)  # type: ignore[arg-type]
    mind.router.extractor = EncoderRelationExtractor(
        intent_gate=mind.intent_gate,
        extraction=extraction,  # type: ignore[arg-type]
    )
    return extraction


def _build_mind(tmp_path: Path) -> SubstrateController:
    return build_substrate_controller(
        seed=0,
        db_path=tmp_path / "intent_gate.sqlite",
        namespace="intent_gate",
        device="cpu",
        hf_token=False,
    )


class TestRequestProducesNoGraftActivation:
    """The headline regression test: a request must not activate grafts."""

    def test_tell_me_a_joke_is_gated_to_unknown(self, tmp_path: Path, fake_host_loader):
        fake_host_loader()
        mind = _build_mind(tmp_path)
        stub = _wire_stubs(
            mind,
            intent_responses={
                "tell me a joke": [
                    ("request", 0.95), ("statement", 0.03), ("greeting", 0.02)
                ],
            },
            relation_responses={
                # Even if GLiNER would say there's a triple, the gate must
                # short-circuit *before* it asks.
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

        frame = mind.comprehend("Tell me a joke")
        assert frame.intent == "unknown"
        assert frame.confidence == 0.0
        # Relation extraction must NEVER have been consulted.
        assert stub.relation_calls == []

    def test_tell_me_a_joke_yields_zero_derived_strength(self, tmp_path: Path, fake_host_loader):
        fake_host_loader()
        mind = _build_mind(tmp_path)
        _wire_stubs(
            mind,
            intent_responses={"tell me a joke": [("request", 0.95)]},
        )
        frame = mind.comprehend("Tell me a joke")
        scale = mind._derived_target_snr_scale(frame)
        assert scale == 0.0

    def test_tell_me_a_joke_writes_nothing_to_memory(self, tmp_path: Path, fake_host_loader):
        fake_host_loader()
        mind = _build_mind(tmp_path)
        _wire_stubs(
            mind,
            intent_responses={"tell me a joke": [("request", 0.95)]},
            relation_responses={
                "tell me a joke": [
                    ExtractedRelation(subject="me", predicate="tell", object="joke", confidence=0.92),
                ],
            },
        )
        before = mind.memory.count()
        mind.comprehend("Tell me a joke")
        after = mind.memory.count()
        assert before == after, "Request must not change semantic memory"

    def test_greeting_is_gated(self, tmp_path: Path, fake_host_loader):
        fake_host_loader()
        mind = _build_mind(tmp_path)
        _wire_stubs(
            mind,
            intent_responses={"hi": [("greeting", 0.91), ("statement", 0.05)]},
        )
        frame = mind.comprehend("Hi")
        assert frame.intent == "unknown"
        assert mind._derived_target_snr_scale(frame) == 0.0


class TestStatementsStillFlowThrough:
    """The gate must not block legitimate declarative content."""

    def test_statement_produces_actionable_frame(self, tmp_path: Path, fake_host_loader):
        fake_host_loader()
        mind = _build_mind(tmp_path)
        _wire_stubs(
            mind,
            intent_responses={"ada lives in rome": [("statement", 0.93)]},
            relation_responses={
                "ada lives in rome": [
                    ExtractedRelation(subject="Ada", predicate="lives_in", object="Rome", confidence=0.85),
                ],
            },
            affect=AffectState(dominant_emotion="neutral", dominant_score=0.6),
        )
        frame = mind.comprehend("Ada lives in Rome")
        assert frame.intent == "memory_ingest_pending"
        assert frame.evidence["deferred_relation_ingest"] is True
        assert mind.deferred_relation_ingest_count() == 1

    def test_statement_writes_to_memory(self, tmp_path: Path, fake_host_loader):
        fake_host_loader()
        mind = _build_mind(tmp_path)
        stub = _wire_stubs(
            mind,
            intent_responses={"ada lives in rome": [("statement", 0.93)]},
            relation_responses={
                "ada lives in rome": [
                    ExtractedRelation(subject="Ada", predicate="lives_in", object="Rome", confidence=0.85),
                ],
            },
        )
        before = mind.memory.count()
        frame = mind.comprehend("Ada lives in Rome")
        assert frame.intent == "memory_ingest_pending"
        assert stub.relation_calls == []
        assert mind.memory.count() == before
        reflections = mind.process_deferred_relation_ingest()
        assert reflections[0]["status"] == "memory_write"
        assert mind.memory.count() > before, "DMN ingest must reach semantic memory"
        assert stub.relation_calls == ["Ada lives in Rome"]

    def test_statement_relation_extraction_is_deferred_when_dmn_online(self, tmp_path: Path, fake_host_loader):
        fake_host_loader()
        mind = _build_mind(tmp_path)
        stub = _wire_stubs(
            mind,
            intent_responses={"ada lives in rome": [("statement", 0.93)]},
            relation_responses={
                "ada lives in rome": [
                    ExtractedRelation(subject="Ada", predicate="lives_in", object="Rome", confidence=0.85),
                ],
            },
        )
        worker = StubBackgroundWorker()
        mind.session.background_worker = worker  # type: ignore[assignment]

        frame = mind.comprehend("Ada lives in Rome")

        assert frame.intent == "memory_ingest_pending"
        assert stub.relation_calls == []
        assert mind.memory.count() == 0
        assert mind.deferred_relation_ingest_count() == 1
        assert worker.notified is True
        assert worker.marked_active is True


class TestPerceptionLeavesEvidenceTrace:
    """Both intent and affect must show up on the frame for downstream use."""

    def test_intent_label_is_recorded_on_unknown_frame(self, tmp_path: Path, fake_host_loader):
        fake_host_loader()
        mind = _build_mind(tmp_path)
        _wire_stubs(
            mind,
            intent_responses={"tell me a joke": [("request", 0.95)]},
        )
        frame = mind.comprehend("Tell me a joke")
        assert frame.evidence["intent_label"] == "request"
        assert frame.evidence["is_actionable"] is False

    def test_affect_summary_is_recorded(self, tmp_path: Path, fake_host_loader):
        fake_host_loader()
        mind = _build_mind(tmp_path)
        _wire_stubs(
            mind,
            intent_responses={"that is amazing": [("statement", 0.7)]},
            relation_responses={"that is amazing": []},
            affect=AffectState(
                dominant_emotion="joy",
                dominant_score=0.9,
                valence=0.8,
                arousal=0.5,
                preference_signal="positive_preference",
                preference_strength=0.9,
            ),
        )
        frame = mind.comprehend("That is amazing")
        affect = frame.evidence.get("affect")
        assert isinstance(affect, dict)
        assert affect["dominant_emotion"] == "joy"
        assert affect["preference_signal"] == "positive_preference"
        assert affect["valence"] == pytest.approx(0.8, rel=1e-6)
