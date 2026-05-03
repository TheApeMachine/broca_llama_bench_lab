from __future__ import annotations

import math
import types
from pathlib import Path
from typing import Any

import pytest

import core.cognition.substrate as substrate_mod
from core.cli import build_substrate_controller
from core.affect.trace import PersistentAffectTrace
from core.encoders.affect import AffectEncoder, AffectState, EmotionScore

from conftest import FakeHost, FakeTokenizer, make_stub_llm_pair, stub_substrate_encoders


def _emotion(label: str, score: float) -> EmotionScore:
    return EmotionScore(label=label, score=float(score))


def _state(
    *,
    dominant: str,
    score: float,
    valence: float,
    arousal: float,
    confidences: list[tuple[str, float]],
) -> AffectState:
    items = sorted(
        (_emotion(label, value) for label, value in confidences),
        key=lambda item: item.score,
        reverse=True,
    )
    total = sum(value for _label, value in confidences)
    entropy = -sum((value / total) * math.log(value / total) for _label, value in confidences if value > 0.0)
    certainty = 1.0 - entropy / math.log(len(confidences))
    return AffectState(
        dominant_emotion=dominant,
        dominant_score=score,
        confidences=items,
        emotions=items,
        valence=valence,
        arousal=arousal,
        entropy=entropy,
        certainty=certainty,
    )


class SequenceAffectEncoder:
    def __init__(self, states: list[AffectState]) -> None:
        self.states = list(states)
        self.calls: list[str] = []

    def detect(self, text: str, *, threshold: float | None = None) -> AffectState:
        _ = threshold
        self.calls.append(text)
        if not self.states:
            raise RuntimeError("SequenceAffectEncoder exhausted")
        return self.states.pop(0)


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


def test_affect_encoder_preserves_full_confidence_distribution() -> None:
    encoder = AffectEncoder(use_onnx=False, threshold=0.15)
    encoder._loaded = True
    encoder._pipeline = lambda text: [
        {"label": "anger", "score": 0.499},
        {"label": "annoyance", "score": 0.348},
        {"label": "disapproval", "score": 0.273},
        {"label": "neutral", "score": 0.039},
    ]

    state = encoder.detect("That is not acceptable")

    assert [item.label for item in state.confidences] == [
        "anger",
        "annoyance",
        "disapproval",
        "neutral",
    ]
    assert [item.label for item in state.emotions] == ["anger", "annoyance", "disapproval"]
    assert state.distribution()["neutral"] == pytest.approx(0.039)
    assert 0.0 <= state.preference_strength <= 1.0
    assert state.certainty == pytest.approx(
        AffectEncoder._distribution_certainty(state.confidences, entropy=state.entropy)
    )


def test_affect_trace_persists_distribution_and_alignment(tmp_path: Path) -> None:
    trace = PersistentAffectTrace(tmp_path / "affect.sqlite", namespace="ut")
    user = _state(
        dominant="anger",
        score=0.7,
        valence=-0.8,
        arousal=0.7,
        confidences=[("anger", 0.7), ("annoyance", 0.2), ("neutral", 0.1)],
    )
    assistant = _state(
        dominant="neutral",
        score=0.6,
        valence=-0.2,
        arousal=0.3,
        confidences=[("anger", 0.1), ("annoyance", 0.2), ("neutral", 0.7)],
    )

    user_id = trace.record(role="user", text="I am angry", affect=user, journal_id=3)
    alignment = trace.alignment(user, assistant)
    assistant_id = trace.record(
        role="assistant",
        text="I hear you.",
        affect=assistant,
        response_to_id=user_id,
        alignment=alignment,
    )

    assert assistant_id > user_id
    summary = trace.summary()
    assert summary["user_count"] == 1
    assert summary["assistant_count"] == 1
    assert summary["paired_count"] == 1
    assert summary["mean_alignment"] == pytest.approx(alignment["alignment"])
    assert summary["recent"][0]["distribution"]["anger"] == pytest.approx(0.7)


def test_chat_reply_records_user_and_assistant_affect_alignment(
    tmp_path: Path,
    fake_host_loader,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_host_loader()
    mind = build_substrate_controller(
        seed=0,
        db_path=tmp_path / "chat_affect.sqlite",
        namespace="chat_affect",
        device="cpu",
        hf_token=False,
    )
    stub_substrate_encoders(
        mind,
        intent_responses={"please help": [("request", 0.98)]},
    )
    user = _state(
        dominant="anger",
        score=0.8,
        valence=-0.9,
        arousal=0.8,
        confidences=[("anger", 0.8), ("annoyance", 0.1), ("neutral", 0.1)],
    )
    assistant = _state(
        dominant="neutral",
        score=0.7,
        valence=0.1,
        arousal=0.2,
        confidences=[("anger", 0.05), ("annoyance", 0.1), ("neutral", 0.85)],
    )
    mind.affect_encoder = SequenceAffectEncoder([user, assistant])  # type: ignore[assignment]
    from core.generation import ChatDecoder

    monkeypatch.setattr(
        ChatDecoder,
        "stream",
        lambda self, *args, **kwargs: ("I understand and will help.", [1], 1.0),
    )

    frame, text = mind.chat_reply([{"role": "user", "content": "Please help"}])

    assert frame.intent == "unknown"
    assert text == "I understand and will help."
    summary = mind.affect_trace.summary()
    assert summary["user_count"] == 1
    assert summary["assistant_count"] == 1
    assert summary["paired_count"] == 1
    assert "affect_alignment" in mind.session.last_chat_meta
    assert mind.session.last_chat_meta["assistant_affect"]["confidences"][0]["label"] == "neutral"
