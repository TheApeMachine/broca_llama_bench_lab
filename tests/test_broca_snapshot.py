from __future__ import annotations

import types
from pathlib import Path

import pytest

import core.broca as broca_mod
from core.broca import BrocaMind, CognitiveFrame
from core.event_bus import EventBus

from conftest import make_stub_llm_pair


class _FakeHost:
    cfg = types.SimpleNamespace(d_model=8)

    def __init__(self):
        self.grafts: list = []
        self.llm, self._stub_tokenizer = make_stub_llm_pair()

    def add_graft(self, slot, graft):
        self.grafts.append((slot, graft))

    def parameters(self):
        import torch

        yield torch.zeros(1)


class _FakeTok:
    def __init__(self, inner):
        self.inner = inner


@pytest.fixture
def fake_host_loader(monkeypatch: pytest.MonkeyPatch):
    def _make() -> _FakeHost:
        host = _FakeHost()
        tok = _FakeTok(host._stub_tokenizer)
        monkeypatch.setattr(broca_mod, "load_llama_broca_host", lambda *a, **k: (host, tok))
        return host

    return _make


def _new_mind(tmp_path: Path) -> BrocaMind:
    return BrocaMind(seed=0, db_path=tmp_path / "snap.sqlite", namespace="snap")


def test_snapshot_has_expected_top_level_keys(tmp_path: Path, fake_host_loader):
    fake_host_loader()
    mind = _new_mind(tmp_path)
    snap = mind.snapshot()
    for key in (
        "ts",
        "model",
        "memory",
        "journal",
        "workspace",
        "background",
        "self_improve",
        "substrate",
        "preferences",
    ):
        assert key in snap, f"missing key {key!r} in snapshot: {list(snap)}"

    assert snap["model"]["id"]
    assert snap["model"]["namespace"] == "snap"
    assert snap["memory"]["count"] == 0
    assert snap["journal"]["count"] == 0
    assert snap["workspace"]["frames_total"] == 0
    assert snap["background"]["running"] is False
    assert snap["self_improve"]["enabled"] is False
    # The hawkes channels list grows with usage; fresh mind starts at 0.
    assert isinstance(snap["substrate"]["hawkes_channels"], int)


def test_snapshot_reflects_workspace_publish(tmp_path: Path, fake_host_loader):
    fake_host_loader()
    mind = _new_mind(tmp_path)
    # Publish a frame directly to the in-memory workspace (no SQLite involved)
    # so we can verify the snapshot picks it up without going through the
    # journal-write path.
    frame = CognitiveFrame(intent="memory_lookup", subject="ada", answer="rome", confidence=0.8)
    mind.workspace.publish(frame)
    snap = mind.snapshot()
    assert snap["workspace"]["frames_total"] >= 1
    latest = snap["workspace"]["latest_frame"]
    assert latest is not None
    assert latest["intent"] == "memory_lookup"
    assert latest["subject"] == "ada"


def test_consolidate_once_publishes_event(tmp_path: Path, fake_host_loader):
    fake_host_loader()
    mind = _new_mind(tmp_path)
    bus = EventBus()
    mind.event_bus = bus
    sub = bus.subscribe("*")
    mind.consolidate_once()
    events = bus.drain(sub)
    topics = [e.topic for e in events]
    assert "consolidation" in topics
    consolidation_ev = next(e for e in events if e.topic == "consolidation")
    assert "reflections" in consolidation_ev.payload


def test_snapshot_includes_last_chat_meta_when_set(tmp_path: Path, fake_host_loader):
    fake_host_loader()
    mind = _new_mind(tmp_path)
    mind._last_chat_meta = {
        "intent": "memory_lookup",
        "confidence": 0.9,
        "eff_temperature": 0.4,
        "bias_token_count": 3,
        "bias_top": [{"token": "rome", "bias": 1.0}],
        "has_broca_features": True,
    }
    snap = mind.snapshot()
    assert snap["last_chat"] is not None
    assert snap["last_chat"]["intent"] == "memory_lookup"
    assert snap["last_chat"]["bias_token_count"] == 3
