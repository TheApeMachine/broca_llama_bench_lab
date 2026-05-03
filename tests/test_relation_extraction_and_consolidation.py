from __future__ import annotations

import types
import uuid
from pathlib import Path

import pytest

import core.cognition.substrate as substrate_mod
from core.cli import build_substrate_controller
from core.memory import ClaimTrust, SymbolicMemory

from conftest import FakeHost, FakeTokenizer, make_stub_llm_pair


def _symbol(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


@pytest.fixture
def fake_host_loader(monkeypatch: pytest.MonkeyPatch):
    def _make() -> FakeHost:
        host = FakeHost()
        tokenizer = FakeTokenizer(host._stub_tokenizer)
        monkeypatch.setattr(substrate_mod, "load_llama_broca_host", lambda *args, **kwargs: (host, tokenizer))
        return host

    return _make


def test_claim_trust_weight_decays_with_prediction_gap():
    no_gap = {"evidence": {}}
    low_gap = {"evidence": {"prediction_gap": 0.1}}
    high_gap = {"evidence": {"prediction_gap": 5.0}}

    assert ClaimTrust.weight(no_gap) == 1.0
    assert 0.5 < ClaimTrust.weight(low_gap) < 1.0
    assert ClaimTrust.weight(high_gap) < 0.25
    assert ClaimTrust.weight(low_gap) > ClaimTrust.weight(high_gap)


def test_consolidation_resists_high_surprise_repeated_claims(tmp_path: Path):
    """Three high-surprise (poison) challengers must NOT outweigh one trusted accepted claim."""

    db = tmp_path / "poisoning.sqlite"
    mem = SymbolicMemory(db, namespace="poison")
    subject = _symbol("subject")
    truth = _symbol("truth")
    poison = _symbol("poison")

    mem.observe_claim(subject, "is in", truth, confidence=1.0, evidence={"prediction_gap": 0.0})
    for _ in range(3):
        mem.record_claim(
            subject,
            "is in",
            poison,
            confidence=1.0,
            status="conflict",
            evidence={"prediction_gap": 9.0, "source": "adversarial"},
        )

    reflections = mem.consolidate_claims_once()
    revisions = [r for r in reflections if r.get("kind") == "belief_revision"]

    assert not revisions, (
        "high-surprise claims should not flip a belief on count alone; got revisions="
        f"{revisions}"
    )
    current = mem.get(subject, "is in")
    assert current is not None and current[0] == truth.lower()


def test_consolidation_still_revises_when_challengers_have_low_surprise(tmp_path: Path):
    """Sanity check: low-surprise challengers should still drive belief revision."""

    db = tmp_path / "low_surprise.sqlite"
    mem = SymbolicMemory(db, namespace="low_surprise")
    subject = _symbol("subject")
    truth = _symbol("truth")
    challenger = _symbol("challenger")

    mem.observe_claim(subject, "is in", truth, confidence=1.0)
    for _ in range(2):
        mem.record_claim(
            subject,
            "is in",
            challenger,
            confidence=1.0,
            status="conflict",
            evidence={"prediction_gap": 0.0},
        )

    reflections = mem.consolidate_claims_once()
    revisions = [r for r in reflections if r.get("kind") == "belief_revision"]

    assert revisions, "two low-surprise corroborating claims with margin should trigger revision"
    current = mem.get(subject, "is in")
    assert current is not None and current[0] == challenger.lower()


def test_runtime_router_uses_encoder_relation_extractor(tmp_path: Path, fake_host_loader):
    """Chat pipeline must route through the encoder-backed extractor.

    The substrate has only one relation extractor (the encoder-backed one); the
    LLM-based path was removed because it parsed imperatives like "Tell me a
    joke" into the triple ``(me, tell, joke)`` and shoved them into memory.
    """

    from core.cognition.encoder_relation_extractor import EncoderRelationExtractor

    fake_host_loader()
    mind = build_substrate_controller(seed=0, db_path=tmp_path / "router_extractor.sqlite", namespace="ext", device="cpu", hf_token=False)

    assert isinstance(mind.router.extractor, EncoderRelationExtractor)
