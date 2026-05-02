from __future__ import annotations

import types
import uuid
from pathlib import Path

import pytest

import core.cognition.substrate as substrate_mod
from core.cognition.substrate import (
    SubstrateController,
    LLMRelationExtractor,
    PersistentSemanticMemory,
    _claim_trust_weight,
)

from conftest import make_stub_llm_pair


def _symbol(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _stub_extractor_pair(extractor=None):
    llm, hf_tok = make_stub_llm_pair(extractor)
    host = types.SimpleNamespace(llm=llm)
    tok = types.SimpleNamespace(inner=hf_tok)
    return host, tok


class FakeHost:
    cfg = types.SimpleNamespace(d_model=8)

    def __init__(self):
        self.grafts: list = []
        self.llm, self._stub_tokenizer = make_stub_llm_pair()

    def add_graft(self, slot, graft):
        self.grafts.append((slot, graft))


class FakeTokenizer:
    def __init__(self, stub_inner):
        self.inner = stub_inner


@pytest.fixture
def fake_host_loader(monkeypatch: pytest.MonkeyPatch):
    def _make() -> FakeHost:
        host = FakeHost()
        tokenizer = FakeTokenizer(host._stub_tokenizer)
        monkeypatch.setattr(substrate_mod, "load_llama_broca_host", lambda *args, **kwargs: (host, tokenizer))
        return host

    return _make


def test_llm_extractor_resolves_subordinate_clause_subject_object():
    host, tok = _stub_extractor_pair(lambda _s: ("apple", "fell from", "tree"))
    extractor = LLMRelationExtractor(host, tok)

    utterance = "the apple, which was incredibly red, fell from the tree ."
    parsed = extractor.extract_claim(utterance, utterance.split())

    assert parsed is not None
    assert parsed.subject == "apple"
    assert parsed.predicate == "fell from"
    assert parsed.obj == "tree"
    assert parsed.evidence["parser"] == "llm_relation_extractor"


def test_llm_extractor_returns_none_for_questions():
    host, tok = _stub_extractor_pair(lambda _s: ("x", "y", "z"))
    extractor = LLMRelationExtractor(host, tok)

    parsed = extractor.extract_claim("where is ada ?", ["where", "is", "ada", "?"])
    assert parsed is None


def test_llm_extractor_returns_none_when_llm_emits_unparseable_output():
    host, tok = _stub_extractor_pair(lambda _s: None)
    extractor = LLMRelationExtractor(host, tok)

    utterance = "ada is in rome ."
    parsed = extractor.extract_claim(utterance, utterance.split())

    assert parsed is None  # no heuristic fallback — extraction failure is final.


def test_claim_trust_weight_decays_with_prediction_gap():
    no_gap = {"evidence": {}}
    low_gap = {"evidence": {"prediction_gap": 0.1}}
    high_gap = {"evidence": {"prediction_gap": 5.0}}

    assert _claim_trust_weight(no_gap) == 1.0
    assert 0.5 < _claim_trust_weight(low_gap) < 1.0
    assert _claim_trust_weight(high_gap) < 0.25
    assert _claim_trust_weight(low_gap) > _claim_trust_weight(high_gap)


def test_consolidation_resists_high_surprise_repeated_claims(tmp_path: Path):
    """Three high-surprise (poison) challengers must NOT outweigh one trusted accepted claim."""

    db = tmp_path / "poisoning.sqlite"
    mem = PersistentSemanticMemory(db, namespace="poison")
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
    mem = PersistentSemanticMemory(db, namespace="low_surprise")
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


def test_runtime_router_uses_llm_relation_extractor(tmp_path: Path, fake_host_loader):
    fake_host_loader()
    mind = SubstrateController(seed=0, db_path=tmp_path / "router_extractor.sqlite", namespace="ext")

    assert isinstance(mind.router.extractor, LLMRelationExtractor)
