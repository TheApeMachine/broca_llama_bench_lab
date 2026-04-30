from __future__ import annotations

import types
import uuid
from pathlib import Path

import pytest
import torch

import asi_broca_core.broca as broca_mod
from asi_broca_core.broca import (
    BrocaMind,
    HeuristicRelationExtractor,
    LLMRelationExtractor,
    PersistentSemanticMemory,
    _claim_trust_weight,
)


def _symbol(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


class FakeHost:
    cfg = types.SimpleNamespace(d_model=8)

    def __init__(self):
        self.grafts: list = []

    def add_graft(self, slot, graft):
        self.grafts.append((slot, graft))


@pytest.fixture
def fake_host_loader(monkeypatch: pytest.MonkeyPatch):
    def _make() -> FakeHost:
        host = FakeHost()
        monkeypatch.setattr(broca_mod, "load_llama_broca_host", lambda *args, **kwargs: (host, object()))
        return host

    return _make


class _StubHFTokenizer:
    """Stand-in for an HF tokenizer surface that the LLM extractor relies on."""

    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 0

    def __call__(self, prompt, return_tensors="pt"):
        return {"input_ids": torch.zeros((1, 4), dtype=torch.long), "attention_mask": torch.ones((1, 4), dtype=torch.long)}

    def decode(self, ids, skip_special_tokens=True):
        return self._decoded


class _StubLLM:
    """Pretends to be an HF causal LM. The single output row is the extractor's answer."""

    def __init__(self, decoded_json: str):
        self._decoded_json = decoded_json

    def parameters(self):
        yield torch.zeros(1)

    def generate(self, *, input_ids, attention_mask=None, max_new_tokens=64, do_sample=False, pad_token_id=None):
        return torch.zeros((1, input_ids.shape[1] + 4), dtype=torch.long)


def _stub_pair(decoded_json: str):
    host = types.SimpleNamespace(llm=_StubLLM(decoded_json))
    tok = types.SimpleNamespace(inner=_StubHFTokenizer())
    tok.inner._decoded = decoded_json
    return host, tok


def test_llm_extractor_handles_subordinate_clause_that_breaks_heuristic():
    utterance = "the apple, which was incredibly red, fell from the tree ."
    toks = utterance.split()

    heuristic = HeuristicRelationExtractor().extract_claim(utterance, toks)
    # The whitespace heuristic mis-assigns subject/object because of the relative clause;
    # we don't pin the exact wrong values, only assert it does not recover the true triple.
    assert heuristic is None or heuristic.subject != "apple" or heuristic.obj != "tree"

    host, tok = _stub_pair('{"subject":"apple","relation":"fell from","object":"tree"}')
    llm_extractor = LLMRelationExtractor(host, tok)
    parsed = llm_extractor.extract_claim(utterance, toks)

    assert parsed is not None
    assert parsed.subject == "apple"
    assert parsed.predicate == "fell from"
    assert parsed.obj == "tree"
    assert parsed.evidence["parser"] == "llm_relation_extractor"


def test_llm_extractor_falls_back_to_heuristic_when_host_lacks_llm():
    host = types.SimpleNamespace()  # no .llm attribute
    tok = types.SimpleNamespace(inner=_StubHFTokenizer())
    extractor = LLMRelationExtractor(host, tok)

    utterance = "ada is in rome ."
    parsed = extractor.extract_claim(utterance, utterance.split())

    assert parsed is not None
    assert parsed.subject == "ada"
    assert parsed.obj == "rome"
    assert parsed.evidence["parser"] == "open_relation_claim"  # fallback path


def test_llm_extractor_skips_questions():
    host, tok = _stub_pair('{"subject":"x","relation":"y","object":"z"}')
    extractor = LLMRelationExtractor(host, tok)

    parsed = extractor.extract_claim("where is ada ?", ["where", "is", "ada", "?"])
    assert parsed is None


def test_llm_extractor_falls_back_when_json_unparseable():
    host, tok = _stub_pair("garbage no json at all")
    extractor = LLMRelationExtractor(host, tok)

    utterance = "ada is in rome ."
    parsed = extractor.extract_claim(utterance, utterance.split())

    assert parsed is not None
    assert parsed.subject == "ada"
    assert parsed.obj == "rome"
    assert parsed.evidence["parser"] == "open_relation_claim"


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
    mind = BrocaMind(seed=0, db_path=tmp_path / "router_extractor.sqlite", namespace="ext")

    assert isinstance(mind.router.extractor, LLMRelationExtractor)
