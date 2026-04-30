from pathlib import Path

import torch

from asi_broca_core.broca import BrocaMind, CognitiveFrame, GlobalWorkspace, working_memory_synthesize
from asi_broca_core.memory import SQLiteActivationMemory
from asi_broca_core.substrate_graph import EpisodeAssociationGraph, merge_epistemic_evidence_dict


def test_episode_association_graph_persistent(tmp_path: Path):
    db = tmp_path / "m.sqlite"
    g = EpisodeAssociationGraph(db)
    g.bump(1, 2)
    g.bump(2, 3)
    assert g.weight(1, 2) > 0
    g2 = EpisodeAssociationGraph(db)
    assert g2.weight(1, 2) == g.weight(1, 2)


def test_workspace_journal_fetch_roundtrip(tmp_path: Path, llama_broca_loaded: None):
    mind = BrocaMind(seed=0, db_path=tmp_path / "b.sqlite", namespace="x")
    mind.answer("where is ada ?")
    row = mind.journal.fetch(1)
    assert row is not None
    assert row["intent"] == "memory_location"
    replay = mind.retrieve_episode(1)
    assert replay.answer == "rome"
    assert replay.evidence.get("retrieved_episode_id") == 1


def test_working_memory_synthesis_binds_episodes():
    ws = GlobalWorkspace()
    a = CognitiveFrame("memory_location", subject="ada", answer="rome", confidence=0.9, evidence={"journal_id": 10})
    b = CognitiveFrame("causal_effect", subject="treatment", answer="helps", confidence=0.8, evidence={"journal_id": 11, "ate": 0.05})
    ws.publish(a)
    ws.publish(b)
    syn = [f for f in ws.frames if f.intent == "synthesis_bundle"]
    assert syn
    assert syn[-1].subject == "ada"
    assert 10 in syn[-1].evidence["episode_ids"]
    assert 11 in syn[-1].evidence["episode_ids"]


def test_merge_epistemic_evidence_dict_union():
    base = {"episode_ids": [1], "instruments": ["a"]}
    inc = {"episode_ids": [1, 2], "instruments": ["b"], "journal_id": 99}
    out = merge_epistemic_evidence_dict(base, inc)
    assert out["episode_ids"][:2] == [1, 2]
    assert set(out["instruments"]) == {"a", "b"}
    assert 99 in out["episode_ids"]


def test_activation_association_spread_matrix(tmp_path: Path):
    store = SQLiteActivationMemory(tmp_path / "act.sqlite")
    k = torch.randn(4)
    v = torch.randn(4)
    i1 = store.write(k, v, metadata={"a": 1})
    i2 = store.write(k + 0.1, v + 0.1, metadata={"b": 2})
    store.bump_association(i1, i2)
    mat = store.normalized_spread_matrix([i1, i2])
    assert mat.shape == (2, 2)
    assert torch.allclose(mat.sum(dim=-1), torch.ones(2), atol=1e-5)


def test_working_memory_synthesize_standalone():
    frames = [
        CognitiveFrame("memory_location", subject="ada", answer="rome", confidence=1.0, evidence={"journal_id": 3}),
        CognitiveFrame("causal_effect", subject="treatment", answer="helps", confidence=1.0, evidence={"journal_id": 4}),
    ]
    syn = working_memory_synthesize(frames)
    assert syn is not None
    assert syn.intent == "synthesis_bundle"
