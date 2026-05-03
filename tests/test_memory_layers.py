from pathlib import Path
import types
import uuid

import torch

import pytest

from core.cli import build_substrate_controller
from core.frame import CognitiveFrame
from core.grafts import TrainableFeatureGraft
from core.memory import WorkspaceJournal
from core.workspace import GlobalWorkspace
import core.cognition.substrate as substrate_mod
from core.memory import SQLiteActivationMemory
from core.substrate.graph import EpisodeAssociationGraph, merge_epistemic_evidence_dict

from conftest import FakeHost, FakeTokenizer, make_stub_llm_pair, stub_substrate_encoders


@pytest.fixture
def fake_host_loader(monkeypatch: pytest.MonkeyPatch):
    def _make(track_grafts: bool = False) -> FakeHost:
        host = FakeHost(track_grafts=track_grafts)
        tokenizer = FakeTokenizer(host._stub_tokenizer)
        monkeypatch.setattr(substrate_mod, "load_llama_broca_host", lambda *args, **kwargs: (host, tokenizer))
        return host

    return _make


def _symbol(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _process_deferred(mind):
    reflections = mind.process_deferred_relation_ingest()
    assert reflections, "expected queued deferred relation ingest"
    return reflections[-1]


def test_episode_association_graph_persistent(tmp_path: Path):
    db = tmp_path / "m.sqlite"
    g = EpisodeAssociationGraph(db)
    g.bump(1, 2)
    g.bump(2, 3)
    assert g.weight(1, 2) > 0
    g2 = EpisodeAssociationGraph(db)
    assert g2.weight(1, 2) == g.weight(1, 2)


def test_workspace_journal_fetch_roundtrip(tmp_path: Path, llama_broca_loaded: None):
    subject = _symbol("subject")
    obj = _symbol("object")
    mind = build_substrate_controller(seed=0, db_path=tmp_path / "b.sqlite", namespace="x", device="cpu", hf_token=False)
    stub_substrate_encoders(mind)
    mind.answer(f"{subject} is in {obj} .")
    _process_deferred(mind)
    mind.answer(f"where is {subject} ?")
    row = mind.journal.fetch(2)
    assert row is not None
    assert row["intent"] == "memory_lookup"
    replay = mind.retrieve_episode(2)
    assert replay.answer == obj
    assert replay.evidence.get("retrieved_episode_id") == 2


def test_workspace_journal_count(tmp_path: Path):
    subject = _symbol("subject")
    obj = _symbol("object")
    journal = WorkspaceJournal(tmp_path / "j.sqlite")
    frame = CognitiveFrame("memory_location", subject=subject, answer=obj, confidence=1.0)

    assert journal.count() == 0
    journal.append(f"where is {subject} ?", frame)
    assert journal.count() == 1


def test_runtime_mind_creates_sqlite_before_model_load_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def fail_load(*args, **kwargs):
        raise RuntimeError("model unavailable")

    db = tmp_path / "early.sqlite"
    monkeypatch.setattr(substrate_mod, "load_llama_broca_host", fail_load)

    with pytest.raises(RuntimeError, match="model unavailable"):
        build_substrate_controller(seed=0, db_path=db, namespace="early", device="cpu", hf_token=False)

    assert db.exists()
    assert WorkspaceJournal(db).count() == 0


def test_runtime_mind_starts_empty_and_learns_observed_location(tmp_path: Path, fake_host_loader):
    fake_host_loader(track_grafts=False)
    db = tmp_path / "learn.sqlite"
    subject = _symbol("subject")
    obj = _symbol("object")

    mind = build_substrate_controller(seed=0, db_path=db, namespace="runtime", device="cpu", hf_token=False)
    stub_substrate_encoders(mind)
    assert mind.memory.count() == 0
    assert mind.comprehend(f"where is {subject} ?").intent == "unknown"

    learned = mind.comprehend(f"{subject} is in {obj} .")
    assert learned.intent == "memory_ingest_pending"
    reflection = _process_deferred(mind)
    assert reflection["status"] == "memory_write"
    pred = reflection["evidence"]["predicate"]
    assert mind.memory.count() == 1
    assert mind.comprehend(f"where is {subject} ?").answer == obj

    restarted = build_substrate_controller(seed=0, db_path=db, namespace="runtime", device="cpu", hf_token=False)
    stub_substrate_encoders(restarted)
    assert restarted.memory.count() == 1
    assert restarted.comprehend(f"where is {subject} ?").answer == obj
    assert restarted.memory.get(subject, pred) is not None


def test_runtime_mind_stores_observed_location_while_background_worker_running(tmp_path: Path, fake_host_loader):
    class RunningBackgroundWorker:
        running = True
        notified = False

        def notify_work(self):
            self.notified = True

        def mark_user_active(self):
            pass

    fake_host_loader(track_grafts=False)
    db = tmp_path / "learn_with_worker.sqlite"
    subject = _symbol("subject")
    obj = _symbol("object")

    mind = build_substrate_controller(seed=0, db_path=db, namespace="runtime", device="cpu", hf_token=False)
    stub_substrate_encoders(mind)
    mind._background_worker = RunningBackgroundWorker()

    learned = mind.comprehend(f"{subject} is in {obj} .")

    assert learned.intent == "memory_ingest_pending"
    assert learned.evidence.get("deferred_relation_ingest") is True
    assert mind.memory.count() == 0
    assert mind._background_worker.notified is True
    reflection = _process_deferred(mind)
    assert reflection["status"] == "memory_write"
    assert reflection["answer"] == obj
    assert mind.memory.count() == 1


def test_runtime_mind_routes_faculties_and_installs_feature_graft(tmp_path: Path, fake_host_loader):
    host = fake_host_loader(track_grafts=True)
    mind = build_substrate_controller(seed=0, db_path=tmp_path / "router.sqlite", namespace="runtime", device="cpu", hf_token=False)
    stub_substrate_encoders(mind)

    assert any(isinstance(graft, TrainableFeatureGraft) for _, graft in host.grafts)
    assert mind.comprehend("what action should i take ?").intent == "active_action"
    assert mind.comprehend("does treatment help ?").intent == "causal_effect"


def test_observed_contradiction_records_counterfactual_without_overwrite(tmp_path: Path, fake_host_loader):
    fake_host_loader(track_grafts=False)
    mind = build_substrate_controller(seed=0, db_path=tmp_path / "conflict.sqlite", namespace="runtime", device="cpu", hf_token=False)
    stub_substrate_encoders(mind)
    subject = _symbol("subject")
    current = _symbol("object")
    challenger = _symbol("object")

    mind.comprehend(f"{subject} is in {current} .")
    _process_deferred(mind)
    mind.comprehend(f"{subject} is in {challenger} .")
    conflict = _process_deferred(mind)

    assert conflict["status"] == "memory_conflict"
    assert conflict["answer"] == current
    assert conflict["evidence"]["claimed_answer"] == challenger
    assert conflict["evidence"]["counterfactual"]["would_change_answer_to"] == challenger
    assert mind.comprehend(f"where is {subject} ?").answer == current
    statuses = [c["status"] for c in mind.memory.claims(subject, conflict["evidence"]["predicate"])]
    assert statuses == ["accepted", "conflict"]


def test_background_consolidation_revises_after_repeated_counterevidence(tmp_path: Path, fake_host_loader):
    fake_host_loader(track_grafts=False)
    mind = build_substrate_controller(seed=0, db_path=tmp_path / "consolidate.sqlite", namespace="runtime", device="cpu", hf_token=False)
    stub_substrate_encoders(mind)
    subject = _symbol("subject")
    current = _symbol("object")
    challenger = _symbol("object")

    mind.comprehend(f"{subject} is in {current} .")
    _process_deferred(mind)
    mind.comprehend(f"{subject} is in {challenger} .")
    _process_deferred(mind)
    assert mind.consolidate_once()[0]["kind"] == "belief_conflict"
    assert mind.comprehend(f"where is {subject} ?").answer == current

    mind.comprehend(f"{subject} is in {challenger} .")
    _process_deferred(mind)
    reflections = mind.consolidate_once()

    assert any(r["kind"] == "belief_revision" for r in reflections)
    assert mind.comprehend(f"where is {subject} ?").answer == challenger
    stored_reflections = mind.memory.reflections(kind="belief_revision")
    assert stored_reflections[-1]["evidence"]["candidate_object"] == challenger


def test_background_worker_start_stop(tmp_path: Path, fake_host_loader):
    fake_host_loader(track_grafts=False)
    mind = build_substrate_controller(seed=0, db_path=tmp_path / "worker.sqlite", namespace="runtime", device="cpu", hf_token=False)

    worker = mind.start_background(interval_s=60.0)

    assert worker.running
    mind.stop_background()
    assert not worker.running


def test_speak_records_motor_replay(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_host_loader) -> None:
    fake_host_loader(track_grafts=False)

    from core.generation import PlanForcedGenerator

    monkeypatch.setattr(
        PlanForcedGenerator,
        "generate",
        classmethod(lambda cls, *a, **k: ("surfaced", [9, 11, 13], 2.25)),
    )
    mind = build_substrate_controller(seed=0, db_path=tmp_path / "speak_replay.sqlite", namespace="runtime", device="cpu", hf_token=False)
    stub_substrate_encoders(mind)
    frame = CognitiveFrame("memory_location", subject=_symbol("subj"), answer=_symbol("loc"), confidence=0.88)

    out = mind.speak(frame)
    assert out == "surfaced"
    assert len(mind.motor_replay) == 1
    row = mind.motor_replay[0]
    assert list(row["speech_plan_tokens"].tolist()) == [9, 11, 13]
    assert abs(float(row["substrate_confidence"]) - 0.88) < 1e-6
    assert abs(float(row["substrate_inertia"]) - 2.25) < 1e-6
    msgs = row["messages"]
    assert len(msgs) == 1 and msgs[0]["role"] == "user"
    assert frame.intent in msgs[0]["content"] and frame.subject in msgs[0]["content"]


def test_working_memory_synthesis_binds_episodes():
    ws = GlobalWorkspace()
    subject = _symbol("subject")
    obj = _symbol("object")
    a = CognitiveFrame("memory_location", subject=subject, answer=obj, confidence=0.9, evidence={"journal_id": 10})
    b = CognitiveFrame("causal_effect", subject="treatment", answer="helps", confidence=0.8, evidence={"journal_id": 11, "ate": 0.05})
    ws.post_frame(a)
    ws.post_frame(b)
    syn = [f for f in ws.frames if f.intent == "synthesis_bundle"]
    assert syn
    assert syn[-1].subject == subject
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
    subject = _symbol("subject")
    obj = _symbol("object")
    frames = [
        CognitiveFrame("memory_location", subject=subject, answer=obj, confidence=1.0, evidence={"journal_id": 3}),
        CognitiveFrame("causal_effect", subject="treatment", answer="helps", confidence=1.0, evidence={"journal_id": 4}),
    ]
    syn = CognitiveFrame.synthesize_bundle(frames)
    assert syn is not None
    assert syn.intent == "synthesis_bundle"
