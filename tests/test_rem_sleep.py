"""Tests for the DMN's REM phase.

The full ``SubstrateController`` requires a Llama checkpoint, so we wire up a synthetic
"mind" exposing only the surfaces the REM phase touches. This isolates the
sleep-cycle plumbing from the heavy substrate stack.
"""

from __future__ import annotations

import random
import sqlite3
import threading
import types
from pathlib import Path

from core.cognition.substrate import SubstrateController  # noqa: F401  (keeps import-time wiring active)
from core.dmn import CognitiveBackgroundWorker, DMNConfig
from core.frame import CognitiveFrame
from core.memory import SymbolicMemory, WorkspaceJournal
from core.causal import build_simpson_scm
from core.calibration.conformal import ConformalPredictor, PersistentConformalCalibration
from core.temporal.hawkes import MultivariateHawkesProcess, PersistentHawkes
from core.idletime.ontological_expansion import OntologicalRegistry, PersistentOntologicalRegistry
from core.learning.preference_learning import DirichletPreference, PersistentPreference
from core.substrate.graph import EpisodeAssociationGraph
from core.frame import FrameDimensions

SKETCH_DIM = FrameDimensions.SKETCH_DIM


def _build_synthetic_mind(tmp: Path) -> types.SimpleNamespace:
    db = tmp / "rem.sqlite"
    memory = SymbolicMemory(db, namespace="t")
    journal = WorkspaceJournal(db, shared_memory=memory)
    graph = EpisodeAssociationGraph(db)
    rng = random.Random(0)

    # Seed a chain X -> Y -> Z so PC has signal to recover.
    for i in range(60):
        x = rng.randint(0, 1)
        y = x if rng.random() < 0.85 else 1 - x
        z = y if rng.random() < 0.85 else 1 - y
        memory.upsert(f"row_{i}", "X", str(x))
        memory.upsert(f"row_{i}", "Y", str(y))
        memory.upsert(f"row_{i}", "Z", str(z))

    # Seed a journal with a clustered intent stream so Hawkes EM has signal.
    # (``WorkspaceJournal.append`` timestamps rows with ``time.time()``; no per-row override API.)
    f = CognitiveFrame("memory_write", subject="row_a", answer="value")
    for i in range(20):
        journal.append(f"utterance {i}", f)

    workspace = types.SimpleNamespace(intrinsic_cues=[])
    spatial_pref = DirichletPreference(n_observations=4)
    causal_pref = DirichletPreference(n_observations=4)
    pomdp = types.SimpleNamespace(C=list(spatial_pref.expected_C()), observation_names=["a", "b", "c", "d"])
    causal_pomdp = types.SimpleNamespace(C=list(causal_pref.expected_C()), observation_names=["a", "b", "c", "d"])
    mind = types.SimpleNamespace(
        memory=memory,
        journal=journal,
        episode_graph=graph,
        workspace=workspace,
        scm=build_simpson_scm(),
        text_encoder=None,
        consolidate_once=lambda: [],
        hawkes=MultivariateHawkesProcess(beta=0.5, baseline=0.05),
        hawkes_persistence=PersistentHawkes(db, namespace="t__hawkes"),
        spatial_preference=spatial_pref,
        causal_preference=causal_pref,
        preference_persistence=PersistentPreference(db, namespace="t__pref"),
        pomdp=pomdp,
        causal_pomdp=causal_pomdp,
        ontology=OntologicalRegistry(dim=SKETCH_DIM),
        ontology_persistence=PersistentOntologicalRegistry(db, namespace="t__ont"),
        relation_conformal=ConformalPredictor(alpha=0.1, method="lac"),
        conformal_calibration=PersistentConformalCalibration(db, namespace="t__conf"),
        motor_replay=[],
        discovered_scm=None,
        process_deferred_relation_ingest=lambda: [],
        deferred_relation_ingest_count=lambda: 0,
        _cognitive_state_lock=threading.Lock(),
        event_bus=types.SimpleNamespace(publish=lambda *args, **kwargs: None),
    )
    return mind


def test_rem_phase_runs_when_idle_threshold_exceeded(tmp_path: Path):
    mind = _build_synthetic_mind(tmp_path)
    cfg = DMNConfig(
        sleep_idle_seconds=0.0,  # always asleep so REM fires immediately
        sleep_max_replay=4,
        sleep_min_observations_for_pc=10,
        sleep_pc_alpha=0.05,
        sleep_hawkes_min_events=4,
    )
    worker = CognitiveBackgroundWorker(mind, interval_s=999.0, config=cfg, rng=random.Random(7))
    reflections = worker.run_once()
    assert isinstance(reflections, list)
    assert all(isinstance(r, dict) for r in reflections)
    summary = worker.last_phase_summary.get("rem_sleep", {})
    assert summary, f"REM did not run; full summary={worker.last_phase_summary}"
    assert summary.get("hawkes", {}).get("ran"), summary
    # Causal discovery should fire and produce at least one undirected edge.
    cd = summary.get("causal_discovery", {})
    assert cd.get("ran"), cd
    assert mind.discovered_scm is not None

    db_file = tmp_path / "rem.sqlite"
    con = sqlite3.connect(str(db_file))
    try:
        hawkes_rows = int(con.execute("SELECT COUNT(*) FROM hawkes_state").fetchone()[0])
        assert hawkes_rows >= 1
    finally:
        con.close()


def test_rem_phase_skipped_when_user_active(tmp_path: Path):
    mind = _build_synthetic_mind(tmp_path)
    cfg = DMNConfig(sleep_idle_seconds=600.0, sleep_min_observations_for_pc=999, sleep_hawkes_min_events=999)
    worker = CognitiveBackgroundWorker(mind, interval_s=999.0, config=cfg, rng=random.Random(7))
    worker.mark_user_active()
    worker.run_once()
    assert "rem_sleep" not in worker.last_phase_summary


def test_rem_phase_persists_preferences(tmp_path: Path):
    mind = _build_synthetic_mind(tmp_path)
    mind.spatial_preference.update(2, polarity=1.0, weight=2.0, reason="positive")
    cfg = DMNConfig(sleep_idle_seconds=0.0, sleep_min_observations_for_pc=999, sleep_hawkes_min_events=999)
    worker = CognitiveBackgroundWorker(mind, interval_s=999.0, config=cfg, rng=random.Random(7))
    worker.run_once()
    loaded = mind.preference_persistence.load("spatial")
    assert loaded is not None
    # Saved alpha matches in-memory alpha after REM.
    assert all(abs(a - b) < 1e-6 for a, b in zip(loaded.alpha, mind.spatial_preference.alpha))


def test_rem_persists_conformal_scores_after_calibration(tmp_path: Path) -> None:
    mind = _build_synthetic_mind(tmp_path)
    warm = ConformalPredictor(alpha=0.1, method="lac", min_calibration=1)
    mind.relation_conformal = warm
    mind.conformal_calibration.hydrate(warm, "relation_extraction")
    warm.calibrate(p_label=0.92)
    cfg = DMNConfig(
        sleep_idle_seconds=0.0,
        sleep_min_observations_for_pc=999,
        sleep_hawkes_min_events=999,
    )
    worker = CognitiveBackgroundWorker(mind, interval_s=999.0, config=cfg, rng=random.Random(11))
    worker.run_once()
    con = sqlite3.connect(str(tmp_path / "rem.sqlite"))
    try:
        scores_n = int(con.execute("SELECT COUNT(*) FROM conformal_scores").fetchone()[0])
        assert scores_n >= 1
    finally:
        con.close()
