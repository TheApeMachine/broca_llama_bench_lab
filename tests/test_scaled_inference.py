from __future__ import annotations

from asi_broca_core.active_inference import (
    ActiveInferenceAgent,
    CoupledEFEAgent,
    build_causal_epistemic_pomdp,
    build_tiger_pomdp,
)
from asi_broca_core.broca import BrocaMind
from asi_broca_core.causal import build_simpson_scm


def test_expand_state_grows_latent_cardinality():
    pomdp = build_tiger_pomdp()
    assert pomdp.n_states == 2
    new_qs = pomdp.expand_state_with_mass("hyp_x", qs=[0.5, 0.5], mass=0.12)
    assert len(new_qs) == 3
    assert pomdp.n_states == 3
    assert pomdp.B[0][2][2] >= 0.0


def test_coupled_agent_argmin_matches_policy_ordering():
    scm = build_simpson_scm()
    spatial = ActiveInferenceAgent(build_tiger_pomdp(), horizon=1, learn=False)
    causal = ActiveInferenceAgent(build_causal_epistemic_pomdp(scm), horizon=1, learn=False)
    coupled = CoupledEFEAgent(spatial, causal)
    d = coupled.decide()
    expect_spatial = d.spatial_min_G <= d.causal_min_G
    assert (d.faculty == "spatial") == expect_spatial
    assert d.action_name == (d.spatial_decision.action_name if expect_spatial else d.causal_decision.action_name)


def test_intrinsic_scan_records_low_semantic_confidence(tmp_path, llama_broca_loaded: None):
    db = tmp_path / "broca.sqlite"
    mind = BrocaMind(seed=0, db_path=db, namespace="sc")
    mind.memory.upsert("ada", "location", "rome", confidence=0.22, evidence={"probe": "metacog"})
    mind.answer("where is ada ?")
    kinds = [c.faculty for c in mind.workspace.intrinsic_cues]
    assert "memory_low_confidence" in kinds


def test_low_memory_confidence_verify_speech_plan(tmp_path, llama_broca_loaded: None):
    db = tmp_path / "broca.sqlite"
    mind = BrocaMind(seed=0, db_path=db, namespace="sp")
    mind.memory.upsert("ada", "location", "rome", confidence=0.22, evidence={"probe": "metacog"})
    frame = mind.comprehend("where is ada ?")
    assert frame.answer == "rome"
    plan = frame.speech_plan()
    assert "verify" in plan


def test_causal_epistemic_pomdp_lists_intervention_action():
    pomdp = build_causal_epistemic_pomdp(build_simpson_scm())
    assert "run_intervention_readout" in pomdp.action_names
    assert pomdp.n_observations == 2
