
from pathlib import Path

from asi_broca_core.broca import BrocaMind, run_broca_experiment, train_broca_bridge


def _speech_hints_answer(speech: str, *needles: str) -> bool:
    low = speech.lower().replace("Ġ", " ")
    return all(n.lower() in low for n in needles)


def test_broca_mind_uses_substrate_and_persists(tmp_path: Path, llama_broca_loaded: None):
    db = tmp_path / "broca.sqlite"
    mind = BrocaMind(seed=0, db_path=db, namespace="t")
    frame, speech = mind.answer("where is ada ?")
    assert frame.answer == "rome"
    assert _speech_hints_answer(speech, "ada", "rome")

    restarted = BrocaMind(seed=0, db_path=db, namespace="t")
    frame2, speech2 = restarted.answer("where is hopper ?")
    assert frame2.answer == "lisbon"
    assert _speech_hints_answer(speech2, "hopper", "lisbon")


def test_trainable_broca_bridge_learns_frame_to_language(llama_broca_loaded: None):
    result = train_broca_bridge(seed=0, steps=60)
    assert result["after_accuracy"] >= result["before_accuracy"]
    assert result["after_accuracy"] >= 0.35
    gen0 = result["generated"][0].lower().replace("Ġ", " ")
    assert "ada" in gen0 and "rome" in gen0


def test_broca_experiment_outputs_expected_rows(tmp_path: Path, llama_broca_loaded: None):
    result = run_broca_experiment(seed=0, db_path=tmp_path / "broca.sqlite", verbose=False)
    speeches_lower = [row["speech"].lower().replace("Ġ", " ") for row in result["rows"]]
    assert any("rome" in s and "ada" in s for s in speeches_lower)
    action_row = next(r for r in result["rows"] if "action" in r["query"])
    assert action_row["intent"] == "active_action"
    assert action_row["speech"].lower().startswith("i should") or "listen" in action_row["speech"].lower()
    assert any("help" in s or "intervention" in s or "treatment" in s for s in speeches_lower)
    assert result["broca_lesion"]["latent_answer"] == "rome"
    assert result["restart"]["latent_answer"] == "lisbon"


