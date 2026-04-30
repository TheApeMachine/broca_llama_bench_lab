
from pathlib import Path

from asi_broca_core.broca import BrocaMind, run_broca_experiment, train_broca_bridge


def test_broca_mind_uses_substrate_and_persists(tmp_path: Path):
    db = tmp_path / "broca.sqlite"
    mind = BrocaMind(seed=0, db_path=db, namespace="t")
    frame, speech = mind.answer("where is ada ?")
    assert frame.answer == "rome"
    assert speech == "ada is in rome ."

    restarted = BrocaMind(seed=0, db_path=db, namespace="t")
    frame2, speech2 = restarted.answer("where is hopper ?")
    assert frame2.answer == "lisbon"
    assert speech2 == "hopper is in lisbon ."


def test_trainable_broca_bridge_learns_frame_to_language():
    result = train_broca_bridge(seed=0, steps=60)
    assert result["before_accuracy"] < 0.25
    assert result["after_accuracy"] >= 0.95
    assert result["generated"][0] == "ada is in rome ."


def test_broca_experiment_outputs_expected_rows(tmp_path: Path):
    result = run_broca_experiment(seed=0, db_path=tmp_path / "broca.sqlite", verbose=False)
    speeches = [row["speech"] for row in result["rows"]]
    assert "ada is in rome ." in speeches
    assert "i should listen first ." in speeches
    assert "intervention says treatment helps ." in speeches
    assert result["broca_lesion"]["latent_answer"] == "rome"
    assert result["restart"]["latent_answer"] == "lisbon"


