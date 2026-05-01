from __future__ import annotations

from pathlib import Path

from core.preference_learning import (
    DirichletPreference,
    PersistentPreference,
    feedback_polarity_from_text,
)


def test_initial_prior_is_uniform_when_no_C_supplied():
    pref = DirichletPreference(n_observations=4)
    mean = pref.mean
    assert all(abs(m - 0.25) < 1e-6 for m in mean)


def test_positive_update_increases_target_and_renormalizes():
    pref = DirichletPreference(n_observations=4)
    before = pref.mean
    pref.update(2, polarity=1.0, weight=5.0, reason="user said thanks")
    after = pref.mean
    assert after[2] > before[2]
    # Other entries renormalize down.
    for i in range(4):
        if i != 2:
            assert after[i] < before[i]


def test_negative_update_shrinks_alpha_strictly_positive():
    pref = DirichletPreference(n_observations=3)
    pref.update(0, polarity=-2.0, weight=2.0)
    assert pref.alpha[0] > 0
    # Mean on index 0 should be strictly less than initial uniform.
    assert pref.mean[0] < 1.0 / 3.0


def test_epistemic_floor_clamps_negative_update():
    pref = DirichletPreference(n_observations=3, prior_strength=10.0)
    initial_alpha = pref.alpha[0]
    pref.update(
        0,
        polarity=-8.0,
        weight=4.0,
        epistemic_alpha_floor=2.5,
    )
    assert pref.alpha[0] >= 2.5 - 1e-6
    assert pref.alpha[0] < initial_alpha


def test_kl_to_uniform_grows_with_concentration():
    pref = DirichletPreference(n_observations=4)
    kl_initial = pref.kl_to_uniform()
    assert kl_initial < 1e-6
    for _ in range(20):
        pref.update(0, polarity=1.0)
    kl_after = pref.kl_to_uniform()
    assert kl_after > kl_initial


def test_persistence_round_trip(tmp_path: Path):
    pref = DirichletPreference(n_observations=4, prior_strength=2.0)
    pref.update(1, polarity=1.0, weight=3.0, reason="hi")
    pref.update(3, polarity=-1.0, weight=1.0, reason="no")

    store = PersistentPreference(tmp_path / "pref.sqlite", namespace="t")
    store.save("spatial", pref)

    loaded = store.load("spatial")
    assert loaded is not None
    assert loaded.n_observations == 4
    assert loaded.prior_strength == pref.prior_strength
    assert all(abs(a - b) < 1e-6 for a, b in zip(loaded.alpha, pref.alpha))
    assert len(loaded.history) == 2
    assert all(hasattr(ev, "timestamp") for ev in loaded.history)


def test_initial_C_rejects_negative_entries():
    try:
        DirichletPreference(n_observations=3, initial_C=[0.1, -0.5, 0.4])
    except ValueError as exc:
        assert "must be non-negative" in str(exc)
    else:
        raise AssertionError("expected ValueError for negative initial_C entry")


def test_feedback_polarity_classifier_basic_signs():
    """Rule-based and deterministic; polarity is in {-1, 0, +1} scale (see ``feedback_polarity_from_text``)."""
    p_pos, _ = feedback_polarity_from_text("Thanks, that was great")
    p_neg, _ = feedback_polarity_from_text("Stop asking me so many questions")
    p_neutral, _ = feedback_polarity_from_text("the sky is blue")
    assert p_pos > 0.0
    assert p_neg < 0.0
    assert abs(p_neutral) < 1e-6


def test_feedback_polarity_detects_no_thanks():
    p_neg, _ = feedback_polarity_from_text("No thanks.")
    assert p_neg == -1.0


def test_no_problem_without_positive_cue_is_neutral():
    p, _ = feedback_polarity_from_text("No problem.")
    assert abs(p) < 1e-6


def test_initial_C_seeds_preference_correctly():
    pref = DirichletPreference(
        n_observations=3, initial_C=[0.1, 0.7, 0.2], prior_strength=10.0
    )
    mean = pref.mean
    # The mean stays in the same relative order as initial_C even with prior strength scaling.
    assert mean[1] > mean[2] > mean[0]
