from __future__ import annotations

from pathlib import Path

from asi_broca_core.preference_learning import (
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
    pref.update(0, polarity=2.0, weight=2.0)
    pref.update(0, polarity=-2.0, weight=2.0)
    assert pref.alpha[0] > 0
    # Mean on index 0 should be strictly less than initial uniform.
    assert pref.mean[0] < 1.0 / 3.0


def test_kl_to_uniform_grows_with_concentration():
    pref = DirichletPreference(n_observations=4)
    kl_initial = pref.kl_to_uniform()
    for _ in range(20):
        pref.update(0, polarity=1.0)
    kl_after = pref.kl_to_uniform()
    assert kl_after > kl_initial
    assert kl_initial < 1e-6


def test_persistence_round_trip(tmp_path: Path):
    pref = DirichletPreference(n_observations=4, prior_strength=2.0)
    pref.update(1, polarity=1.0, weight=3.0, reason="hi")
    pref.update(3, polarity=-1.0, weight=1.0, reason="no")

    store = PersistentPreference(tmp_path / "pref.sqlite", namespace="t")
    store.save("spatial", pref)

    loaded = store.load("spatial")
    assert loaded is not None
    assert loaded.n_observations == 4
    assert all(abs(a - b) < 1e-6 for a, b in zip(loaded.alpha, pref.alpha))
    assert len(loaded.history) == 2


def test_feedback_polarity_classifier_basic_signs():
    p_pos, _ = feedback_polarity_from_text("Thanks, that was great")
    p_neg, _ = feedback_polarity_from_text("Stop asking me so many questions")
    p_neutral, _ = feedback_polarity_from_text("the sky is blue")
    assert p_pos == 1.0
    assert p_neg == -1.0
    assert p_neutral == 0.0


def test_initial_C_seeds_preference_correctly():
    pref = DirichletPreference(n_observations=3, initial_C=[0.1, 0.7, 0.2], prior_strength=10.0)
    mean = pref.mean
    # The mean stays in the same relative order as initial_C even with prior strength scaling.
    assert mean[1] > mean[2] > mean[0]
