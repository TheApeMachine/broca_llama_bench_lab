from __future__ import annotations

import random
from pathlib import Path

from asi_broca_core.conformal import (
    ConformalPredictor,
    PersistentConformalCalibration,
    empirical_coverage,
)


def _synthetic_calibration(rng: random.Random, n: int, k: int = 4) -> list[tuple[dict[str, float], str]]:
    """Generate (distribution, true label) pairs from a noisy categorical model."""

    labels = [f"L{i}" for i in range(k)]
    out = []
    for _ in range(n):
        true_idx = rng.randrange(k)
        noise = rng.random()
        # Noisier outputs assign probability mass to wrong labels too.
        scores = [rng.random() * (0.3 if i != true_idx else 1.0) for i in range(k)]
        scores[true_idx] += 0.3 - 0.2 * noise
        z = sum(scores)
        distribution = {labels[i]: scores[i] / z for i in range(k)}
        out.append((distribution, labels[true_idx]))
    return out


def test_lac_predictor_meets_target_coverage():
    rng = random.Random(0)
    target_alpha = 0.1
    predictor = ConformalPredictor(alpha=target_alpha, method="lac")
    train = _synthetic_calibration(rng, 400)
    test = _synthetic_calibration(rng, 200)
    for dist, label in train:
        predictor.calibrate(dist[label])
    coverage = empirical_coverage(predictor, test)
    # Empirical coverage should be at least 1 - α with margin for sampling noise.
    assert coverage >= 0.85, f"coverage too low: {coverage:.3f}"
    assert coverage <= 0.99


def test_aps_predictor_meets_target_coverage():
    rng = random.Random(1)
    target_alpha = 0.2
    predictor = ConformalPredictor(alpha=target_alpha, method="aps")
    train = _synthetic_calibration(rng, 400)
    test = _synthetic_calibration(rng, 200)
    for dist, label in train:
        predictor.calibrate(dist[label], p_distribution=dist)
    coverage = empirical_coverage(predictor, test)
    assert coverage >= 0.7, f"APS coverage {coverage:.3f}"


def test_predict_set_is_nonempty_and_includes_top_under_no_calibration():
    predictor = ConformalPredictor(alpha=0.1, method="lac", min_calibration=8)
    dist = {"a": 0.6, "b": 0.3, "c": 0.1}
    result = predictor.predict_set(dist)
    assert "a" in result.labels
    # With no calibration, the threshold is +inf so the set holds everything.
    assert result.set_size == 3


def test_set_size_collapses_to_one_for_confident_prediction():
    rng = random.Random(2)
    predictor = ConformalPredictor(alpha=0.1, method="lac")
    train = _synthetic_calibration(rng, 200)
    for dist, label in train:
        predictor.calibrate(dist[label])
    confident = {"a": 0.97, "b": 0.02, "c": 0.01}
    result = predictor.predict_set(confident)
    assert result.confident, f"set size {result.set_size}, labels={result.labels}"
    assert result.ambiguity == 0.0


def test_persistent_calibration_round_trip(tmp_path: Path):
    db = tmp_path / "conformal.sqlite"
    store = PersistentConformalCalibration(db, namespace="t")
    predictor = ConformalPredictor(alpha=0.1, method="lac")
    for p in [0.7, 0.8, 0.5, 0.6, 0.9, 0.4, 0.65, 0.55]:
        predictor.calibrate(p)
    store.persist(predictor, channel="rel")
    fresh = ConformalPredictor(alpha=0.1, method="lac")
    store.hydrate(fresh, channel="rel")
    assert len(fresh) == len(predictor)
    assert sorted(fresh._scores) == sorted(predictor._scores)


def test_threshold_monotonic_in_alpha():
    rng = random.Random(3)
    predictor = ConformalPredictor(alpha=0.5, method="lac")
    for _ in range(120):
        predictor.calibrate(rng.random())
    predictor.alpha = 0.1
    t_relaxed = predictor.threshold()
    predictor.alpha = 0.4
    t_strict = predictor.threshold()
    # Higher alpha => tighter threshold (smaller score allowed).
    assert t_strict <= t_relaxed + 1e-9
