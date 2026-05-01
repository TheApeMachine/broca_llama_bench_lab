from __future__ import annotations

import random
from pathlib import Path

from asi_broca_core.hawkes import (
    MultivariateHawkesProcess,
    PersistentHawkes,
    fit_excitation_em,
)


def test_intensity_jumps_after_observation_and_decays():
    proc = MultivariateHawkesProcess(beta=1.0, baseline=0.05)
    proc.couple("ada", "london", weight=0.8)
    base = proc.intensity("london", t=0.0)
    proc.observe("ada", t=0.0)
    just_after = proc.intensity("london", t=0.001)
    much_later = proc.intensity("london", t=10.0)
    assert just_after > base, f"expected excitation, got base={base:.4f} after={just_after:.4f}"
    assert much_later < just_after, "intensity must decay over time"
    assert much_later >= base - 1e-9


def test_self_excitation_default_makes_repeats_more_likely():
    proc = MultivariateHawkesProcess(beta=0.5, baseline=0.05)
    proc.observe("ada", t=0.0)
    after_one = proc.intensity("ada", t=0.1)
    proc.observe("ada", t=0.1)
    after_two = proc.intensity("ada", t=0.1001)
    assert after_two > after_one, f"second event should excite further: {after_one:.4f} -> {after_two:.4f}"


def test_intensity_vector_returns_all_channels():
    proc = MultivariateHawkesProcess(beta=0.5, baseline=0.05)
    proc.couple("ada", "london", weight=0.8)
    proc.observe("ada", t=1.0)
    intensities = proc.intensity_vector(t=1.001)
    assert "ada" in intensities and "london" in intensities
    assert intensities["ada"] >= proc.baseline


def test_persistent_hawkes_round_trip(tmp_path: Path):
    proc = MultivariateHawkesProcess(beta=0.4, baseline=0.07)
    proc.couple("a", "b", weight=0.5)
    proc.observe("a", t=1.0)
    proc.observe("b", t=2.0)

    db = tmp_path / "hawkes.sqlite"
    store = PersistentHawkes(db, namespace="t")
    store.save(proc)

    loaded = store.load()
    assert loaded is not None
    assert loaded.channels == proc.channels
    assert loaded.alpha == proc.alpha
    assert loaded.mu == proc.mu


def test_em_recovers_self_excitation_pattern():
    rng = random.Random(0)
    # Synthetic clustered events on a single channel.
    events: list[tuple[str, float]] = []
    t = 0.0
    cluster_centers = [10.0, 25.0, 40.0]
    for center in cluster_centers:
        events.append(("a", center))
        for _ in range(4):
            events.append(("a", center + rng.random() * 0.5))
    mu, alpha = fit_excitation_em(events, ["a"], beta=1.0, iterations=15)
    # In a clustered self-exciting sequence, alpha[0][0] should dominate mu.
    assert alpha[0][0] > mu[0]


def test_nll_increases_when_alpha_misspecified():
    rng = random.Random(1)
    proc = MultivariateHawkesProcess(beta=0.5, baseline=0.1)
    events = []
    for k in range(50):
        events.append(("a", float(k)))
        if rng.random() < 0.5:
            events.append(("b", float(k) + 0.1))
    proc.couple("a", "b", weight=1.0)
    correct = proc.negative_log_likelihood(events)
    proc.couple("a", "b", weight=0.0)  # disable real coupling
    misspec = proc.negative_log_likelihood(events)
    assert correct < misspec, f"correctly-specified alpha should have lower NLL: correct={correct:.3f} misspec={misspec:.3f}"
