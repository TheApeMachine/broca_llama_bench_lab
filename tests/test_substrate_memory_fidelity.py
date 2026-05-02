"""Semantic-memory fidelity benchmark produces finite confidence deltas."""

from __future__ import annotations

import math

from research_lab.benchmarks.substrate_eval import bench_memory_fidelity


def test_bench_memory_fidelity_reports_finite_avg_confidence_error() -> None:
    r = bench_memory_fidelity(n_triples=12, seed=2)
    err = r.details.get("avg_confidence_error")
    assert err is not None
    assert isinstance(err, float)
    assert math.isfinite(err)
