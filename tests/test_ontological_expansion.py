from __future__ import annotations

from pathlib import Path

import torch

from core.frame.continuous_frame import SKETCH_DIM, stable_sketch
from core.idletime.ontological_expansion import (
    OntologicalRegistry,
    PersistentOntologicalRegistry,
    gram_schmidt_orthogonalize,
)


def test_gram_schmidt_produces_orthogonal_axes():
    a = torch.randn(SKETCH_DIM, generator=torch.Generator().manual_seed(0))
    b = torch.randn(SKETCH_DIM, generator=torch.Generator().manual_seed(1))
    a = a / a.norm()
    b_o = gram_schmidt_orthogonalize(b, [a])
    assert abs(float(torch.dot(a, b_o).item())) < 1e-5
    assert abs(float(b_o.norm().item()) - 1.0) < 1e-5


def test_registry_promotes_only_after_threshold():
    registry = OntologicalRegistry(dim=SKETCH_DIM, frequency_threshold=3)
    base = stable_sketch("ada", dim=SKETCH_DIM)
    for i in range(2):
        registry.observe("ada")
    assert registry.maybe_promote("ada", base) is None
    registry.observe("ada")
    promoted = registry.maybe_promote("ada", base)
    assert promoted is not None
    assert promoted.name == "ada"
    # A second call returns the already-promoted concept (idempotent).
    assert registry.maybe_promote("ada", base) is promoted


def test_promoted_axes_are_mutually_orthogonal():
    registry = OntologicalRegistry(dim=SKETCH_DIM, frequency_threshold=1)
    names = ["ada", "alan", "alice", "bob"]
    for n in names:
        registry.observe(n)
        registry.maybe_promote(n, stable_sketch(n, dim=SKETCH_DIM))
    axes = list(registry.promoted.values())
    for i in range(len(axes)):
        for j in range(i + 1, len(axes)):
            cos = float(torch.dot(axes[i].axis, axes[j].axis).item())
            assert (
                abs(cos) < 1e-4
            ), f"axes {axes[i].name} and {axes[j].name} not orthogonal: cos={cos}"


def test_vector_for_returns_promoted_axis_after_promotion():
    registry = OntologicalRegistry(dim=SKETCH_DIM, frequency_threshold=2)
    base = stable_sketch("project_x", dim=SKETCH_DIM)
    v_pre = registry.vector_for("project_x", base)
    cos_pre = float(torch.dot(v_pre, base / base.norm()).item())
    assert cos_pre > 0.99  # before promotion the vector is the normalized sketch

    for _ in range(3):
        registry.observe("project_x")
        registry.maybe_promote("project_x", base)
    v_post = registry.vector_for("project_x", base)
    # After promotion the registry returns the dedicated axis directly.
    assert torch.allclose(v_post, registry.promoted["project_x"].axis)


def test_persistence_round_trip(tmp_path: Path):
    registry = OntologicalRegistry(dim=SKETCH_DIM, frequency_threshold=1)
    registry.observe("ada")
    registry.maybe_promote("ada", stable_sketch("ada", dim=SKETCH_DIM))
    store = PersistentOntologicalRegistry(tmp_path / "ont.sqlite", namespace="t")
    store.save(registry)

    loaded = store.load(dim=SKETCH_DIM, frequency_threshold=1)
    assert "ada" in loaded.promoted
    assert torch.allclose(
        loaded.promoted["ada"].axis, registry.promoted["ada"].axis, atol=1e-6
    )
