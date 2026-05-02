from __future__ import annotations

import torch

from core.memory.hopfield import (
    HopfieldAssociativeMemory,
    derived_inverse_temperature,
    hopfield_update,
)


def _orthonormal_basis(d: int, n: int, seed: int = 0) -> torch.Tensor:
    if n > d:
        raise ValueError(
            f"_orthonormal_basis requires n <= d for a full QR factorization, got n={n}, d={d}"
        )
    g = torch.Generator()
    g.manual_seed(seed)
    raw = torch.empty(n, d).normal_(0.0, 1.0, generator=g)
    q, _ = torch.linalg.qr(raw.t())
    return q.t()  # n x d, rows orthonormal


def test_one_step_update_collapses_onto_nearest_pattern():
    d = 64
    keys = _orthonormal_basis(d, 4, seed=11)
    values = keys.clone()  # autoassociative
    target = keys[2].clone()
    noisy = target + 0.05 * torch.randn(d, generator=torch.Generator().manual_seed(7))
    retrieved, weights, energy = hopfield_update(
        noisy, keys, values, beta=20.0, iterations=1
    )
    cos = float(torch.nn.functional.cosine_similarity(retrieved, target, dim=0).item())
    assert cos > 0.99, f"expected near-perfect recovery, got cos={cos:.4f}"
    # Weight should be sharply concentrated on index 2.
    assert int(weights.argmax().item()) == 2
    assert float(weights.max().item()) > 0.9


def test_iterated_update_converges_to_fixed_point():
    d = 32
    keys = _orthonormal_basis(d, 3, seed=3)
    values = keys.clone()
    query = keys[0] + 0.4 * torch.randn(d, generator=torch.Generator().manual_seed(2))
    state, _, _ = hopfield_update(query, keys, values, beta=15.0, iterations=5)
    state2, _, _ = hopfield_update(state, keys, values, beta=15.0, iterations=1)
    delta = float((state - state2).norm().item())
    assert delta < 1e-5, f"iterate did not converge: ||x_t - x_{{t+1}}||={delta:.6f}"


def test_derived_inverse_temperature_scales_with_dim():
    d = 128
    keys = torch.randn(50, d, generator=torch.Generator().manual_seed(0))
    beta = derived_inverse_temperature(keys)
    # Should be in the rough order of √d / σ; with σ ≈ 1, β ≈ √128 ≈ 11.3
    assert 5.0 < beta < 30.0


def test_associative_memory_basic_remember_and_retrieve():
    mem = HopfieldAssociativeMemory(d_model=16)
    keys = _orthonormal_basis(16, 3, seed=5)
    values = keys.clone()
    for k, v in zip(keys, values):
        mem.remember(k, v, metadata={"k": True})
    target = keys[1]
    noisy = target + 0.05 * torch.randn(16, generator=torch.Generator().manual_seed(3))
    retrieved, weights = mem.retrieve(noisy, beta=15.0, iterations=1)
    cos = float(torch.nn.functional.cosine_similarity(retrieved, target, dim=0).item())
    assert cos > 0.95
    assert mem.last_debug["n"] == 3
    assert mem.last_debug["weight_max"] > 0.5


def test_empty_memory_returns_zeros():
    mem = HopfieldAssociativeMemory(d_model=8)
    retrieved, weights = mem.retrieve(torch.randn(8))
    assert torch.equal(retrieved, torch.zeros(8))
    assert weights.numel() == 0


def test_high_load_retrieval_accuracy():
    """Sanity-check Theorem 3: with β well above √d, dense storage still recovers."""

    d = 128
    n = 200
    keys = torch.randn(n, d, generator=torch.Generator().manual_seed(1))
    keys = keys / keys.norm(dim=-1, keepdim=True)
    values = keys.clone()
    target_idx = 42
    query = keys[target_idx].clone() + 0.01 * torch.randn(
        d, generator=torch.Generator().manual_seed(11)
    )
    retrieved, _, _ = hopfield_update(query, keys, values, beta=50.0, iterations=2)
    cos = float(
        torch.nn.functional.cosine_similarity(retrieved, keys[target_idx], dim=0).item()
    )
    assert cos > 0.85, f"high-β retrieval failed at n=200: cos={cos:.4f}"
