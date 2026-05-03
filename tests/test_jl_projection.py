from __future__ import annotations

import pytest
import torch

from core.swm import JLProjection


def test_changes_dim_and_preserves_batch_shape():
    p = JLProjection(name="up", d_in=64, d_out=10_000, seed=0)
    x = torch.randn(3, 5, 64)
    y = p.apply(x)
    assert y.shape == (3, 5, 10_000)


def test_deterministic_under_seed():
    a = JLProjection(name="a", d_in=64, d_out=10_000, seed=42)
    b = JLProjection(name="b", d_in=64, d_out=10_000, seed=42)
    assert torch.allclose(a.matrix, b.matrix, atol=0.0)


def test_distinct_seeds_diverge():
    a = JLProjection(name="a", d_in=64, d_out=10_000, seed=1)
    b = JLProjection(name="b", d_in=64, d_out=10_000, seed=2)
    diff = (a.matrix - b.matrix).abs().mean().item()
    assert diff > 1e-4


def test_jl_preserves_pairwise_inner_products_in_expectation():
    """JL preserves cosine within tight bounds at d_out >> d_in."""

    p = JLProjection(name="p", d_in=128, d_out=10_000, seed=0)

    g = torch.Generator(device="cpu").manual_seed(123)
    s1 = torch.empty(128, dtype=torch.float32)
    s2 = torch.empty(128, dtype=torch.float32)
    s1.normal_(mean=0.0, std=1.0, generator=g)
    s2.normal_(mean=0.0, std=1.0, generator=g)

    cos_in = torch.nn.functional.cosine_similarity(s1.view(1, -1), s2.view(1, -1), dim=1).item()
    e1 = p.apply(s1.view(1, -1)).view(-1)
    e2 = p.apply(s2.view(1, -1)).view(-1)
    cos_out = torch.nn.functional.cosine_similarity(e1.view(1, -1), e2.view(1, -1), dim=1).item()

    assert abs(cos_out - cos_in) < 0.1, (
        f"JL distortion too large: input cosine {cos_in:.4f}, output cosine {cos_out:.4f}"
    )


def test_rejects_dim_mismatch():
    p = JLProjection(name="p", d_in=64, d_out=128, seed=0)
    with pytest.raises(ValueError):
        p.apply(torch.randn(3, 65))
