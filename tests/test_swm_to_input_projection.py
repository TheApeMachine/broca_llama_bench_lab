from __future__ import annotations

import torch

from core.grafting.alignment import SWMToInputProjection


def _synthetic_embedding(*, vocab: int, dim: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    w = torch.empty(int(vocab), int(dim), dtype=torch.float32)
    w.normal_(mean=0.0, std=1.0 / float(dim) ** 0.5, generator=g)
    return w


def test_changes_dim_from_swm_to_target():
    w_in = _synthetic_embedding(vocab=512, dim=64, seed=11)
    proj = SWMToInputProjection(name="swm_to_llama", d_swm=10_000, w_in_target=w_in, seed=7)
    s = torch.randn(2, 3, 10_000)
    e = proj.apply(s)
    assert e.shape == (2, 3, 64)


def test_deterministic_under_seed():
    w_in = _synthetic_embedding(vocab=512, dim=64, seed=11)
    p1 = SWMToInputProjection(name="a", d_swm=10_000, w_in_target=w_in, seed=42)
    p2 = SWMToInputProjection(name="b", d_swm=10_000, w_in_target=w_in, seed=42)
    assert torch.allclose(p1.matrix, p2.matrix, atol=0.0)


def test_distinct_seeds_diverge():
    w_in = _synthetic_embedding(vocab=512, dim=64, seed=11)
    p1 = SWMToInputProjection(name="a", d_swm=10_000, w_in_target=w_in, seed=1)
    p2 = SWMToInputProjection(name="b", d_swm=10_000, w_in_target=w_in, seed=2)
    diff = (p1.matrix - p2.matrix).abs().mean().item()
    assert diff > 1e-4, f"distinct seeds produced near-identical matrices, mean diff {diff:.6f}"


def test_jl_preserves_pairwise_inner_products_in_expectation():
    """Johnson-Lindenstrauss preserves cosines up to vanishing distortion at D_swm=10k."""

    w_in = _synthetic_embedding(vocab=4096, dim=128, seed=11)
    proj = SWMToInputProjection(name="p", d_swm=10_000, w_in_target=w_in, seed=0)

    g = torch.Generator(device="cpu").manual_seed(123)
    s1 = torch.empty(10_000, dtype=torch.float32)
    s2 = torch.empty(10_000, dtype=torch.float32)
    s1.normal_(mean=0.0, std=1.0 / 100.0, generator=g)
    s2.normal_(mean=0.0, std=1.0 / 100.0, generator=g)

    cos_in = torch.nn.functional.cosine_similarity(s1.view(1, -1), s2.view(1, -1), dim=1).item()
    e1 = proj.apply(s1.view(1, -1)).view(-1)
    e2 = proj.apply(s2.view(1, -1)).view(-1)
    cos_out = torch.nn.functional.cosine_similarity(e1.view(1, -1), e2.view(1, -1), dim=1).item()

    # JL distortion at D_swm=10k -> d=128 with sub-Gaussian sources is small.
    assert abs(cos_out - cos_in) < 0.1, (
        f"JL distortion too large: input cosine {cos_in:.4f}, output cosine {cos_out:.4f}"
    )
