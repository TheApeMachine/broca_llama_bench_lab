from __future__ import annotations

import torch

from core.grafting.alignment import RidgeAlignment


def _synthetic_embedding(*, vocab: int, dim: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    w = torch.empty(int(vocab), int(dim), dtype=torch.float32)
    w.normal_(mean=0.0, std=1.0 / float(dim) ** 0.5, generator=g)
    return w


def test_tied_embeddings_yield_identity_alignment():
    """Llama-3.2 has tied embeddings; W_a = pinv(W) @ W must be the identity."""
    w = _synthetic_embedding(vocab=512, dim=64, seed=11)
    align = RidgeAlignment(name="tied", w_in=w, w_out=w)
    eye = torch.eye(64, dtype=torch.float32)
    diff = (align.matrix - eye).abs().max().item()
    assert diff < 1e-3, f"expected ≈identity for tied embeddings, max abs deviation {diff:.4f}"


def test_alignment_application_changes_dim_correctly():
    w = _synthetic_embedding(vocab=512, dim=64, seed=11)
    align = RidgeAlignment(name="tied", w_in=w, w_out=w)
    h = torch.randn(3, 5, 64)
    e = align.apply(h)
    assert e.shape == (3, 5, 64)


def test_alignment_output_lives_in_input_embedding_subspace():
    """Wa is constructed so h @ Wa = c @ W_in for c = h @ pinv(W_out).

    Every output is therefore a linear combination of W_in's rows by
    construction — i.e. it lives in the input embedding span. This is the
    structural property that lets latent rollout feed the result back as an
    embedding.
    """

    w_in = _synthetic_embedding(vocab=512, dim=64, seed=11)
    w_out = _synthetic_embedding(vocab=512, dim=64, seed=22)
    align = RidgeAlignment(name="untied", w_in=w_in, w_out=w_out)

    # Project a random hidden batch through W_a, then verify the result is
    # already invariant under the orthogonal projector onto row-space of W_in.
    h = torch.randn(16, 64)
    e = align.apply(h)
    p_in = torch.linalg.pinv(w_in.to(torch.float64)) @ w_in.to(torch.float64)
    e_projected = e.to(torch.float64) @ p_in
    drift = (e.to(torch.float64) - e_projected).norm() / e.to(torch.float64).norm().clamp_min(1e-12)
    assert drift.item() < 1e-3, (
        f"alignment output drifted out of input subspace, relative error {drift.item():.6f}"
    )


def test_application_is_linear():
    w = _synthetic_embedding(vocab=512, dim=64, seed=11)
    align = RidgeAlignment(name="tied", w_in=w, w_out=w)
    h1 = torch.randn(64)
    h2 = torch.randn(64)
    a, b = 1.7, -0.4
    lhs = align.apply((a * h1 + b * h2).view(1, -1)).view(-1)
    rhs = a * align.apply(h1.view(1, -1)).view(-1) + b * align.apply(h2.view(1, -1)).view(-1)
    diff = (lhs - rhs).abs().max().item()
    assert diff < 1e-4, f"alignment is not linear, max diff {diff:.6f}"


def test_rejects_mismatched_shapes():
    w_in = _synthetic_embedding(vocab=512, dim=64, seed=11)
    w_out = _synthetic_embedding(vocab=512, dim=128, seed=22)
    try:
        RidgeAlignment(name="bad", w_in=w_in, w_out=w_out)
    except ValueError as exc:
        assert "shape" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError for mismatched shapes")


def test_rejects_underdetermined_vocab():
    """V < d means W_out cannot be left-inverted on the d-dim subspace."""

    w_in = _synthetic_embedding(vocab=8, dim=64, seed=11)
    w_out = _synthetic_embedding(vocab=8, dim=64, seed=22)
    try:
        RidgeAlignment(name="bad", w_in=w_in, w_out=w_out)
    except ValueError as exc:
        assert "vocab" in str(exc).lower() or "v >= d" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError when V < d")
