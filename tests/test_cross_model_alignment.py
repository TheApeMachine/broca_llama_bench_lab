from __future__ import annotations

import torch

from core.grafting.alignment import CrossModelAlignment


def _synthetic_embedding(*, vocab: int, dim: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    w = torch.empty(int(vocab), int(dim), dtype=torch.float32)
    w.normal_(mean=0.0, std=1.0 / float(dim) ** 0.5, generator=g)
    return w


def test_changes_dim_from_d_a_to_d_b():
    w_out_a = _synthetic_embedding(vocab=512, dim=128, seed=11)
    w_in_b = _synthetic_embedding(vocab=512, dim=64, seed=22)
    cross = CrossModelAlignment(name="A_to_B", w_out_source=w_out_a, w_in_target=w_in_b)
    h = torch.randn(3, 4, 128)
    e = cross.apply(h)
    assert e.shape == (3, 4, 64)


def test_rejects_2d_inputs_only():
    w = _synthetic_embedding(vocab=512, dim=64, seed=0)
    bad = w.view(8, 64, 64)
    try:
        CrossModelAlignment(name="bad", w_out_source=bad, w_in_target=w)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for non-2D matrix")


def test_truncates_to_shared_vocab_prefix():
    """Different vocab sizes are truncated to the shared prefix; no silent extension."""

    w_out_a = _synthetic_embedding(vocab=512, dim=64, seed=11)
    w_in_b = _synthetic_embedding(vocab=300, dim=32, seed=22)  # smaller V
    cross = CrossModelAlignment(name="A_to_B", w_out_source=w_out_a, w_in_target=w_in_b)
    assert cross.matrix.shape == (64, 32)


def test_recovers_target_input_when_source_equals_target():
    """If A = B (same model, same shape), CrossModelAlignment reduces to RidgeAlignment."""

    w = _synthetic_embedding(vocab=512, dim=64, seed=11)
    cross = CrossModelAlignment(name="self", w_out_source=w, w_in_target=w)
    eye = torch.eye(64, dtype=torch.float32)
    diff = (cross.matrix - eye).abs().max().item()
    assert diff < 1e-3, f"self-cross-alignment should be identity, max deviation {diff:.4f}"
