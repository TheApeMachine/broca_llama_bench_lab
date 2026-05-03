from __future__ import annotations

import pytest
import torch

from core.grafting.alignment import AlignmentRegistry, RidgeAlignment


def _embed(*, vocab: int, dim: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    w = torch.empty(int(vocab), int(dim), dtype=torch.float32)
    w.normal_(mean=0.0, std=1.0 / float(dim) ** 0.5, generator=g)
    return w


def test_register_and_get_round_trip():
    reg = AlignmentRegistry()
    w = _embed(vocab=128, dim=32, seed=1)
    a = RidgeAlignment(name="organ.self", w_in=w, w_out=w)
    reg.register(a)
    assert reg.get("organ.self") is a


def test_duplicate_register_raises():
    reg = AlignmentRegistry()
    w = _embed(vocab=128, dim=32, seed=1)
    a = RidgeAlignment(name="dup", w_in=w, w_out=w)
    reg.register(a)
    b = RidgeAlignment(name="dup", w_in=w, w_out=w)
    with pytest.raises(ValueError):
        reg.register(b)


def test_missing_name_raises():
    reg = AlignmentRegistry()
    with pytest.raises(KeyError):
        reg.get("nonexistent")


def test_has_and_names_and_iter():
    reg = AlignmentRegistry()
    w = _embed(vocab=128, dim=32, seed=1)
    reg.register(RidgeAlignment(name="a", w_in=w, w_out=w))
    reg.register(RidgeAlignment(name="b", w_in=w, w_out=w))
    assert reg.has("a") and reg.has("b")
    assert sorted(reg.names()) == ["a", "b"]
    assert sorted(x.name for x in reg) == ["a", "b"]
    assert len(reg) == 2
