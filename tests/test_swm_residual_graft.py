from __future__ import annotations

import pytest
import torch

from core.grafting.alignment import SWMToInputProjection
from core.grafts.swm_residual_graft import SWMResidualGraft, SWM_INJECT_SLOT_KEY
from core.swm import SubstrateWorkingMemory, SWMSource


def _projection(*, d_swm: int, d_target: int, vocab: int = 256, seed: int = 0) -> SWMToInputProjection:
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = torch.empty(vocab, d_target, dtype=torch.float32)
    w.normal_(mean=0.0, std=1.0 / d_target ** 0.5, generator=g)
    return SWMToInputProjection(name="t", d_swm=d_swm, w_in_target=w, seed=seed)


def test_no_slot_means_no_change():
    swm = SubstrateWorkingMemory()
    proj = _projection(d_swm=swm.dim, d_target=64)
    g = SWMResidualGraft(swm=swm, projection=proj)
    x = torch.randn(2, 5, 64)
    state = {"last_indices": torch.tensor([4, 4])}
    out = g(x, state)
    assert torch.allclose(out, x)


def test_slot_injection_only_modifies_last_position():
    swm = SubstrateWorkingMemory()
    proj = _projection(d_swm=swm.dim, d_target=64)
    g = SWMResidualGraft(swm=swm, projection=proj)
    swm.write("active", torch.randn(swm.dim), source=SWMSource.SUBSTRATE_ALGEBRA)

    x = torch.randn(1, 4, 64)
    state = {SWM_INJECT_SLOT_KEY: "active", "last_indices": torch.tensor([3])}
    out = g(x, state)

    # Positions 0..2 are untouched; position 3 differs.
    assert torch.allclose(out[:, :3], x[:, :3])
    assert not torch.allclose(out[:, 3], x[:, 3])


def test_missing_last_indices_raises():
    swm = SubstrateWorkingMemory()
    proj = _projection(d_swm=swm.dim, d_target=64)
    g = SWMResidualGraft(swm=swm, projection=proj)
    swm.write("active", torch.randn(swm.dim), source=SWMSource.LLAMA)
    with pytest.raises(RuntimeError, match="last_indices"):
        g(torch.randn(1, 3, 64), {SWM_INJECT_SLOT_KEY: "active"})


def test_residual_dim_mismatch_raises():
    swm = SubstrateWorkingMemory()
    proj = _projection(d_swm=swm.dim, d_target=64)
    g = SWMResidualGraft(swm=swm, projection=proj)
    swm.write("active", torch.randn(swm.dim), source=SWMSource.LLAMA)
    with pytest.raises(ValueError, match="residual"):
        g(torch.randn(1, 3, 32), {SWM_INJECT_SLOT_KEY: "active", "last_indices": torch.tensor([2])})


def test_unknown_slot_raises():
    swm = SubstrateWorkingMemory()
    proj = _projection(d_swm=swm.dim, d_target=64)
    g = SWMResidualGraft(swm=swm, projection=proj)
    with pytest.raises(KeyError):
        g(torch.randn(1, 3, 64), {SWM_INJECT_SLOT_KEY: "ghost", "last_indices": torch.tensor([2])})


def test_disabled_graft_is_pass_through():
    swm = SubstrateWorkingMemory()
    proj = _projection(d_swm=swm.dim, d_target=64)
    g = SWMResidualGraft(swm=swm, projection=proj)
    g.enabled = False
    swm.write("active", torch.randn(swm.dim), source=SWMSource.LLAMA)

    x = torch.randn(1, 4, 64)
    out = g(x, {SWM_INJECT_SLOT_KEY: "active", "last_indices": torch.tensor([3])})
    assert torch.allclose(out, x)
