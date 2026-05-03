from __future__ import annotations

import math

import pytest
import torch

from core.calibration.recursion_halt import DEFAULT_MAX_ROUNDS, RecursionHalt
from core.swm import SubstrateWorkingMemory, SWMSource


def _orthogonal_pair(dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(0)
    a = torch.empty(dim, dtype=torch.float32)
    b = torch.empty(dim, dtype=torch.float32)
    a.normal_(mean=0.0, std=1.0 / math.sqrt(dim), generator=g)
    b.normal_(mean=0.0, std=1.0 / math.sqrt(dim), generator=g)
    return a / a.norm(), b / b.norm()


def test_default_max_rounds_is_3():
    assert DEFAULT_MAX_ROUNDS == 3


def test_first_check_never_halts_via_convergence():
    swm = SubstrateWorkingMemory()
    halt = RecursionHalt(swm=swm)
    swm.write("thought", torch.randn(swm.dim), source=SWMSource.SUBSTRATE_ALGEBRA)
    decision = halt.check(slot_name="thought", rounds_completed=1)
    assert decision.halt is False
    assert decision.cosine_to_previous == float("-inf")


def test_identical_consecutive_thought_halts_via_convergence():
    swm = SubstrateWorkingMemory()
    halt = RecursionHalt(swm=swm)
    v = torch.randn(swm.dim)
    swm.write("thought", v, source=SWMSource.SUBSTRATE_ALGEBRA)
    halt.check(slot_name="thought", rounds_completed=1)
    swm.write("thought", v, source=SWMSource.SUBSTRATE_ALGEBRA)
    decision = halt.check(slot_name="thought", rounds_completed=2)
    assert decision.halt is True
    assert decision.reason == "converged"
    assert decision.cosine_to_previous > 0.99


def test_reaching_max_rounds_halts():
    swm = SubstrateWorkingMemory()
    halt = RecursionHalt(swm=swm, max_rounds=2)
    swm.write("thought", torch.randn(swm.dim), source=SWMSource.SUBSTRATE_ALGEBRA)
    halt.check(slot_name="thought", rounds_completed=1)
    swm.write("thought", torch.randn(swm.dim), source=SWMSource.SUBSTRATE_ALGEBRA)
    decision = halt.check(slot_name="thought", rounds_completed=2)
    assert decision.halt is True
    assert decision.reason == "max_rounds_reached"


def test_reset_clears_history():
    swm = SubstrateWorkingMemory()
    halt = RecursionHalt(swm=swm)
    swm.write("thought", torch.randn(swm.dim), source=SWMSource.SUBSTRATE_ALGEBRA)
    halt.check(slot_name="thought", rounds_completed=1)
    halt.reset()
    swm.write("thought", torch.randn(swm.dim), source=SWMSource.SUBSTRATE_ALGEBRA)
    decision = halt.check(slot_name="thought", rounds_completed=1)
    assert decision.cosine_to_previous == float("-inf")


def test_orthogonal_thoughts_do_not_halt():
    swm = SubstrateWorkingMemory()
    halt = RecursionHalt(swm=swm)
    a, b = _orthogonal_pair(swm.dim)
    swm.write("thought", a, source=SWMSource.SUBSTRATE_ALGEBRA)
    halt.check(slot_name="thought", rounds_completed=1)
    swm.write("thought", b, source=SWMSource.SUBSTRATE_ALGEBRA)
    decision = halt.check(slot_name="thought", rounds_completed=2)
    assert decision.halt is False, f"orthogonal thoughts should not converge, cos={decision.cosine_to_previous:.4f}"


def test_rejects_non_positive_max_rounds():
    swm = SubstrateWorkingMemory()
    with pytest.raises(ValueError):
        RecursionHalt(swm=swm, max_rounds=0)


def test_convergence_floor_scales_with_dim():
    swm_small = SubstrateWorkingMemory(dim=1024)
    swm_large = SubstrateWorkingMemory(dim=10_000)
    h_small = RecursionHalt(swm=swm_small)
    h_large = RecursionHalt(swm=swm_large)
    # Larger dim → tighter quasi-orthogonality → higher floor required for "converged".
    assert h_large.convergence_floor > h_small.convergence_floor
