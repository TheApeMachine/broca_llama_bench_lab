from __future__ import annotations

import pytest
import torch

from core.swm import SubstrateWorkingMemory, SWMSource
from core.symbolic import DEFAULT_VSA_DIM, hypervector


def _atom(name: str) -> torch.Tensor:
    return hypervector(name, dim=DEFAULT_VSA_DIM, base_seed=0)


def test_default_dim_matches_vsa_dim():
    swm = SubstrateWorkingMemory()
    assert swm.dim == DEFAULT_VSA_DIM == 10_000


def test_write_and_read_round_trips():
    swm = SubstrateWorkingMemory()
    v = _atom("ada")
    swm.write("subject", v, source=SWMSource.GLINER2)
    slot = swm.read("subject")
    assert slot.name == "subject"
    assert slot.source is SWMSource.GLINER2
    assert torch.allclose(slot.vector, v.float())


def test_write_increments_tick():
    swm = SubstrateWorkingMemory()
    a = swm.write("a", _atom("a"), source=SWMSource.LLAMA)
    b = swm.write("b", _atom("b"), source=SWMSource.LLAMA)
    assert b.written_at_tick == a.written_at_tick + 1


def test_overwrite_replaces_slot():
    swm = SubstrateWorkingMemory()
    swm.write("x", _atom("x"), source=SWMSource.LLAMA)
    swm.write("x", _atom("y"), source=SWMSource.GLICLASS)
    slot = swm.read("x")
    assert slot.source is SWMSource.GLICLASS
    assert torch.allclose(slot.vector, _atom("y").float())


def test_missing_slot_raises():
    swm = SubstrateWorkingMemory()
    with pytest.raises(KeyError):
        swm.read("nope")


def test_dim_mismatch_raises():
    swm = SubstrateWorkingMemory()
    with pytest.raises(ValueError):
        swm.write("bad", torch.randn(7), source=SWMSource.LLAMA)


def test_bind_then_unbind_round_trips_via_swm_api():
    swm = SubstrateWorkingMemory()
    swm.write("ROLE_OBJECT", _atom("ROLE_OBJECT"), source=SWMSource.SUBSTRATE_ALGEBRA)
    swm.write("london", _atom("london"), source=SWMSource.SUBSTRATE_ALGEBRA)
    swm.bind_slots("ROLE_OBJECT", "london", into="bound")
    swm.unbind_slots("bound", "ROLE_OBJECT", into="recovered")

    london = swm.read("london").vector
    recovered = swm.read("recovered").vector
    cos = torch.nn.functional.cosine_similarity(
        recovered.view(1, -1), london.view(1, -1), dim=1
    ).item()
    assert cos > 0.65, f"unbind via SWM lost the filler, cos={cos:.4f}"


def test_bundle_slots_keeps_all_constituents_above_floor():
    swm = SubstrateWorkingMemory()
    names = [f"x_{i}" for i in range(8)]
    for n in names:
        swm.write(n, _atom(n), source=SWMSource.SUBSTRATE_ALGEBRA)
    swm.bundle_slots(names, into="bag")
    bag = swm.read("bag").vector
    for n in names:
        cos = torch.nn.functional.cosine_similarity(
            bag.view(1, -1), swm.read(n).vector.view(1, -1), dim=1
        ).item()
        assert cos > 0.1, f"bundle dropped constituent {n!r}, cos={cos:.4f}"


def test_cleanup_to_slot_finds_nearest():
    swm = SubstrateWorkingMemory()
    for n in ("paris", "london", "rome", "berlin"):
        swm.write(n, _atom(n), source=SWMSource.SUBSTRATE_ALGEBRA)
    london = swm.read("london").vector
    name, cos = swm.cleanup_to_slot(london)
    assert name == "london"
    assert cos > 0.95


def test_iteration_visits_every_slot():
    swm = SubstrateWorkingMemory()
    for n in ("a", "b", "c"):
        swm.write(n, _atom(n), source=SWMSource.LLAMA)
    visited = sorted(slot.name for slot in swm)
    assert visited == ["a", "b", "c"]


def test_remove_purges_slot():
    swm = SubstrateWorkingMemory()
    swm.write("ephemeral", _atom("ephemeral"), source=SWMSource.LLAMA)
    swm.remove("ephemeral")
    assert not swm.has("ephemeral")
    with pytest.raises(KeyError):
        swm.remove("ephemeral")
