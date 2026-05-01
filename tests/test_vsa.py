from __future__ import annotations

import math

import torch

from asi_broca_core.vsa import (
    DEFAULT_VSA_DIM,
    VSACodebook,
    bind,
    bundle,
    cleanup,
    cosine,
    hypervector,
    permute,
    unbind,
)


def test_hypervector_is_deterministic_and_unit_norm():
    a = hypervector("ada", dim=4096, base_seed=7)
    b = hypervector("ada", dim=4096, base_seed=7)
    assert torch.allclose(a, b, atol=0.0)
    assert abs(float(a.norm().item()) - 1.0) < 1e-5


def test_independent_atoms_are_quasi_orthogonal_in_high_d():
    dim = 8192
    atoms = [hypervector(f"atom_{i}", dim=dim, base_seed=0) for i in range(32)]
    cosines = []
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            cosines.append(abs(cosine(atoms[i], atoms[j])))
    avg = sum(cosines) / len(cosines)
    # Expectation is ~0; std ~ 1/sqrt(d). At d=8192 the average abs-cos must
    # comfortably stay under 0.05.
    assert avg < 0.05, f"avg |cos|={avg:.4f} too large for d={dim}"


def test_bind_unbind_round_trip_recovers_filler():
    dim = 8192
    role = hypervector("ROLE_OBJECT", dim=dim)
    filler = hypervector("london", dim=dim)
    encoded = bind(role, filler)
    recovered = unbind(encoded, role)
    cos = cosine(recovered, filler)
    assert cos > 0.95, f"unbind recovered cos={cos:.4f}, expected > 0.95"


def test_bundle_preserves_constituents_above_capacity_floor():
    dim = 8192
    items = [hypervector(f"x_{i}", dim=dim) for i in range(8)]
    bundled = bundle(items)
    cosines = [cosine(bundled, x) for x in items]
    # All constituents must remain detectable above the noise floor.
    for c in cosines:
        assert c > 0.1, f"bundle dropped a constituent: cos={c:.4f}"


def test_permute_is_invertible_and_distinct():
    v = hypervector("seq", dim=4096)
    p = permute(v, shift=1)
    # Permuted vector should be quasi-orthogonal to the original.
    assert abs(cosine(v, p)) < 0.1
    # Inverse shift restores it exactly.
    inv = permute(p, shift=-1)
    assert torch.allclose(v, inv, atol=0.0)


def test_codebook_round_trips_triple():
    book = VSACodebook(dim=8192, base_seed=11)
    encoded = book.encode_triple("ada", "works in", "london")
    obj_name, obj_cos = book.decode_role(encoded, "ROLE_OBJECT", candidates=["paris", "rome", "london", "berlin"])
    assert obj_name == "london", f"decoded role got {obj_name!r} instead of 'london' (cos={obj_cos:.4f})"
    assert obj_cos > 0.4
    subj_name, _ = book.decode_role(encoded, "ROLE_SUBJECT", candidates=["alan", "ada", "babbage"])
    assert subj_name == "ada"


def test_cleanup_chooses_closest_atom_under_noise():
    book = VSACodebook(dim=8192)
    target = book.atom("london")
    rng = torch.Generator()
    rng.manual_seed(0)
    noise = torch.empty_like(target).normal_(0.0, 0.05, generator=rng)
    noisy = target + noise
    name, cos = cleanup(noisy, {"paris": book.atom("paris"), "london": book.atom("london"), "rome": book.atom("rome")})
    assert name == "london"
    assert cos > 0.7


def test_default_vsa_dim_is_used_when_unspecified():
    v = hypervector("anything")
    assert v.shape == (DEFAULT_VSA_DIM,)
