"""Vector Symbolic Architecture / Hyperdimensional Computing.

Implements Holographic Reduced Representations (HRR) — Plate's classical VSA —
on top of PyTorch. HRR uses circular convolution for binding (so binding is
commutative and associative) and circular correlation for unbinding, with
elementwise sum for bundling and tensor roll for permutation. Hypervectors are
seeded deterministically per atom name so the same atom yields the same vector
across processes, which is what makes algebra over a persistent substrate
possible.

Why this implementation:

* HRR is the only VSA flavor where binding has an exact inverse (circular
  correlation) and where vectors stay in the same domain. That is what
  unlocks zero-shot analogy and fact unbinding without re-encoding.
* Circular convolution is computed via FFT in O(d log d) — fast enough to call
  per utterance even at d = 10 000.
* Bundling preserves direction so a noisy bundle of N vectors still has high
  cosine with each constituent (capacity ~ 0.5 * d / log d for HRR).

Reference: Plate, T. A. (1995). *Holographic Reduced Representations*. IEEE
Transactions on Neural Networks 6(3).
"""

from __future__ import annotations

import logging
import math
import threading
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_VSA_DIM",
    "VSACodebook",
    "bind",
    "bundle",
    "cleanup",
    "cosine",
    "hypervector",
    "permute",
    "unbind",
]

DEFAULT_VSA_DIM = 10_000


def _atom_seed(name: str, *, base_seed: int) -> int:
    """Deterministic 64-bit seed for an atom name, salted by ``base_seed``."""
    h = (
        0xCBF29CE484222325 ^ ((int(base_seed) + 1) * 0x9E3779B185EBCA87)
    ) & 0xFFFFFFFFFFFFFFFF

    for byte in str(name).encode("utf-8", errors="replace"):
        h ^= int(byte)
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    
    # torch.Generator.manual_seed wants a non-negative int <= 2**63-1.
    return int(h & 0x7FFFFFFFFFFFFFFF)


def hypervector(
    name: str,
    *,
    dim: int = DEFAULT_VSA_DIM,
    base_seed: int = 0,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Generate a unit-norm Gaussian hypervector deterministically from ``name``.

    Two calls with the same ``(name, dim, base_seed)`` triple return identical
    vectors so the substrate's algebra is reproducible across runs and across
    machines. Atoms drawn from independent names are mutually quasi-orthogonal
    in expectation (cos ≈ 0, with std ≈ 1/√dim).
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(_atom_seed(name, base_seed=base_seed))
    v = torch.empty(int(dim), dtype=torch.float32)
    v.normal_(mean=0.0, std=1.0 / math.sqrt(float(dim)), generator=g)
    v = v / v.norm().clamp_min(1e-12)
    
    if device is not None:
        v = v.to(device=device)
    
    return v.to(dtype=dtype)


def bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Circular convolution binding (commutative, associative, exactly invertible)."""

    if a.shape != b.shape:
        raise ValueError(
            f"VSA bind requires matching shapes, got {a.shape} vs {b.shape}"
        )
    
    out_dtype = torch.promote_types(a.dtype, b.dtype)
    compute_dtype = torch.promote_types(out_dtype, torch.float32)
    
    aa = a.to(compute_dtype)
    bb = b.to(compute_dtype)
    fa = torch.fft.rfft(aa)
    fb = torch.fft.rfft(bb)
    raw = torch.fft.irfft(fa * fb, n=a.shape[-1])
    
    target_dtype = out_dtype if out_dtype.is_floating_point else compute_dtype
    
    return raw.to(target_dtype)


def unbind(c: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Recover ``b`` from ``c = bind(a, b)`` using circular correlation.

    Unbinding by ``a`` is convolution with ``a*`` (the time-reversed sequence,
    or equivalently the complex conjugate in frequency space). For unit-norm
    Gaussian atoms this is also the *inverse* of binding: ``unbind(bind(a, b),
    a) ≈ b`` up to noise that vanishes as dim → ∞.
    """
    if c.shape != a.shape:
        raise ValueError(
            f"VSA unbind requires matching shapes, got {c.shape} vs {a.shape}"
        )

    out_dtype = torch.promote_types(c.dtype, a.dtype)
    compute_dtype = torch.promote_types(out_dtype, torch.float32)

    cc = c.to(compute_dtype)
    aa = a.to(compute_dtype)
    fc = torch.fft.rfft(cc)
    fa = torch.fft.rfft(aa)
    raw = torch.fft.irfft(fc * fa.conj(), n=c.shape[-1])

    target_dtype = out_dtype if out_dtype.is_floating_point else compute_dtype

    return raw.to(target_dtype)


def bundle(vectors: Iterable[torch.Tensor], *, normalize: bool = True) -> torch.Tensor:
    """Bundle (superposition) via elementwise sum, optionally renormalized.

    Cosine similarity between each constituent and the bundle stays positive
    until capacity (~0.5 * d / log d) is exceeded, after which crosstalk
    dominates and clean unbinding fails.
    """
    items = list(vectors)
    
    if not items:
        raise ValueError("bundle expects at least one vector")
    
    common_dtype = items[0].dtype
    
    for i, v in enumerate(items[1:], start=1):
        common_dtype = torch.promote_types(common_dtype, v.dtype)
    
    out = torch.stack([v.to(torch.float32) for v in items], dim=0).sum(dim=0)
    
    if normalize:
        out = out / out.norm().clamp_min(1e-12)
    
    return out.to(dtype=common_dtype)


def permute(v: torch.Tensor, *, shift: int = 1) -> torch.Tensor:
    """Cyclic shift by ``shift`` positions; encodes sequence order."""
    return torch.roll(v, shifts=int(shift), dims=-1)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity computed and returned as a Python float."""
    return float(
        F.cosine_similarity(
            a.view(-1).to(torch.float32), b.view(-1).to(torch.float32), dim=0
        ).item()
    )


def cleanup(
    query: torch.Tensor, codebook: Mapping[str, torch.Tensor]
) -> tuple[str, float]:
    """Snap a noisy hypervector back to the nearest codebook atom by cosine."""
    if not codebook:
        raise ValueError("cleanup requires a non-empty codebook")

    keys = list(codebook.keys())
    q = query.view(-1).to(torch.float32)
    device = q.device

    atoms = torch.stack(
        [codebook[k].view(-1).to(device=device, dtype=torch.float32) for k in keys],
        dim=0,
    )

    qn = q.norm().clamp_min(1e-12)
    atoms_norm = atoms.norm(dim=1).clamp_min(1e-12)
    dots = torch.matmul(atoms, q)
    cos_sims = dots / (atoms_norm * qn)
    best_i = int(torch.argmax(cos_sims).item())
    
    return keys[best_i], float(cos_sims[best_i].item())


class VSACodebook:
    """Lazy persistent codebook of role / filler hypervectors.

    The substrate accumulates atoms over time as new entities, predicates, and
    intents appear; reusing the codebook means the same triple
    (``subject ⊗ ada``) always projects to the same hypervector across runs,
    which is what makes long-lived semantic algebra coherent.
    """

    def __init__(
        self,
        *,
        dim: int = DEFAULT_VSA_DIM,
        base_seed: int = 0,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.dim = int(dim)
        self.base_seed = int(base_seed)
        self.device = device
        self.dtype = dtype
        self._atoms: dict[str, torch.Tensor] = {}
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._atoms)

    def __contains__(self, name: object) -> bool:
        return name in self._atoms

    def atom(self, name: str) -> torch.Tensor:
        with self._lock:
            v = self._atoms.get(name)

            if v is None:
                v = hypervector(
                    name,
                    dim=self.dim,
                    base_seed=self.base_seed,
                    dtype=self.dtype,
                    device=self.device,
                )
            
                self._atoms[name] = v
            
                logger.debug(
                    "VSACodebook.atom: registered name=%r dim=%d total=%d",
                    name,
                    self.dim,
                    len(self._atoms),
                )
            
            return v

    def encode_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
        *,
        role_subject: str = "ROLE_SUBJECT",
        role_predicate: str = "ROLE_PREDICATE",
        role_object: str = "ROLE_OBJECT",
    ) -> torch.Tensor:
        """Encode (subject, predicate, object) as a single bundle of role/filler bindings.

        The returned hypervector contains all three bindings in superposition.
        Recovering the object given the predicate is one ``unbind`` followed by
        a ``cleanup`` — the system gets analogical query for free.
        """
        return bundle(
            [
                bind(self.atom(role_subject), self.atom(subject)),
                bind(self.atom(role_predicate), self.atom(predicate)),
                bind(self.atom(role_object), self.atom(obj)),
            ]
        )

    def decode_role(
        self,
        encoded: torch.Tensor,
        role: str,
        *,
        candidates: Sequence[str] | None = None,
    ) -> tuple[str, float]:
        """Unbind a role from an encoded triple and clean up against the codebook.

        ``candidates`` lets callers restrict the cleanup to a subset (e.g. only
        known objects) so unrelated atoms can't accidentally win the cosine
        race when the codebook is large.
        """
        unbound = unbind(encoded, self.atom(role))
        
        if candidates is None:
            with self._lock:
                books = dict(self._atoms)
        else:
            with self._lock:
                atom_copy = dict(self._atoms)
        
            books = {
                name: atom_copy[name] for name in candidates if name in atom_copy
            }
        
            unknown = sorted(set(candidates) - set(books))
        
            if unknown:
                logger.debug(
                    "VSACodebook.decode_role: ignoring unknown candidates=%s", unknown
                )
        
            if not books:
                raise ValueError(
                    "decode_role: none of the requested candidates exist in the codebook — refusing cleanup over an empty book",
                )
        
        name, cos = cleanup(unbound, books)
        
        logger.debug(
            "VSACodebook.decode_role: role=%s -> name=%r cos=%.4f candidate_count=%d",
            role,
            name,
            cos,
            len(books),
        )
        
        return name, cos
