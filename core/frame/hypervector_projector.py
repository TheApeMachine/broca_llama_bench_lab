"""Sparse random projection of VSA hypervectors into the Broca tail.

A 10,000-dim hypervector contains the bound triple
(subject ⊗ ada + predicate ⊗ lives_in + object ⊗ rome) in superposition.
The Broca feature vector that drives grafts can only spend ``VSA_INJECTION_DIM``
slots on it, so the hypervector is squeezed through a deterministic sparse
projection: ``k`` non-zero indices per output dimension, signs from a
hash-derived ±1 stream.
"""

from __future__ import annotations

import math

import torch

from .dimensions import FrameDimensions


class HypervectorProjector:
    """Deterministic sparse projection from ``[d_in]`` hypervector to ``[dim_out]``.

    The projection is parameter-free given ``seed``; the same seed yields the
    same projection across processes. ``n_nonzero`` is the number of input
    indices that participate in each output coordinate; it scales the
    output's variance to unit by dividing by ``√n_nonzero``.
    """

    def __init__(self, *, dim_out: int | None = None, n_nonzero: int = 128) -> None:
        self.dim_out = int(dim_out) if dim_out is not None else FrameDimensions.VSA_INJECTION_DIM
        if self.dim_out <= 0:
            raise ValueError(f"HypervectorProjector.dim_out must be positive, got {self.dim_out}")
        self.n_nonzero = int(n_nonzero)
        if self.n_nonzero <= 0:
            raise ValueError(
                f"HypervectorProjector.n_nonzero must be positive, got {self.n_nonzero}"
            )

    def project(self, hypervector: torch.Tensor, *, seed: int) -> torch.Tensor:
        d_in = int(hypervector.numel())
        k = max(1, min(self.n_nonzero, d_in))
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed) & 0x7FFFFFFFFFFFFFFF)
        out = torch.zeros(self.dim_out, dtype=torch.float32)
        v32 = hypervector.detach().float().cpu().view(-1)
        scale = 1.0 / math.sqrt(float(k))

        for j in range(self.dim_out):
            idx = torch.randint(0, d_in, (k,), generator=g)
            sgn = torch.randint(0, 2, (k,), generator=g, dtype=torch.float32) * 2.0 - 1.0
            out[j] = (v32[idx] * sgn).sum() * scale

        return out / out.norm().clamp_min(1e-12)
