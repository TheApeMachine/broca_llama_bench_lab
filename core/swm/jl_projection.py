"""Closed-form Johnson-Lindenstrauss projection between arbitrary dims.

Used to lift an organ's encoder hidden state (typical d_organ = 384..2048) up
into the substrate working memory's canonical hyperdim
(``DEFAULT_VSA_DIM = 10 000``) without training. JL guarantees that pairwise
inner products and cosine similarities are preserved up to an O(1/√d_target)
distortion that vanishes well before D_swm.

The projection is deterministic given a seed and entirely parameter-free
beyond that — no learned weights, no fitting.
"""

from __future__ import annotations

import math

import torch


class JLProjection:
    """Deterministic Johnson-Lindenstrauss projection ``R^{d_in} -> R^{d_out}``."""

    def __init__(self, *, name: str, d_in: int, d_out: int, seed: int = 0) -> None:
        if int(d_in) <= 0 or int(d_out) <= 0:
            raise ValueError(
                f"JLProjection requires positive dims, got d_in={d_in}, d_out={d_out}"
            )

        self.name = str(name)
        self.d_in = int(d_in)
        self.d_out = int(d_out)
        self.seed = int(seed)

        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed & 0x7FFFFFFFFFFFFFFF)

        m = torch.empty(self.d_in, self.d_out, dtype=torch.float32)
        # JL with N(0, 1/d_out): projected vectors have unit-variance inner
        # products in expectation when the source has unit-variance entries.
        m.normal_(mean=0.0, std=1.0 / math.sqrt(float(self.d_out)), generator=g)

        self._matrix = m.contiguous()

    @property
    def matrix(self) -> torch.Tensor:
        return self._matrix

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.d_in:
            raise ValueError(
                f"JLProjection.apply: expected last dim {self.d_in}, got {x.shape[-1]}"
            )

        m = self._matrix.to(device=x.device, dtype=x.dtype)
        return x @ m

    def __repr__(self) -> str:
        return f"JLProjection(name={self.name!r}, d_in={self.d_in}, d_out={self.d_out}, seed={self.seed})"
