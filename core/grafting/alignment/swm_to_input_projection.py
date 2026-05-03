"""Closed-form projection from the Substrate Working Memory onto a target organ's input space.

The SWM lives in the substrate's canonical hyperdimensional space (the
existing 10 000-dim VSA basis from ``core.symbolic.vsa``). To inject SWM
state into a frozen organ's residual stream we need a fixed projection
``R^D_swm -> R^d_target`` that respects two constraints:

1. **Lossless on the algebraic structure that matters.** VSA atoms are
   quasi-orthogonal i.i.d. Gaussians. The Johnson-Lindenstrauss lemma
   guarantees that any random orthogonal projection from D_swm down to
   d_target preserves pairwise distances and inner products up to a
   distortion that vanishes as D_swm grows. At D_swm=10 000 and target
   d=2048, JL distortion is well below the substrate's own SNR floor.

2. **Lands in the column space of the target's W_in.** A pure JL
   projection lands somewhere in ``R^d_target`` but not necessarily inside
   the manifold the target organ knows how to read. Composing the JL with
   the target's input embedding column-space projector
   ``P = W_in (W_inᵀ W_in)⁻¹ W_inᵀ`` solves this; it's the orthogonal
   projector onto the row space of W_in (the space of valid input
   embeddings, in row-vec convention).

The matrix is computed once at load time from the target's W_in plus a
deterministic JL seed. No training, no learned parameters.
"""

from __future__ import annotations

import math

import torch

from .base import BaseAlignment


class SWMToInputProjection(BaseAlignment):
    """Substrate-to-organ projection: D_swm -> d_target via JL ∘ column-space projector."""

    def __init__(
        self,
        *,
        name: str,
        d_swm: int,
        w_in_target: torch.Tensor,
        seed: int = 0,
    ) -> None:
        if w_in_target.ndim != 2:
            raise ValueError(
                f"SWMToInputProjection requires 2-D w_in_target, got ndim={w_in_target.ndim}"
            )

        v, d = w_in_target.shape

        if v < d:
            raise ValueError(
                f"SWMToInputProjection requires V >= d for column-space projector, got V={v}, d={d}"
            )

        super().__init__(name=name, d_in=int(d_swm), d_out=int(d))

        # Step 1: deterministic Johnson-Lindenstrauss matrix R: [D_swm, d_target].
        # Entries N(0, 1/d_target) so columns of R have unit-variance dot products.
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed) & 0x7FFFFFFFFFFFFFFF)
        r = torch.empty(int(d_swm), int(d), dtype=torch.float64)
        r.normal_(mean=0.0, std=1.0 / math.sqrt(float(d)), generator=g)

        # Step 2: column-space projector P = W_in @ pinv(W_in) — orthogonal
        # projector onto the row space of W_in in row-vec convention. For full
        # column rank W_in this equals (W_inᵀ W_in)⁻¹ W_inᵀ pre-multiplied by
        # W_in, which is the d×d projector. We compute it via pinv to inherit
        # SVD truncation.
        # CPU first, dtype after — MPS does not support float64.
        w_in_f = w_in_target.detach().to(device="cpu").to(dtype=torch.float64)
        # P : [d, d] orthogonal projector onto col-space of W_inᵀ.
        p = pinv_left_projector(w_in_f)

        # Compose: any SWM vector s of shape [..., D_swm] becomes
        # e = (s @ R) @ P, i.e. the JL output snapped onto valid input space.
        self._matrix = (r @ p).to(dtype=torch.float32).contiguous()

    @property
    def matrix(self) -> torch.Tensor:
        return self._matrix


def pinv_left_projector(w: torch.Tensor) -> torch.Tensor:
    """Orthogonal projector onto the row space of ``w`` in row-vec convention.

    For ``w`` of shape ``[V, d]`` with full column rank (V >= d), the rows of
    ``w`` span a d-dim subspace of ``R^d`` — typically all of ``R^d``, in
    which case the projector is identity. When the rows underfill ``R^d``
    (rare for foundation-model embeddings), the projector restricts to the
    span without distortion.

    Implementation: ``P = pinv(w) @ w`` returns a ``[d, d]`` orthogonal
    projector onto the row space, derived from the same SVD that
    ``torch.linalg.pinv`` uses internally.
    """

    return torch.linalg.pinv(w) @ w
