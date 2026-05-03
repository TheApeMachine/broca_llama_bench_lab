"""LatentMAS within-organ alignment ``W_a = pinv(W_out) @ W_in``.

Given a frozen organ's input embedding matrix ``W_in : [V, d]`` and language
model head ``W_out : [V, d]``, the closed-form Wasserstein-optimal projection
that maps a hidden state back into the input embedding distribution is

    W_a = (W_outᵀ W_out + λI)⁻¹ W_outᵀ W_in.

For Llama-3.2 with tied embeddings (``W_out == W_in``), this reduces to the
projector onto the column space of ``W_inᵀ`` — effectively identity when
``V >> d`` and the embedding has full column rank, which is what makes the
LatentMAS m=40-step latent rollout work without drift.

The implementation uses ``torch.linalg.pinv`` so the regularisation behaviour
falls out of the SVD truncation tolerance built into PyTorch — no λ
hyperparameter is exposed because the rule of "no tunables" forbids one.
"""

from __future__ import annotations

import torch

from .base import BaseAlignment


class RidgeAlignment(BaseAlignment):
    """Closed-form ``W_a`` derived from one organ's input/output projections."""

    def __init__(
        self,
        *,
        name: str,
        w_in: torch.Tensor,
        w_out: torch.Tensor,
    ) -> None:
        if w_in.ndim != 2 or w_out.ndim != 2:
            raise ValueError(
                f"RidgeAlignment requires 2-D embedding matrices, got w_in.ndim={w_in.ndim}, w_out.ndim={w_out.ndim}"
            )

        if w_in.shape != w_out.shape:
            raise ValueError(
                f"RidgeAlignment requires w_in.shape == w_out.shape ([V, d]), got {tuple(w_in.shape)} vs {tuple(w_out.shape)}"
            )

        v, d = w_in.shape

        if v < d:
            raise ValueError(
                f"RidgeAlignment requires V >= d (vocab >= hidden), got V={v}, d={d}; W_out cannot be inverted on the d-dim subspace"
            )

        super().__init__(name=name, d_in=int(d), d_out=int(d))

        # CPU first, dtype after — MPS does not support float64.
        w_in_f = w_in.detach().to(device="cpu").to(dtype=torch.float64)
        w_out_f = w_out.detach().to(device="cpu").to(dtype=torch.float64)

        # pinv(W_out) @ W_in is the limit-of-λ→0 form of the LatentMAS Wₐ.
        # SVD truncation inside torch.linalg.pinv handles conditioning.
        self._matrix = (torch.linalg.pinv(w_out_f) @ w_in_f).to(dtype=torch.float32).contiguous()

    @property
    def matrix(self) -> torch.Tensor:
        return self._matrix
