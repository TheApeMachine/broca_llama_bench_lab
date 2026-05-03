"""Cross-model LatentMAS alignment for organs with shared pretrained ancestry.

When two frozen organs derive from the same pretrained backbone (e.g. GLiNER2
and GLiClass both fine-tuned from DeBERTa-v3), their input embedding matrices
span related subspaces of a shared semantic space. The same Wasserstein-optimal
ridge projection that LatentMAS uses within an organ generalises directly:

    W_AB = pinv(W_out_A) @ W_in_B          shape [d_A, d_B]

Applied as ``e_B = h_A @ W_AB`` it projects organ A's hidden state into the
column space of organ B's input embedding — the same linearised "what input
would have produced this hidden" derivation as the within-organ Wₐ, but with
the target embedding swapped out for B's.

For organs with *different* hidden dims (e.g. GLiNER2 base d=768 and GLiClass
small d=384) the projection changes dimension cleanly. The vocabularies need
not match in size, but they must align row-by-row over their shared prefix —
a non-shared codebook breaks the "same conceptual space" premise and the
projection silently degrades. Disagreement past the shared prefix is truncated
explicitly (no silent fallback over imagined rows).
"""

from __future__ import annotations

import torch

from .base import BaseAlignment


class CrossModelAlignment(BaseAlignment):
    """Closed-form projection from organ A's hidden space to organ B's input space."""

    def __init__(
        self,
        *,
        name: str,
        w_out_source: torch.Tensor,
        w_in_target: torch.Tensor,
    ) -> None:
        if w_out_source.ndim != 2 or w_in_target.ndim != 2:
            raise ValueError(
                f"CrossModelAlignment requires 2-D matrices, got w_out_source.ndim={w_out_source.ndim}, w_in_target.ndim={w_in_target.ndim}"
            )

        v_a, d_a = w_out_source.shape
        v_b, d_b = w_in_target.shape

        if v_a < d_a:
            raise ValueError(
                f"CrossModelAlignment requires V_A >= d_A for stable pseudoinverse, got V_A={v_a}, d_A={d_a}"
            )

        super().__init__(name=name, d_in=int(d_a), d_out=int(d_b))

        # CPU first, dtype after — MPS does not support float64.
        w_out_a = w_out_source.detach().to(device="cpu").to(dtype=torch.float64)
        w_in_b = w_in_target.detach().to(device="cpu").to(dtype=torch.float64)

        if v_a != v_b:
            # Different vocabularies: align the row spaces by truncating to the
            # shorter vocab. The shared prefix is the only place the row
            # correspondence is honest; pretending past it is the kind of
            # silent fallback CLAUDE.md forbids.
            v_shared = min(v_a, v_b)
            w_out_a = w_out_a[:v_shared]
            w_in_b = w_in_b[:v_shared]

        # W_AB = pinv(W_out_A) @ W_in_B
        # shapes: [d_A, V_shared] @ [V_shared, d_B] -> [d_A, d_B]
        self._matrix = (
            (torch.linalg.pinv(w_out_a) @ w_in_b).to(dtype=torch.float32).contiguous()
        )

    @property
    def matrix(self) -> torch.Tensor:
        return self._matrix
