"""Base interface for closed-form latent-space alignments.

An alignment is a fixed linear operator ``W : R^{d_in} -> R^{d_out}`` derived
once from a pair of pretrained matrices and then frozen for the lifetime of
the substrate. Subclasses differ only in *which* matrices feed the closed-form
derivation, never in the contract: ``apply(x)`` projects, ``matrix`` exposes
the underlying tensor for inspection.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class BaseAlignment(ABC):
    """Abstract closed-form alignment operator.

    Concrete subclasses compute the alignment matrix in ``__init__`` and never
    update it afterwards. Calling ``apply`` projects a tensor whose final axis
    matches ``d_in``; the returned tensor's final axis is ``d_out``.
    """

    def __init__(self, *, name: str, d_in: int, d_out: int) -> None:
        self.name = str(name)
        self.d_in = int(d_in)
        self.d_out = int(d_out)

        if self.d_in <= 0 or self.d_out <= 0:
            raise ValueError(
                f"{type(self).__name__} requires positive dims, got d_in={self.d_in}, d_out={self.d_out}"
            )

    @property
    @abstractmethod
    def matrix(self) -> torch.Tensor:
        """The closed-form alignment matrix of shape ``[d_in, d_out]``."""

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Project ``x`` whose last dim is ``d_in`` to a tensor with last dim ``d_out``."""

        if x.shape[-1] != self.d_in:
            raise ValueError(
                f"{type(self).__name__}.apply: expected last dim {self.d_in}, got {x.shape[-1]}"
            )

        m = self.matrix.to(device=x.device, dtype=x.dtype)

        return x @ m

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, d_in={self.d_in}, d_out={self.d_out})"
