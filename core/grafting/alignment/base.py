"""Base interface for closed-form latent-space alignments.

An alignment is a fixed linear operator ``W : R^{d_in} -> R^{d_out}`` derived
once from a pair of pretrained matrices and then frozen for the lifetime of
the substrate. Subclasses differ only in *which* matrices feed the closed-form
derivation, never in the contract: ``apply(x)`` projects, ``matrix`` exposes
the underlying tensor for inspection.

The base class caches per-(device, dtype) copies of the alignment matrix so
that hot paths (e.g. :class:`SWMResidualGraft` firing every chat-decode
token) do not transfer ~80MB of matrix data to the host's accelerator on
every call. The cache is keyed on ``(device, dtype)`` and never purged —
alignments are immutable for the substrate's lifetime and all known target
devices fit in host memory simultaneously.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class BaseAlignment(ABC):
    """Abstract closed-form alignment operator with device-resident caching."""

    def __init__(self, *, name: str, d_in: int, d_out: int) -> None:
        self.name = str(name)
        self.d_in = int(d_in)
        self.d_out = int(d_out)
        self._device_cache: dict[tuple[str, torch.dtype], torch.Tensor] = {}

        if self.d_in <= 0 or self.d_out <= 0:
            raise ValueError(
                f"{type(self).__name__} requires positive dims, got d_in={self.d_in}, d_out={self.d_out}"
            )

    @property
    @abstractmethod
    def matrix(self) -> torch.Tensor:
        """The closed-form alignment matrix of shape ``[d_in, d_out]``."""

    def matrix_on(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return the alignment matrix resident on ``device`` with ``dtype``.

        First call per ``(device, dtype)`` pays the transfer; later calls are
        cache hits. Detaches the cached tensor so autograd cannot accidentally
        propagate through the alignment.
        """

        key = (str(device), dtype)
        cached = self._device_cache.get(key)

        if cached is None:
            cached = self.matrix.detach().to(device=device, dtype=dtype).contiguous()
            self._device_cache[key] = cached

        return cached

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Project ``x`` whose last dim is ``d_in`` to a tensor with last dim ``d_out``."""

        if x.shape[-1] != self.d_in:
            raise ValueError(
                f"{type(self).__name__}.apply: expected last dim {self.d_in}, got {x.shape[-1]}"
            )

        m = self.matrix_on(device=x.device, dtype=x.dtype)

        return x @ m

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, d_in={self.d_in}, d_out={self.d_out})"
