"""A single Substrate Working Memory slot."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .source import SWMSource


@dataclass(frozen=True)
class SWMSlot:
    """One named, typed, write-once vector in the substrate's working memory.

    ``vector`` lives at the canonical SWM dimensionality
    (``core.symbolic.DEFAULT_VSA_DIM``). ``frozen=True`` enforces the rule
    that mutation goes through the working-memory API, not through ad-hoc
    tensor edits.
    """

    name: str
    vector: torch.Tensor
    source: SWMSource
    written_at_tick: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(f"SWMSlot.name must be a non-empty string, got {self.name!r}")

        if not isinstance(self.vector, torch.Tensor):
            raise TypeError(f"SWMSlot.vector must be torch.Tensor, got {type(self.vector).__name__}")

        if self.vector.ndim != 1:
            raise ValueError(f"SWMSlot.vector must be 1-D, got shape {tuple(self.vector.shape)}")

        if not isinstance(self.source, SWMSource):
            raise TypeError(f"SWMSlot.source must be SWMSource, got {type(self.source).__name__}")

        if int(self.written_at_tick) < 0:
            raise ValueError(f"SWMSlot.written_at_tick must be >= 0, got {self.written_at_tick}")
