"""AlgebraicMemoryAdapter — VSA / Hopfield / ontology helpers."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ..frame import SubwordProjector


_SUBWORD = SubwordProjector()


class AlgebraicMemoryAdapter:
    """Vectors for triples, Hopfield writes, ontology-backed concept axes."""

    def __init__(
        self,
        *,
        vsa: Any,
        hopfield_memory: Any,
        ontology: Any,
    ) -> None:
        self._vsa = vsa
        self._hopfield = hopfield_memory
        self._ontology = ontology

    def encode_triple(self, subject: str, predicate: str, obj: str) -> torch.Tensor:
        return self._vsa.encode_triple(subject, predicate, obj)

    def padded_hopfield_sketch(self, sketch: torch.Tensor) -> torch.Tensor:
        d = self._hopfield.d_model
        out = torch.zeros(d, dtype=torch.float32)
        s = sketch.detach().float().view(-1)
        n = min(int(s.numel()), d)
        if n > 0:
            out[:n] = s[:n]
        return out

    def remember(
        self,
        a_sketch: torch.Tensor,
        b_sketch: torch.Tensor,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._hopfield.remember(
            self.padded_hopfield_sketch(a_sketch),
            self.padded_hopfield_sketch(b_sketch),
            metadata=dict(metadata or {}),
        )

    def vector_for_concept(
        self, name: str, *, base_sketch: torch.Tensor | None = None
    ) -> torch.Tensor:
        self._ontology.observe(name)
        sketch = base_sketch if base_sketch is not None else _SUBWORD.encode(name)
        promoted = self._ontology.maybe_promote(name, sketch)
        if promoted is not None:
            return promoted.axis
        return F.normalize(sketch.detach().to(torch.float32).flatten(), dim=0)
