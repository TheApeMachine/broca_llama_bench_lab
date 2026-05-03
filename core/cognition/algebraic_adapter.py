"""AlgebraicMemoryAdapter — VSA / Hopfield / ontology helpers on top of the substrate.

The substrate controller used to inline four small wrappers around the
algebraic-memory primitives. They cluster cleanly under one concern:
representing concepts as continuous vectors and storing role-filler bound
triples in the Hopfield store.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from ..frame import SubwordProjector


if TYPE_CHECKING:
    from .substrate import SubstrateController


_SUBWORD = SubwordProjector()


class AlgebraicMemoryAdapter:
    """Thin façade over ``mind.vsa``, ``mind.hopfield_memory``, ``mind.ontology``."""

    def __init__(self, mind: "SubstrateController") -> None:
        self._mind = mind

    def encode_triple(self, subject: str, predicate: str, obj: str) -> torch.Tensor:
        return self._mind.vsa.encode_triple(subject, predicate, obj)

    def padded_hopfield_sketch(self, sketch: torch.Tensor) -> torch.Tensor:
        d = self._mind.hopfield_memory.d_model
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
        self._mind.hopfield_memory.remember(
            self.padded_hopfield_sketch(a_sketch),
            self.padded_hopfield_sketch(b_sketch),
            metadata=dict(metadata or {}),
        )

    def vector_for_concept(
        self, name: str, *, base_sketch: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return the substrate's preferred vector for a concept name.

        Routes through the ontology registry so frequent concepts use their
        promoted orthogonal axis; less-frequent ones still use the hashed
        sketch. Always observes the access (so the next call can flip
        promotion).
        """

        mind = self._mind
        mind.ontology.observe(name)
        sketch = base_sketch if base_sketch is not None else _SUBWORD.encode(name)
        promoted = mind.ontology.maybe_promote(name, sketch)
        if promoted is not None:
            return promoted.axis
        return F.normalize(sketch.detach().to(torch.float32).flatten(), dim=0)
