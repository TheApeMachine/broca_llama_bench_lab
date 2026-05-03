"""Pack a cognitive frame's fields into the continuous feature vectors that
grafts inject into the host residual stream.

The packer is constructed once with the text encoder it should use (lexical
:class:`SubwordProjector` or host-driven :class:`EmbeddingProjector`) and
reused across every frame. There is no per-call encoder argument; the
substrate has exactly one packer per session.

Two outputs:

* :meth:`cognitive` — three sketches (intent / subject / answer) concatenated
  with the numeric tail. Width: :meth:`FrameDimensions.cognitive_frame_dim`.
* :meth:`broca` — cognitive frame plus the sparse-projected VSA injection
  tail. Width: :meth:`FrameDimensions.broca_feature_dim`.

The packer is the only place that orders these subvectors. Every consumer
that decodes Broca features must trust this order.
"""

from __future__ import annotations

from typing import Any

import torch

from .dimensions import FrameDimensions
from .hypervector_projector import HypervectorProjector
from .numeric_tail import NumericTail
from .subword_projector import SubwordProjector
from .text_encoder import TextEncoder


class FramePacker:
    """Concatenates sketches and the numeric/VSA tails into graft-ready vectors."""

    def __init__(
        self,
        text_encoder: TextEncoder | None = None,
        *,
        hypervector_projector: HypervectorProjector | None = None,
    ) -> None:
        self._text_encoder: TextEncoder = text_encoder if text_encoder is not None else SubwordProjector()
        self._hypervector = hypervector_projector or HypervectorProjector()

    @property
    def text_encoder(self) -> TextEncoder:
        return self._text_encoder

    def cognitive(
        self,
        intent: str,
        subject: str,
        answer: str,
        confidence: float,
        evidence: dict[str, Any] | None,
    ) -> torch.Tensor:
        return torch.cat(
            [
                self._sketch(intent),
                self._sketch(subject),
                self._sketch(answer),
                NumericTail.encode(confidence, evidence),
            ],
            dim=0,
        )

    def broca(
        self,
        intent: str,
        subject: str,
        answer: str,
        confidence: float,
        evidence: dict[str, Any] | None,
        *,
        vsa_bundle: torch.Tensor | None = None,
        vsa_projection_seed: int = 0,
    ) -> torch.Tensor:
        base = self.cognitive(intent, subject, answer, confidence, evidence)
        if vsa_bundle is None:
            tail = torch.zeros(FrameDimensions.VSA_INJECTION_DIM, dtype=torch.float32)
        else:
            tail = self._hypervector.project(vsa_bundle, seed=int(vsa_projection_seed))
        return torch.cat([base, tail], dim=0)

    def _sketch(self, text: str) -> torch.Tensor:
        v = self._text_encoder(str(text))
        if not isinstance(v, torch.Tensor):
            v = torch.tensor(v, dtype=torch.float32)
        v = v.detach().float().cpu().view(-1)
        if v.numel() != FrameDimensions.SKETCH_DIM:
            raise ValueError(
                f"text encoder returned dim {v.numel()}, expected {FrameDimensions.SKETCH_DIM}"
            )
        return v
