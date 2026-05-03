"""GraftFeatureAdapter — frame → graft input vectors.

Two thin wrappers that compose VSA + frame-packer + chat orchestrator's
content-bias logic. Lifted out of the substrate controller.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from ..frame import CognitiveFrame


if TYPE_CHECKING:
    from .substrate import SubstrateController


logger = logging.getLogger(__name__)


class GraftFeatureAdapter:
    """Stateless façade over ``mind.frame_packer`` + content-bias derivation."""

    def __init__(self, mind: "SubstrateController") -> None:
        self._mind = mind

    def broca_features(self, frame: CognitiveFrame) -> torch.Tensor:
        """Sketch frame + numeric tail + sparse VSA injection for :class:`TrainableFeatureGraft`."""

        mind = self._mind
        vsa_vec: torch.Tensor | None = None
        if frame.subject and frame.answer and str(frame.answer).lower() not in {"", "unknown"}:
            pr = str((frame.evidence or {}).get("predicate", frame.intent))
            try:
                vsa_vec = mind.encode_triple_vsa(
                    str(frame.subject), pr, str(frame.answer)
                )
            except (RuntimeError, ValueError, TypeError):
                logger.debug(
                    "GraftFeatureAdapter.broca_features: VSA encode skipped",
                    exc_info=True,
                )
        return mind.frame_packer.broca(
            frame.intent,
            frame.subject,
            frame.answer,
            float(frame.confidence),
            frame.evidence,
            vsa_bundle=vsa_vec,
            vsa_projection_seed=int(mind.seed),
        )

    def content_logit_bias(self, frame: CognitiveFrame) -> dict[int, float]:
        """Token-ID bonuses derived from frame content for scripted host scoring."""

        from .chat_orchestrator import ChatOrchestrator

        return ChatOrchestrator(self._mind)._content_logit_bias(frame)
