"""MacroAdapter — substrate-side façade over the macro chunking registry.

Three small wrappers the controller used to inline:

* :meth:`recent_intents` — the last N intents from the workspace journal,
  used as the prefix the chunking compiler matches against.
* :meth:`find_matching_macro` — registry lookup by intent prefix or by
  feature similarity (Hopfield-style cosine).
* :meth:`speech_features` — pull the FrameDimensions.broca_feature_dim()-shaped
  feature vector for one compiled macro.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

from ..idletime.chunking import CompiledMacro, macro_frame_features


if TYPE_CHECKING:
    from .substrate import SubstrateController


class MacroAdapter:
    """Stateless façade over ``mind.macro_registry`` + chunking compiler."""

    def __init__(self, mind: "SubstrateController") -> None:
        self._mind = mind

    def recent_intents(self, *, limit: int = 8) -> list[str]:
        try:
            rows = self._mind.journal.recent(limit=int(limit))
        except Exception:
            return []
        return [str(r.get("intent", "") or "unknown") for r in rows]

    def find_matching(
        self,
        *,
        recent_intents: Sequence[str] | None = None,
        features: torch.Tensor | None = None,
    ) -> CompiledMacro | None:
        mind = self._mind
        if features is not None:
            return mind.macro_registry.find_macro_by_features(
                features,
                min_cosine=mind.chunking_compiler.config.hopfield_weight_min_for_oneshot,
            )
        recent = list(recent_intents) if recent_intents is not None else self.recent_intents()
        return mind.macro_registry.find_macro_matching_prefix(recent)

    @staticmethod
    def speech_features(macro: CompiledMacro) -> torch.Tensor:
        return macro_frame_features(macro)
