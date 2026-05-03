"""MacroAdapter — journal-derived intents + macro registry lookup."""

from __future__ import annotations

from typing import Any, Sequence

import torch

from ..idletime.chunking import CompiledMacro, macro_frame_features


class MacroAdapter:
    """Macro chunk registry façade."""

    def __init__(
        self,
        *,
        journal: Any,
        macro_registry: Any,
        chunking_compiler: Any,
    ) -> None:
        self._journal = journal
        self._macros = macro_registry
        self._compiler = chunking_compiler

    def recent_intents(self, *, limit: int = 8) -> list[str]:
        rows = self._journal.recent(limit=int(limit))
        return [str(r.get("intent", "") or "unknown") for r in rows]

    def find_matching(
        self,
        *,
        recent_intents: Sequence[str] | None = None,
        features: torch.Tensor | None = None,
    ) -> CompiledMacro | None:
        if features is not None:
            return self._macros.find_macro_by_features(
                features,
                min_cosine=self._compiler.config.hopfield_weight_min_for_oneshot,
            )
        recent = list(recent_intents) if recent_intents is not None else self.recent_intents()
        return self._macros.find_macro_matching_prefix(recent)

    @staticmethod
    def speech_features(macro: CompiledMacro) -> torch.Tensor:
        return macro_frame_features(macro)
