"""BareLanguageHostGenerator — the baseline arm of the architecture benchmark.

Runs the frozen LLM with grafts disabled (or never attached). Used by
:mod:`research_lab.benchmarks.architecture_eval` to score a baseline that
has no substrate guidance, so the eval can quantify how much the grafted
substrate adds.

This is **not** a system feature. The substrate's chat path always uses
:class:`~core.generation.PlanForcedGenerator`; the bare-LM path is a
benchmark concern only.
"""

from __future__ import annotations

from typing import Any

import torch

from core.generation import TokenBatch, TokenDecoder
from core.host.tokenizer import speech_seed_ids


class BareLanguageHostGenerator:
    """Stateless wrapper that runs the host without grafts for benchmarking."""

    @classmethod
    def generate(
        cls,
        model: torch.nn.Module,
        tokenizer: Any,
        *,
        prefix: str | None = None,
        max_new_tokens: int = 5,
    ) -> str:
        ids = speech_seed_ids(tokenizer, prefix)
        generated: list[int] = []
        device = next(model.parameters()).device
        for _ in range(max_new_tokens):
            row = ids + generated
            batch_ids, mask, _ = TokenBatch.from_id_rows(
                [row], tokenizer.pad_id, device=device
            )
            logits = model(batch_ids, mask)
            pred = int(logits[0, mask.long().sum().item() - 1].argmax().item())
            generated.append(pred)
        return TokenDecoder.decode(tokenizer, generated)
