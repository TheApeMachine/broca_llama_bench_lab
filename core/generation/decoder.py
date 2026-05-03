"""TokenBatch, TokenDecoder, PlanForcedGenerator.

Three small classes that previously lived as free functions in the
substrate monolith (``_batch_from_ids``, ``decode_generation``,
``generate_from_plan``). Each is stateless; methods are classmethods so
callers don't have to instantiate.

``generate_without_substrate`` (the bare-LM benchmark arm) does not live
here — it is a benchmark concern and lives in
:mod:`research_lab.benchmarks.bare_language_host`.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch

from ..host.tokenizer import speech_seed_ids


class TokenBatch:
    """Stateless pad-and-mask helper for batched forward passes."""

    @classmethod
    def from_id_rows(
        cls,
        rows: Sequence[Sequence[int]],
        pad_id: int,
        *,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_len = max(1, max(len(r) for r in rows))
        ids = torch.full((len(rows), max_len), pad_id, dtype=torch.long)
        mask = torch.zeros((len(rows), max_len), dtype=torch.bool)
        lengths = torch.tensor([len(r) for r in rows], dtype=torch.long)
        for i, row in enumerate(rows):
            if not row:
                continue
            ids[i, : len(row)] = torch.tensor(row, dtype=torch.long)
            mask[i, : len(row)] = True
        if device is not None:
            ids = ids.to(device)
            mask = mask.to(device)
            lengths = lengths.to(device)
        return ids, mask, lengths


class TokenDecoder:
    """Stateless decoder; prefers :meth:`decode_tokens`, falls back to per-id decode."""

    @classmethod
    def decode(cls, tokenizer: Any, generated: Sequence[int]) -> str:
        dec = getattr(tokenizer, "decode_tokens", None)
        if callable(dec):
            return str(dec(list(generated))).strip()
        return " ".join(tokenizer.decode_id(int(i)) for i in generated)


class PlanForcedGenerator:
    """Run the host step-by-step under a fixed lexical plan.

    Each call performs ``min(max_new_tokens, len(plan_ids))`` forward passes,
    populating ``broca_plan_token_ids`` / ``broca_step`` / ``broca_features``
    in ``extra_state`` so the lexical and feature grafts can bias the host
    toward the plan. Returns ``(text_out, generated_ids, inertia_tail)``
    where ``inertia_tail`` is ``log1p(prefix_len + generated_len)``.
    """

    @classmethod
    def generate(
        cls,
        model: torch.nn.Module,
        tokenizer: Any,
        plan_tokens: Sequence[str],
        *,
        prefix: str | None = None,
        max_new_tokens: int | None = None,
        broca_features: torch.Tensor | None = None,
    ) -> tuple[str, list[int], float]:
        plan_ids = list(tokenizer.encode_plan_words(plan_tokens, lowercase=True))
        max_new_tokens = max_new_tokens or len(plan_ids)
        ids = speech_seed_ids(tokenizer, prefix)
        generated: list[int] = []
        params_fn = getattr(model, "parameters", None)
        if not callable(params_fn):
            raise RuntimeError(
                "PlanForcedGenerator.generate requires model.parameters() for device placement"
            )
        device = next(params_fn()).device
        steps = range(min(max_new_tokens, len(plan_ids)))
        for step in steps:
            row = ids + generated
            batch_ids, mask, _ = TokenBatch.from_id_rows(
                [row], tokenizer.pad_id, device=device
            )
            extra: dict[str, Any] = {
                "broca_plan_token_ids": torch.tensor([plan_ids], device=device),
                "broca_step": torch.tensor([step], device=device),
                "tokenizer": tokenizer,
            }
            if broca_features is not None:
                extra["broca_features"] = broca_features.to(device)
            logits = model(batch_ids, mask, extra_state=extra)
            pred = int(logits[0, mask.long().sum().item() - 1].argmax().item())
            generated.append(pred)
        text_out = TokenDecoder.decode(tokenizer, generated)
        inertia_tail = math.log1p(float(max(1, len(ids) + len(generated))))
        return text_out, generated, inertia_tail
