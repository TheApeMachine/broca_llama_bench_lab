"""Predictive-coding style discrepancy between top-down lexical grafts and inputs.

Computes teacher-forcing cross-entropy with Broca lexical priming enabled versus a
baseline forward pass with grafts disabled. A strictly positive gap means the host
assigns lower likelihood to the observed tokens when forced toward the substrate
plan — the scalar surprise signal used by ``BrocaMind``.
"""

from __future__ import annotations

from typing import Any, Sequence

from contextlib import nullcontext

import torch
import torch.nn.functional as F

from .tokenizer import SPEECH_BRIDGE_PREFIX


def _batch_from_ids(rows: Sequence[Sequence[int]], pad_id: int, *, device: torch.device | str):
    max_len = max(1, max(len(r) for r in rows))
    ids = torch.full((len(rows), max_len), pad_id, dtype=torch.long)
    mask = torch.zeros((len(rows), max_len), dtype=torch.bool)
    for i, row in enumerate(rows):
        if not row:
            continue
        ids[i, : len(row)] = torch.tensor(row, dtype=torch.long)
        mask[i, : len(row)] = True
    ids = ids.to(device)
    mask = mask.to(device)
    return ids, mask


@torch.no_grad()
def lexical_plan_cross_entropy_mean(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    prefix_ids: Sequence[int],
    target_ids: Sequence[int],
    plan_ids: Sequence[int],
    grafts_on: bool,
) -> float:
    """Mean negative log-likelihood of ``target_ids`` under teacher-forced prefixes."""

    if not target_ids:
        return 0.0
    device = next(model.parameters()).device
    pad_id = int(tokenizer.pad_id)
    total_nll = 0.0
    row = list(prefix_ids)
    graft_cm = model.grafts_enabled(grafts_on) if hasattr(model, "grafts_enabled") else nullcontext()
    with graft_cm:
        for step, tgt in enumerate(target_ids):
            tid = int(tgt)
            batch_ids, mask = _batch_from_ids([row], pad_id, device=device)
            extra: dict = {}
            if grafts_on:
                extra["broca_plan_token_ids"] = torch.tensor([list(plan_ids)], device=device)
                extra["broca_step"] = torch.tensor([min(step, max(0, len(plan_ids) - 1))], device=device)
                extra["tokenizer"] = tokenizer
            logits = model(batch_ids, mask, extra_state=extra if extra else None)
            last_pos = int(mask.long().sum().item()) - 1
            logp = F.log_softmax(logits[0, last_pos], dim=-1)[tid]
            total_nll -= float(logp.item())
            row.append(tid)
    return total_nll / float(len(target_ids))


@torch.no_grad()
def lexical_surprise_gap(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    utterance: str,
    plan_words: Sequence[str],
    prefix: str | None = None,
) -> tuple[float, float, float]:
    """Returns ``(mean_nll_graft, mean_nll_plain, gap)`` where ``gap = graft - plain``."""

    prefix_ids = tokenizer.encode(prefix if prefix is not None else SPEECH_BRIDGE_PREFIX)
    target_ids = tokenizer.encode(utterance)
    plan_ids = tokenizer.encode_plan_words(list(plan_words))
    ce_g = lexical_plan_cross_entropy_mean(
        model,
        tokenizer,
        prefix_ids=prefix_ids,
        target_ids=target_ids,
        plan_ids=plan_ids,
        grafts_on=True,
    )
    ce_p = lexical_plan_cross_entropy_mean(
        model,
        tokenizer,
        prefix_ids=prefix_ids,
        target_ids=target_ids,
        plan_ids=plan_ids,
        grafts_on=False,
    )
    return ce_g, ce_p, float(ce_g - ce_p)
