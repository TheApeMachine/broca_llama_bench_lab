"""Predictive-coding style discrepancy between top-down lexical grafts and inputs.

Teacher-forced cross-entropy compares logits from ``lm_head(final_hidden.pre)``
vs ``lm_head(final_hidden.post)`` in a **single** transformer forward per step
(when the host returns ``return_cache`` with pre/post graft states). That avoids
an extra full forward with all grafts disabled.

Falls back to a two-pass graft-on/graft-off loop for hosts without ``lm_head`` or
cache support.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from ..host.tokenizer import speech_seed_ids

logger = logging.getLogger(__name__)


def _batch_from_ids(rows: Sequence[Sequence[int]], pad_id: int, *, device: torch.device | str):
    if not rows:
        z_ids = torch.full((0, 1), pad_id, dtype=torch.long, device=device)
        z_mask = torch.zeros((0, 1), dtype=torch.bool, device=device)
        return z_ids, z_mask
    max_len = max(1, max(len(r) for r in rows))
    ids = torch.full((len(rows), max_len), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((len(rows), max_len), dtype=torch.bool, device=device)
    for i, row in enumerate(rows):
        if not row:
            continue
        ids[i, : len(row)] = torch.tensor(row, dtype=torch.long, device=device)
        mask[i, : len(row)] = True
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
    broca_features: torch.Tensor | None = None,
) -> float:
    """Mean negative log-likelihood of ``target_ids`` under teacher-forced prefixes.

    Complexity: each target token runs a full forward over the growing prefix (length
    grows with step), so cost scales quadratically in utterance length unless the host
    supports KV-cache incremental forwards with graft state replay.
    """

    if not target_ids:
        return 0.0
    device = next(model.parameters()).device
    pad_id = int(tokenizer.pad_id)
    total_nll = 0.0
    row = list(prefix_ids)
    graft_cm = model.grafts_enabled(grafts_on) if hasattr(model, "grafts_enabled") else nullcontext()
    lm_head = getattr(model, "lm_head", None)
    plan_tensor = torch.tensor([list(plan_ids)], device=device)
    bf_device = broca_features.to(device) if broca_features is not None else None

    with graft_cm:
        for step, tgt in enumerate(target_ids):
            tid = int(tgt)
            batch_ids, mask = _batch_from_ids([row], pad_id, device=device)
            extra: dict = {}
            if grafts_on:
                extra["broca_plan_token_ids"] = plan_tensor
                extra["broca_step"] = torch.tensor([min(step, max(0, len(plan_ids) - 1))], device=device)
                extra["tokenizer"] = tokenizer
                if bf_device is not None:
                    extra["broca_features"] = bf_device

            last_pos = max(int(mask[0].long().sum().item()) - 1, 0)

            if grafts_on and lm_head is not None:
                out = model(batch_ids, mask, extra_state=extra, return_cache=True)
                if isinstance(out, tuple):
                    _, cache = out
                    h_post = cache.get("final_hidden.post")
                    if h_post is not None:
                        dtype = lm_head.weight.dtype
                        logits_row = lm_head(h_post.to(dtype))[0, last_pos]
                        total_nll -= float(F.log_softmax(logits_row, dim=-1)[tid])
                        row.append(tid)
                        continue

            logits = model(batch_ids, mask, extra_state=extra if extra else None)
            if isinstance(logits, tuple):
                logits = logits[0]
            total_nll -= float(F.log_softmax(logits[0, last_pos], dim=-1)[tid])
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
    broca_features: torch.Tensor | None = None,
) -> tuple[float, float, float]:
    """``(mean_nll_graft, mean_nll_plain, gap)`` with ``gap = graft - plain``.

    Like :func:`lexical_plan_cross_entropy_mean`, the dual CE path performs one forward
    per target token over an lengthening prefix (quadratic in utterance length for long
    sequences) unless KV-cache reuse is added at the host layer.
    """

    prefix_ids = speech_seed_ids(tokenizer, prefix)
    target_ids = tokenizer.encode(utterance)
    plan_ids = tokenizer.encode_plan_words(list(plan_words), lowercase=True)

    if not target_ids:
        return 0.0, 0.0, 0.0

    device = next(model.parameters()).device
    pad_id = int(tokenizer.pad_id)
    row = list(prefix_ids)
    sum_graft = 0.0
    sum_plain = 0.0
    lm_head = getattr(model, "lm_head", None)
    plan_tensor = torch.tensor([list(plan_ids)], device=device)
    prepared_broca = broca_features.to(device) if broca_features is not None else None

    graft_cm = model.grafts_enabled(True) if hasattr(model, "grafts_enabled") else nullcontext()
    use_dual = True
    with graft_cm:
        for step, tgt in enumerate(target_ids):
            tid = int(tgt)
            batch_ids, mask = _batch_from_ids([row], pad_id, device=device)
            # Mirror lexical_plan_cross_entropy_mean ``extra`` (incl. empty ``plan_ids``:
            # ``broca_step`` uses ``min(step, max(0, len(plan_ids)-1))``, same as graft-on CE).
            extra: dict = {}
            extra["broca_plan_token_ids"] = plan_tensor
            extra["broca_step"] = torch.tensor([min(step, max(0, len(plan_ids) - 1))], device=device)
            extra["tokenizer"] = tokenizer
            if prepared_broca is not None:
                extra["broca_features"] = prepared_broca
            last_pos = max(int(mask[0].long().sum().item()) - 1, 0)

            if lm_head is None:
                use_dual = False
                break

            out = model(batch_ids, mask, extra_state=extra, return_cache=True)
            if not isinstance(out, tuple):
                use_dual = False
                break
            _, cache = out
            h_pre = cache.get("final_hidden.pre")
            h_post = cache.get("final_hidden.post")
            if h_pre is None or h_post is None:
                use_dual = False
                break

            dtype = lm_head.weight.dtype
            logits_plain = lm_head(h_pre.to(dtype))[0, last_pos]
            logits_graft = lm_head(h_post.to(dtype))[0, last_pos]
            sum_plain -= float(F.log_softmax(logits_plain, dim=-1)[tid])
            sum_graft -= float(F.log_softmax(logits_graft, dim=-1)[tid])
            row.append(tid)

    if use_dual:
        n = float(len(target_ids))
        ce_p = sum_plain / n
        ce_g = sum_graft / n
        gap = float(ce_g - ce_p)
        logger.debug(
            "lexical_surprise_gap: path=dual_ce n_targets=%d ce_g=%.6f ce_p=%.6f gap=%.6f utterance_preview=%r",
            len(target_ids),
            ce_g,
            ce_p,
            gap,
            (utterance[:100] + "…") if len(utterance) > 100 else utterance,
        )
        return ce_g, ce_p, gap

    ce_g = lexical_plan_cross_entropy_mean(
        model,
        tokenizer,
        prefix_ids=prefix_ids,
        target_ids=target_ids,
        plan_ids=plan_ids,
        grafts_on=True,
        broca_features=broca_features,
    )
    ce_p = lexical_plan_cross_entropy_mean(
        model,
        tokenizer,
        prefix_ids=prefix_ids,
        target_ids=target_ids,
        plan_ids=plan_ids,
        grafts_on=False,
    )
    gap_fb = float(ce_g - ce_p)
    logger.debug(
        "lexical_surprise_gap: path=fallback_two_pass n_targets=%d ce_g=%.6f ce_p=%.6f gap=%.6f utterance_preview=%r",
        len(target_ids),
        ce_g,
        ce_p,
        gap_fb,
        (utterance[:100] + "…") if len(utterance) > 100 else utterance,
    )
    return ce_g, ce_p, gap_fb
