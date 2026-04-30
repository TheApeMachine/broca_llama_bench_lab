"""Paired lm-eval host parity: vanilla HFLM vs empty ``LlamaBrocaHost`` shell."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from types import MethodType
from typing import Any

import torch
from asi_broca_core.llama_broca_host import LlamaBrocaHost


def is_llama_hf_model_id(model_id: str) -> bool:
    s = model_id.lower()
    return "llama" in s or "meta-llama" in s


def build_hflm(
    model_id: str,
    *,
    device_s: str,
    coarse: str,
    attn_eager: bool = False,
    batch_size: int | str = "auto",
    max_batch_size: int = 64,
) -> Any:
    from lm_eval.models.huggingface import HFLM

    dtype = "float16" if coarse in ("cuda", "mps") else "float32"
    extra: dict[str, Any] = {
        "pretrained": model_id,
        "device": device_s,
        "dtype": dtype,
        "trust_remote_code": True,
        "batch_size": batch_size,
        "max_batch_size": max_batch_size,
    }
    if attn_eager:
        extra["attn_implementation"] = "eager"
    return HFLM(**extra)


def _broca_augmented_model_call(self: Any, inps: torch.Tensor, attn_mask=None, labels=None) -> torch.Tensor:
    """Causal logits through ``LlamaBrocaHost``; seq2seq path delegates to HF implementation."""

    with (
        torch.no_grad(),
        torch.autocast(
            device_type=self.device.type,
            dtype=self.mixed_precision_dtype,
            enabled=self.mixed_precision_dtype is not None,
        ),
    ):
        if attn_mask is not None or labels is not None:
            from lm_eval.models.huggingface import HFLM

            return HFLM._model_call(self, inps, attn_mask, labels)
        return self.model(inps)


def wrap_hflm_with_broca_host(lm: Any) -> None:
    """Replace ``lm._model`` with ``LlamaBrocaHost(inner)`` and patch ``_model_call``."""

    inner = lm._model
    if getattr(inner.config, "model_type", None) != "llama":
        raise TypeError(
            "Broca-host lm-eval pairing needs a Llama ``model_type`` checkpoint "
            f"(got {type(inner).__name__} / model_type={getattr(inner.config, 'model_type', None)!r})."
        )
    host = LlamaBrocaHost(inner)
    lm._model = host
    lm._model_call = MethodType(_broca_augmented_model_call, lm)


def _metrics_subset(results_block: dict[str, Any]) -> dict[str, float | str]:
    out: dict[str, float | str] = {}
    for k, v in results_block.items():
        if not isinstance(k, str):
            continue
        if k in {"alias"}:
            continue
        if "stderr" in k or k.endswith("_stderr"):
            out[k] = str(v)
            continue
        if "," not in k:
            continue
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def run_paired_lm_eval(
    *,
    model_id: str,
    preset_tasks: str,
    device_s: str,
    coarse: str,
    limit: str | None,
    out_dir: Path,
    num_fewshot: int | None = 0,
    bootstrap_iters: int = 0,
) -> Path:
    """Run Eleuther twice when checkpoint is Llama; otherwise baseline only.

    This is a host-parity check. It should stay near zero delta because no
    Broca grafts or faculty frames are attached in Eleuther's plain MC harness.
    """

    from lm_eval import simple_evaluate

    tasks = [t.strip() for t in preset_tasks.split(",") if t.strip()]
    limit_arg: int | float | None
    if limit is None or limit == "":
        limit_arg = None
    else:
        limit_arg = int(limit)

    attn_eager = coarse == "mps"

    lm1 = build_hflm(model_id, device_s=device_s, coarse=coarse, attn_eager=attn_eager)
    r_base = simple_evaluate(
        model=lm1,
        tasks=tasks,
        batch_size=getattr(lm1, "batch_size", "auto"),
        num_fewshot=num_fewshot,
        limit=limit_arg,
        log_samples=False,
        bootstrap_iters=bootstrap_iters,
    )
    del lm1
    gc.collect()

    pair: dict[str, Any] = {
        "kind": "lm_eval_host_parity",
        "model_id": model_id,
        "tasks": tasks,
        "limit_per_task": limit,
        "baseline_lm_eval_pure_hf": {
            "note": "vanilla causal LM logits (Eleuther Harness); comparable to public leaderboards.",
            "per_task_metrics": {t: _metrics_subset(r_base["results"].get(t, {})) for t in tasks},
        },
    }

    if not is_llama_hf_model_id(model_id):
        pair["broca_host_parity_lm_eval"] = {
            "skipped": True,
            "reason": "non-Llama checkpoint — Broca host only applies to LlamaFamily models",
        }
        pair_path = out_dir / "lm_eval_pair.json"
        pair_path.write_text(json.dumps(pair, indent=2, default=str), encoding="utf-8")
        return pair_path

    lm2 = build_hflm(model_id, device_s=device_s, coarse=coarse, attn_eager=attn_eager)
    try:
        wrap_hflm_with_broca_host(lm2)
    except Exception as exc:
        pair["broca_host_parity_lm_eval"] = {"skipped": True, "reason": str(exc)}
        pair_path = out_dir / "lm_eval_pair.json"
        pair_path.write_text(json.dumps(pair, indent=2, default=str), encoding="utf-8")
        del lm2
        gc.collect()
        return pair_path

    r_br = simple_evaluate(
        model=lm2,
        tasks=tasks,
        batch_size=getattr(lm2, "batch_size", "auto"),
        num_fewshot=num_fewshot,
        limit=limit_arg,
        log_samples=False,
        bootstrap_iters=bootstrap_iters,
    )
    del lm2
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    deltas: dict[str, dict[str, float | None]] = {}
    for task in tasks:
        b = r_base["results"].get(task, {})
        e = r_br["results"].get(task, {})
        row: dict[str, float | None] = {}
        for key in set(b) & set(e):
            if ",none" not in key:
                continue
            if isinstance(b.get(key), str) or isinstance(e.get(key), str):
                continue
            try:
                row[str(key)] = float(e[key]) - float(b[key])
            except (TypeError, ValueError):
                row[str(key)] = None
        deltas[task] = row

    pair["broca_host_parity_lm_eval"] = {
        "note": (
            "Same weights; logits flow through an empty LlamaBrocaHost shell. This is a parity check, "
            "not the active Broca architecture benchmark."
        ),
        "per_task_metrics": {t: _metrics_subset(r_br["results"].get(t, {})) for t in tasks},
    }
    pair["delta_host_parity_minus_baseline"] = deltas

    pair_path = out_dir / "lm_eval_pair.json"
    pair_path.write_text(json.dumps(pair, indent=2, default=str), encoding="utf-8")
    return pair_path


