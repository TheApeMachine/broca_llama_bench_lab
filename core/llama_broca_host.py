"""Frozen Llama-family hosts with Broca graft slots.

Loads ``meta-llama/Llama-3.2-1B-Instruct`` through Hugging Face transformers.
The host intentionally treats the model as the language organ: weights are
frozen, and external faculties can write into named residual-stream slots.

Supported graft slots:
  final_hidden          normalized hidden states immediately before ``lm_head``
  logits                vocabulary logits after ``lm_head``
  layer.{i}.post        output hidden states of HF Llama decoder layer i

The layer slots are installed as HF forward hooks, so real Llama layers can be
intervened on without rewriting the transformers model internals. Hooks are
removed when the last graft on that layer slot is detached (see ``remove_graft`` /
``clear_slot_grafts``).
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
import types
from typing import Any, Dict, Iterator, Optional

import torch
import torch.nn as nn

from .device_utils import inference_dtype, pick_torch_device
from .hf_tokenizer_compat import HuggingFaceBrocaTokenizer
from .host import freeze_module

logger = logging.getLogger(__name__)


def quiet_transformers_benchmark_log_warnings() -> None:
    """Lower log level for benign transformers deprecation noise during scripted loads.

    Older checkpoints still deserialize ``torch_dtype``; accessing the compatibility
    property logs a one-shot warning via ``logging`` that dominates benchmark output.
    This does not change model behavior."""
    try:
        for name in ("transformers.configuration_utils", "transformers.modeling_utils"):
            logging.getLogger(name).setLevel(logging.ERROR)
    except Exception:  # pragma: no cover
        pass


class LlamaBrocaHost(nn.Module):
    """Broca-compatible host wrapping ``transformers.LlamaForCausalLM``.

    Broca graft API: ``add_graft``, ``remove_graft``, ``clear_slot_grafts``,
    ``grafts_enabled``, ``forward``.  The wrapped HF model remains frozen.
    """

    def __init__(self, causal_lm: Any) -> None:
        super().__init__()
        self.llm = causal_lm
        freeze_module(self.llm)
        cfg = causal_lm.config
        self.cfg = types.SimpleNamespace(
            d_model=int(cfg.hidden_size),
            max_seq_len=int(getattr(cfg, "max_position_embeddings", 8192)),
            n_layers=int(getattr(cfg, "num_hidden_layers", 0)),
        )
        self.grafts = nn.ModuleDict()
        self._grafts_enabled = True
        self._active_state: dict[str, Any] | None = None
        self._active_cache: Optional[Dict[str, torch.Tensor]] = None
        self._hook_handles: dict[int, Any] = {}
        self.graft_mixer_tau: float = 1.0
        self.graft_mixer_max_delta_rms_ratio: float = 8.0
        self.graft_mixer_max_delta_rms_floor: float = 10.0
        self.last_graft_mix: dict[str, Any] = {}
        self.enable_last_graft_mix_diagnostics: bool = False
        self._last_graft_mix_lock = threading.Lock()

    @staticmethod
    def layer_post_slot(layer_idx: int) -> str:
        """Canonical slot name for interventions after decoder layer ``layer_idx``."""

        return f"layer.{int(layer_idx)}.post"

    @staticmethod
    def _slot_key(slot: str) -> str:
        return slot.replace(".", "__")

    @staticmethod
    def _parse_layer_slot(slot: str) -> int | None:
        parts = slot.split(".")
        if len(parts) == 3 and parts[0] == "layer" and parts[2] == "post":
            try:
                return int(parts[1])
            except ValueError:
                return None
        return None

    @property
    def lm_head(self) -> nn.Linear:
        return self.llm.lm_head

    @property
    def config(self) -> Any:
        return self.llm.config

    @property
    def name_or_path(self) -> str:
        return getattr(self.llm, "name_or_path", "")

    @property
    def layers(self) -> Any:
        return self.llm.model.layers

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate to HF generation.

        The explicit Broca generation helpers in ``broca.py`` call ``forward``
        directly so grafts are active there.  Plain HF ``generate`` is delegated
        for leaderboard compatibility.
        """

        return self.llm.generate(*args, **kwargs)

    def add_graft(self, slot: str, graft: nn.Module) -> None:
        key = self._slot_key(slot)
        if key not in self.grafts:
            self.grafts[key] = nn.ModuleList()
        self.grafts[key].append(graft)
        setattr(graft, "slot", slot)
        layer_idx = self._parse_layer_slot(slot)
        if layer_idx is not None:
            self._ensure_layer_hook(layer_idx)

    def _ensure_layer_hook(self, layer_idx: int) -> None:
        if layer_idx in self._hook_handles:
            return
        layers = self.layers
        if layer_idx < 0 or layer_idx >= len(layers):
            raise IndexError(f"Llama layer index {layer_idx} out of range 0..{len(layers)-1}")

        def _hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
            state = self._active_state
            if state is None:
                return output
            slot = f"layer.{layer_idx}.post"
            if isinstance(output, tuple):
                hidden = output[0]
                new_hidden = self._apply_grafts(slot, hidden, state, self._active_cache)
                return (new_hidden, *output[1:])
            hidden = output
            return self._apply_grafts(slot, hidden, state, self._active_cache)

        self._hook_handles[layer_idx] = layers[layer_idx].register_forward_hook(_hook)

    def _teardown_layer_hook_if_unused(self, layer_idx: int) -> None:
        slot = self.layer_post_slot(layer_idx)
        key = self._slot_key(slot)
        if key in self.grafts and len(self.grafts[key]) > 0:
            return
        handle = self._hook_handles.pop(layer_idx, None)
        if handle is not None:
            handle.remove()

    def remove_graft(self, slot: str, index: int = -1) -> nn.Module | None:
        """Remove one graft from a slot (default: last appended). Returns the module or ``None``."""

        key = self._slot_key(slot)
        if key not in self.grafts:
            return None
        lst = self.grafts[key]
        if len(lst) == 0:
            del self.grafts[key]
            layer_idx = self._parse_layer_slot(slot)
            if layer_idx is not None:
                self._teardown_layer_hook_if_unused(layer_idx)
            return None
        removed = lst.pop(index)
        if len(lst) == 0:
            del self.grafts[key]
        layer_idx = self._parse_layer_slot(slot)
        if layer_idx is not None:
            self._teardown_layer_hook_if_unused(layer_idx)
        return removed

    def clear_slot_grafts(self, slot: str) -> list[nn.Module]:
        """Detach all grafts for ``slot`` and remove layer hooks when applicable."""

        key = self._slot_key(slot)
        if key not in self.grafts:
            return []
        lst = self.grafts[key]
        out: list[nn.Module] = []
        while len(lst) > 0:
            out.append(lst.pop(-1))
        out.reverse()
        del self.grafts[key]
        layer_idx = self._parse_layer_slot(slot)
        if layer_idx is not None:
            self._teardown_layer_hook_if_unused(layer_idx)
        return out

    def clear_all_grafts(self) -> list[tuple[str, nn.Module]]:
        """Remove every graft; returns ``(canonical_slot, module)`` pairs in arbitrary order."""

        pairs: list[tuple[str, nn.Module]] = []
        for key, lst in list(self.grafts.items()):
            slot = key.replace("__", ".")
            while len(lst) > 0:
                m = lst.pop(-1)
                pairs.append((slot, m))
            del self.grafts[key]
        for layer_idx in list(self._hook_handles.keys()):
            self._teardown_layer_hook_if_unused(layer_idx)
        return pairs

    @contextlib.contextmanager
    def grafts_enabled(self, enabled: bool) -> Iterator[None]:
        old = self._grafts_enabled
        self._grafts_enabled = bool(enabled)
        try:
            yield
        finally:
            self._grafts_enabled = old

    def graft_report(self) -> str:
        lines: list[str] = []
        for key, modules in self.grafts.items():
            slot = key.replace("__", ".")
            for i, module in enumerate(modules):
                total = sum(p.numel() for p in module.parameters())
                trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
                lines.append(f"{slot}[{i}]: {module.__class__.__name__} params={total} trainable={trainable}")
        return "\n".join(lines) if lines else "<no grafts>"

    def _graft_priority(self, graft: nn.Module, state: dict) -> float:
        priority = getattr(graft, "mixer_priority", 1.0)
        if callable(priority):
            priority = priority(state)
        try:
            return max(float(priority), 1e-6)
        except Exception:
            return 1.0

    def _mix_graft_deltas(self, slot: str, x0: torch.Tensor, deltas: list[torch.Tensor], mods: list[nn.Module], state: dict) -> torch.Tensor:
        stacked = torch.stack(deltas, dim=0)
        reduce_dims = tuple(range(1, stacked.ndim))
        delta_rms = stacked.detach().pow(2).mean(dim=reduce_dims).sqrt()
        priorities = torch.tensor([self._graft_priority(g, state) for g in mods], device=stacked.device, dtype=stacked.dtype)
        tau = float(state.get("graft_mixer_tau", getattr(self, "graft_mixer_tau", 1.0)))
        tau = max(tau, 1e-6)
        scores = torch.log1p(delta_rms.to(stacked.dtype)) + torch.log(priorities)
        weights = torch.softmax(scores / tau, dim=0)
        view_shape = (weights.shape[0],) + (1,) * (stacked.ndim - 1)
        mixed = (weights.view(view_shape) * stacked).sum(dim=0)

        mixed_rms = mixed.detach().pow(2).mean().sqrt()
        base_rms = x0.detach().pow(2).mean().sqrt()
        ratio = float(state.get("graft_mixer_max_delta_rms_ratio", self.graft_mixer_max_delta_rms_ratio))
        floor = float(state.get("graft_mixer_max_delta_rms_floor", self.graft_mixer_max_delta_rms_floor))
        cap = (base_rms * max(ratio, 0.0)).clamp_min(max(floor, 1e-6))
        scale = torch.clamp(cap / mixed_rms.clamp_min(1e-12), max=1.0).to(dtype=mixed.dtype)
        mixed = mixed * scale
        if self.enable_last_graft_mix_diagnostics:
            snapshot = {
                "weights": weights.detach().cpu(),
                "scores": scores.detach().cpu(),
                "delta_rms": delta_rms.detach().cpu(),
                "scale": float(scale.detach().cpu()),
            }
            with self._last_graft_mix_lock:
                self.last_graft_mix[slot] = snapshot
        return x0 + mixed

    def _apply_grafts(self, slot: str, x: torch.Tensor, state: dict, cache: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        if cache is not None:
            cache[f"{slot}.pre"] = x.detach().clone()
        key = self._slot_key(slot)
        x0 = x
        if self._grafts_enabled and key in self.grafts:
            mods = list(self.grafts[key])
            if len(mods) == 1:
                x = mods[0](x0, state)
            elif len(mods) > 1:
                deltas: list[torch.Tensor] = []
                for graft in mods:
                    deltas.append(graft(x0, state) - x0)
                x = self._mix_graft_deltas(slot, x0, deltas, mods, state)
        if cache is not None:
            cache[f"{slot}.post"] = x.detach().clone()
        return x

    def forward(
        self,
        idx: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        extra_state: Optional[dict] = None,
        return_cache: bool = False,
    ):
        if idx.ndim != 2:
            raise ValueError("idx must have shape [batch, seq]")
        _, seq_len = idx.shape
        cfg_max = int(getattr(self.cfg, "max_seq_len", 8192))
        if seq_len > cfg_max:
            raise ValueError(f"sequence length {seq_len} exceeds max_seq_len {cfg_max}")
        if attention_mask is None:
            mask = torch.ones_like(idx, dtype=torch.bool)
        else:
            mask = attention_mask.bool()
        attn = mask.long()

        tokenizer = None
        if extra_state:
            tokenizer = extra_state.get("tokenizer")
        past_key_values = None
        return_past_key_values = False
        if extra_state:
            past_key_values = extra_state.get("past_key_values")
            return_past_key_values = bool(extra_state.get("return_past_key_values", False))

        if past_key_values is None:
            last_indices = mask.long().sum(dim=1).clamp_min(1) - 1
        else:
            # With KV cache, hidden/logits cover only the new tokens; grafts index that slice.
            seq_new = idx.shape[1]
            last_indices = torch.full((idx.shape[0],), max(0, seq_new - 1), device=idx.device, dtype=torch.long)

        state: dict[str, Any] = {
            "model": self,
            "tokenizer": tokenizer,
            "attention_mask": mask,
            "token_ids": idx,
            "last_indices": last_indices,
        }
        if extra_state:
            for _k, _v in extra_state.items():
                if _k in ("past_key_values", "return_past_key_values"):
                    continue
                state[_k] = _v

        if logger.isEnabledFor(logging.DEBUG) and extra_state and self._grafts_enabled:
            broca_hints = tuple(
                sorted(k for k in extra_state.keys() if k.startswith("broca_") or k in ("tokenizer",))
            )
            if broca_hints:
                n_graft_slots = sum(len(v) for v in self.grafts.values())
                logger.debug(
                    "LlamaBrocaHost.forward: bsz=%d seq=%d graft_slots_populated=%d extra_keys=%s return_cache=%s",
                    idx.shape[0],
                    seq_len,
                    n_graft_slots,
                    broca_hints,
                    bool(return_cache),
                )

        cache: Optional[Dict[str, torch.Tensor]] = {} if return_cache else None

        lm = self.llm
        self._active_state = state
        self._active_cache = cache
        try:
            model_kwargs: dict[str, Any] = {
                "input_ids": idx,
                "attention_mask": attn,
                "return_dict": True,
                "use_cache": True,
            }
            if past_key_values is not None:
                model_kwargs["past_key_values"] = past_key_values
            out = lm.model(**model_kwargs)
        finally:
            self._active_state = None
            self._active_cache = None

        hidden = out.last_hidden_state
        dtype = lm.lm_head.weight.dtype
        hidden = hidden.to(dtype)
        hidden = self._apply_grafts("final_hidden", hidden, state, cache)
        logits = lm.lm_head(hidden)
        logits = self._apply_grafts("logits", logits, state, cache)

        if return_cache:
            assert cache is not None
            return logits, cache
        if return_past_key_values:
            return logits, out.past_key_values
        return logits


def resolve_hf_hub_token(token: str | bool | None) -> str | bool:
    """Non-empty strings are tokens; empty values use ``HF_TOKEN``; ``False`` disables auth."""

    if token is False:
        return False
    if isinstance(token, str):
        ts = token.strip()
        if ts:
            return ts
    env = os.environ.get("HF_TOKEN", "").strip()
    return env if env else True


def load_llama_broca_host(
    model_id: str = "meta-llama/Llama-3.2-1B-Instruct",
    *,
    device: torch.device | str | None = None,
    torch_dtype: torch.dtype | None = None,
    token: str | bool | None = None,
    trust_remote_code: bool = False,
) -> tuple[LlamaBrocaHost, HuggingFaceBrocaTokenizer]:
    quiet_transformers_benchmark_log_warnings()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            'Llama backend requires transformers; run `uv sync --extra benchmark` or `pip install -e ".[benchmark]"`.'
        ) from e

    dev = device if isinstance(device, torch.device) else pick_torch_device(device)
    tk = resolve_hf_hub_token(token)
    dtype = torch_dtype or inference_dtype(dev)
    attn_impl = "eager" if dev.type == "mps" else "sdpa"

    model_kwargs = dict(
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        token=tk,
        attn_implementation=attn_impl,
        trust_remote_code=trust_remote_code,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    except TypeError:
        model_kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    tok_inner = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
        token=tk,
        trust_remote_code=trust_remote_code,
        clean_up_tokenization_spaces=False,
    )
    if getattr(tok_inner, "pad_token_id", None) is None and getattr(tok_inner, "eos_token_id", None) is not None:
        tok_inner.pad_token = tok_inner.eos_token
        tok_inner.pad_token_id = int(tok_inner.eos_token_id)

    wrapper = LlamaBrocaHost(model).to(dev)
    tokenizer = HuggingFaceBrocaTokenizer(tok_inner)
    wrapper.eval()
    return wrapper, tokenizer
