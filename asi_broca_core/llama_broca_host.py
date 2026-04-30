
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
import os
import types
from typing import Any, Dict, Iterator, Optional

import torch
import torch.nn as nn

from .device_utils import inference_dtype, pick_torch_device
from .hf_tokenizer_compat import HuggingFaceBrocaTokenizer
from .host import freeze_module


def quiet_transformers_benchmark_log_warnings() -> None:
    """Lower log level for benign transformers deprecation noise during scripted loads.

    Older checkpoints still deserialize ``torch_dtype``; accessing the compatibility
    property logs a one-shot warning via ``logging`` that dominates benchmark output.
    This does not change model behavior."""
    try:
        import logging

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
                stacked = torch.stack(deltas, dim=0)
                tau = float(state.get("graft_mixer_tau", getattr(self, "graft_mixer_tau", 1.0)))
                tau = max(tau, 1e-6)
                scores = -stacked.detach().pow(2).mean(dim=(1, 2, 3))
                w = torch.softmax(scores / tau, dim=0).view(-1, 1, 1, 1)
                x = x0 + (w * stacked).sum(dim=0)
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
        last_indices = mask.long().sum(dim=1).clamp_min(1) - 1
        attn = mask.long()

        tokenizer = None
        if extra_state:
            tokenizer = extra_state.get("tokenizer")
        state: dict[str, Any] = {
            "model": self,
            "tokenizer": tokenizer,
            "attention_mask": mask,
            "token_ids": idx,
            "last_indices": last_indices,
        }
        if extra_state:
            state.update(extra_state)

        cache: Optional[Dict[str, torch.Tensor]] = {} if return_cache else None

        lm = self.llm
        self._active_state = state
        self._active_cache = cache
        try:
            out = lm.model(input_ids=idx, attention_mask=attn, return_dict=True)
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
        raise ImportError("Llama backend requires transformers; pip install -r requirements-benchmark.txt") from e

    dev = device if isinstance(device, torch.device) else pick_torch_device(device)
    tk = resolve_hf_hub_token(token)
    dtype = torch_dtype or inference_dtype(dev)
    attn_impl = "eager" if dev.type == "mps" else "sdpa"

    model_kwargs = dict(
        dtype=dtype,
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
