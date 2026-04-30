from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Dict, Iterator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TinyConfig:
    vocab_size: int
    d_model: int = 96
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 192
    max_seq_len: int = 64
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TinyConfig):
        super().__init__()
        if cfg.d_model % cfg.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seq_len, d_model = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~causal.view(1, 1, seq_len, seq_len), torch.finfo(scores.dtype).min)
        if attention_mask is not None:
            scores = scores.masked_fill(~attention_mask.view(bsz, 1, 1, seq_len), torch.finfo(scores.dtype).min)
        probs = F.softmax(scores, dim=-1)
        out = probs @ v
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, d_model)
        return self.proj(self.dropout(out))


class TinyBlock(nn.Module):
    def __init__(self, cfg: TinyConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )


class TinyCausalTransformer(nn.Module):
    """A small transformer with explicit graft slots.

    This host exists so the experiment can be run on a laptop while still doing
    real layer surgery. Grafts receive and return tensors, and can therefore be
    inserted into the residual stream just like adapters or external-memory
    modules in larger models.

    Slots:
      block.{i}.post_attn
      block.{i}.post_mlp
      final_hidden
      logits
    """

    def __init__(self, cfg: TinyConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([TinyBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self.grafts = nn.ModuleDict()
        self._grafts_enabled = True

    @staticmethod
    def _slot_key(slot: str) -> str:
        return slot.replace(".", "__")

    def add_graft(self, slot: str, graft: nn.Module) -> None:
        key = self._slot_key(slot)
        if key not in self.grafts:
            self.grafts[key] = nn.ModuleList()
        self.grafts[key].append(graft)
        setattr(graft, "slot", slot)

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
        if self._grafts_enabled and key in self.grafts:
            for graft in self.grafts[key]:
                x = graft(x, state)
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
        bsz, seq_len = idx.shape
        if seq_len > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {seq_len} exceeds max_seq_len {self.cfg.max_seq_len}")
        if attention_mask is None:
            attention_mask = torch.ones_like(idx, dtype=torch.bool)
        positions = torch.arange(seq_len, device=idx.device).view(1, seq_len)
        x = self.tok_emb(idx) + self.pos_emb(positions)
        x = F.dropout(x, p=self.cfg.dropout, training=self.training)
        cache: Optional[Dict[str, torch.Tensor]] = {} if return_cache else None
        state = {
            "model": self,
            "token_ids": idx,
            "attention_mask": attention_mask,
            "last_indices": attention_mask.long().sum(dim=1).clamp_min(1) - 1,
        }
        if extra_state:
            state.update(extra_state)
        for i, block in enumerate(self.blocks):
            x = x + block.attn(block.ln1(x), attention_mask)
            x = self._apply_grafts(f"block.{i}.post_attn", x, state, cache)
            x = x + block.mlp(block.ln2(x))
            x = self._apply_grafts(f"block.{i}.post_mlp", x, state, cache)
        x = self.ln_f(x)
        x = self._apply_grafts("final_hidden", x, state, cache)
        logits = self.lm_head(x)
        logits = self._apply_grafts("logits", logits, state, cache)
        if return_cache:
            assert cache is not None
            return logits, cache
        return logits


def freeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def count_parameters(module: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


