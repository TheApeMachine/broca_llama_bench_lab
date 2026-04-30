from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseGraft(nn.Module):
    """Base class for modules inserted into a host model's internal slots."""

    def __init__(self):
        super().__init__()
        self.enabled = True

    def forward(self, x: torch.Tensor, state: dict) -> torch.Tensor:  # pragma: no cover
        return x


def _last_indices(state: dict, x: torch.Tensor) -> torch.Tensor:
    if "last_indices" in state:
        return state["last_indices"].to(x.device)
    mask = state.get("attention_mask")
    if mask is None:
        return torch.full((x.shape[0],), x.shape[1] - 1, device=x.device, dtype=torch.long)
    return mask.long().sum(dim=1).clamp_min(1).to(x.device) - 1


def _trigger_mask(token_ids: torch.Tensor, trigger_ids: Sequence[int] | None) -> torch.Tensor:
    if not trigger_ids:
        return torch.ones(token_ids.shape[0], device=token_ids.device, dtype=torch.bool)
    mask = torch.zeros(token_ids.shape[0], device=token_ids.device, dtype=torch.bool)
    for tid in trigger_ids:
        mask |= (token_ids == int(tid)).any(dim=1)
    return mask


class KVMemoryGraft(BaseGraft):
    """Content-addressable memory grafted into a residual stream.

    Keys and values are activation vectors. A query hidden state retrieves
    nearest keys, then adds the retrieved value direction to the final token.
    This changes logits through the host's internal representation.
    """

    def __init__(
        self,
        d_model: int,
        *,
        strength: float = 16.0,
        temperature: float = 0.03,
        threshold: float | None = 0.85,
        gate_sharpness: float = 40.0,
        top_k: int | None = None,
        max_items: int = 8192,
        query_mode: str = "sequence_mean",
    ):
        super().__init__()
        if query_mode not in {"token", "sequence_mean"}:
            raise ValueError("query_mode must be 'token' or 'sequence_mean'")
        self.d_model = int(d_model)
        self.strength = float(strength)
        self.temperature = float(temperature)
        self.threshold = threshold
        self.gate_sharpness = float(gate_sharpness)
        self.top_k = top_k
        self.max_items = int(max_items)
        self.query_mode = query_mode
        self.register_buffer("keys", torch.empty(0, d_model))
        self.register_buffer("values", torch.empty(0, d_model))
        self.metadata: list[dict] = []
        self.last_debug: dict = {}

    def clear(self) -> None:
        self.keys = self.keys.new_empty(0, self.d_model)
        self.values = self.values.new_empty(0, self.d_model)
        self.metadata.clear()
        self.last_debug = {}

    @torch.no_grad()
    def remember(self, key: torch.Tensor, value: torch.Tensor, metadata: dict | None = None) -> None:
        key = key.detach().reshape(-1, self.d_model).to(self.keys.device)
        value = value.detach().reshape(-1, self.d_model).to(self.values.device)
        if key.shape[0] != value.shape[0]:
            raise ValueError("key and value must contain the same number of rows")
        self.keys = torch.cat([self.keys, key], dim=0)[-self.max_items:]
        self.values = torch.cat([self.values, value], dim=0)[-self.max_items:]
        for _ in range(key.shape[0]):
            self.metadata.append(dict(metadata or {}))
        self.metadata = self.metadata[-self.max_items:]

    def _retrieve(self, queries: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = F.normalize(queries, dim=-1)
        k = F.normalize(self.keys.to(x.device), dim=-1)
        sims = q @ k.T
        if self.top_k is not None and self.top_k < sims.shape[-1]:
            vals, idx = sims.topk(self.top_k, dim=-1)
            masked = torch.full_like(sims, torch.finfo(sims.dtype).min)
            sims = masked.scatter(-1, idx, vals)
        weights = F.softmax(sims / max(self.temperature, 1e-6), dim=-1)
        retrieved = weights @ self.values.to(x.device)
        max_sim = sims.max(dim=-1).values
        if self.threshold is None:
            gate = torch.ones_like(max_sim).unsqueeze(-1)
        else:
            gate = torch.sigmoid((max_sim - self.threshold) * self.gate_sharpness).unsqueeze(-1)
        return self.strength * gate * retrieved, weights, gate.squeeze(-1)

    def forward(self, x: torch.Tensor, state: dict) -> torch.Tensor:
        if not self.enabled or self.keys.numel() == 0:
            return x
        bsz, seq_len, d_model = x.shape
        mask = state.get("attention_mask")
        if mask is None:
            mask = torch.ones(bsz, seq_len, device=x.device, dtype=torch.bool)
        if self.query_mode == "token":
            delta, weights, gate = self._retrieve(x.reshape(-1, d_model), x)
            self.last_debug = {"weights": weights.detach().cpu(), "gate": gate.detach().cpu()}
            return x + delta.reshape(bsz, seq_len, d_model)

        weights_for_mean = mask.to(x.dtype).unsqueeze(-1)
        lengths = weights_for_mean.sum(dim=1).clamp_min(1.0)
        queries = (x * weights_for_mean).sum(dim=1) / lengths
        delta, weights, gate = self._retrieve(queries, x)
        out = x.clone()
        last = _last_indices(state, x)
        out[torch.arange(bsz, device=x.device), last] += delta
        self.last_debug = {"weights": weights.detach().cpu(), "gate": gate.detach().cpu()}
        return out


class BottleneckAdapterGraft(BaseGraft):
    """Trainable spare neural capacity inserted into a frozen residual stream."""

    def __init__(self, d_model: int, bottleneck: int = 24, scale: float = 1.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.down = nn.Linear(d_model, bottleneck)
        self.up = nn.Linear(bottleneck, d_model)
        self.scale = float(scale)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.down.bias)
        nn.init.normal_(self.up.weight, std=0.02)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor, state: dict) -> torch.Tensor:
        if not self.enabled:
            return x
        return x + self.scale * self.up(F.gelu(self.down(self.norm(x))))


class LoRALinear(nn.Module):
    """Low-rank trainable delta around a frozen nn.Linear module."""

    def __init__(self, base: nn.Linear, r: int = 4, alpha: float = 8.0):
        super().__init__()
        if r <= 0:
            raise ValueError("rank r must be positive")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r
        self.A = nn.Parameter(torch.empty(self.r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, self.r))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + F.linear(F.linear(x, self.A), self.B) * self.scaling


def _get_parent(root: nn.Module, dotted_name: str) -> Tuple[nn.Module, str]:
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def inject_lora(
    model: nn.Module,
    *,
    target_substrings: Iterable[str] = ("qkv", "proj", "mlp.0", "mlp.2"),
    r: int = 4,
    alpha: float = 8.0,
) -> list[str]:
    replaced: list[str] = []
    for name, module in list(model.named_modules()):
        if name and isinstance(module, nn.Linear) and any(s in name for s in target_substrings):
            parent, child = _get_parent(model, name)
            setattr(parent, child, LoRALinear(module, r=r, alpha=alpha))
            replaced.append(name)
    return replaced




class FeatureVectorGraft(BaseGraft):
    """Trainable bridge from a numeric faculty-state vector to residual stream.

    The state dictionary must contain ``faculty_features`` with shape
    [batch, d_features]. Only this small bridge needs to train; the host model
    can remain frozen.
    """

    def __init__(self, d_features: int, d_model: int, *, trigger_ids: Sequence[int] | None = None, strength: float = 1.0):
        super().__init__()
        self.d_features = int(d_features)
        self.d_model = int(d_model)
        self.trigger_ids = tuple(int(x) for x in trigger_ids) if trigger_ids else tuple()
        self.strength = float(strength)
        self.norm = nn.LayerNorm(self.d_features)
        self.project = nn.Linear(self.d_features, self.d_model)
        nn.init.normal_(self.project.weight, std=0.02)
        nn.init.zeros_(self.project.bias)

    def forward(self, x: torch.Tensor, state: dict) -> torch.Tensor:
        if not self.enabled or "faculty_features" not in state:
            return x
        feats = state["faculty_features"]
        if not isinstance(feats, torch.Tensor):
            feats = torch.tensor(feats, device=x.device, dtype=x.dtype)
        feats = feats.to(x.device, x.dtype)
        if feats.ndim == 1:
            feats = feats.view(1, -1).expand(x.shape[0], -1)
        if feats.shape[-1] != self.d_features:
            raise ValueError(f"expected faculty_features dim {self.d_features}, got {feats.shape[-1]}")
        applies = _trigger_mask(state["token_ids"], self.trigger_ids)
        if not bool(applies.any()):
            return x
        delta = self.project(self.norm(feats)) * self.strength
        out = x.clone()
        last = _last_indices(state, x)
        rows = torch.arange(x.shape[0], device=x.device)[applies]
        out[rows, last[applies]] += delta[applies]
        return out

class TriggeredTokenDirectionGraft(BaseGraft):
    """Injects a selected token's embedding direction into the final hidden state.

    This is a compact bridge from a non-neural faculty into the neural host. The
    faculty computes a discrete answer/action, and the graft writes the
    corresponding token direction into the residual stream.
    """

    def __init__(self, token_by_name: Mapping[str, int], *, trigger_ids: Sequence[int] | None = None, strength: float = 18.0):
        super().__init__()
        self.token_by_name = {str(k): int(v) for k, v in token_by_name.items()}
        self.trigger_ids = tuple(int(x) for x in trigger_ids) if trigger_ids else tuple()
        self.strength = float(strength)
        self.last_name: str | None = None
        self.last_token_id: int | None = None

    def choose_name(self, state: dict) -> str | None:  # pragma: no cover
        return None

    def forward(self, x: torch.Tensor, state: dict) -> torch.Tensor:
        if not self.enabled:
            return x
        token_ids = state["token_ids"]
        applies = _trigger_mask(token_ids, self.trigger_ids)
        if not bool(applies.any()):
            return x
        name = self.choose_name(state)
        if name is None or name not in self.token_by_name:
            return x
        out = x.clone()
        model = state["model"]
        tok_id = self.token_by_name[name]
        direction = F.normalize(model.lm_head.weight[tok_id].detach().to(x.device, x.dtype), dim=0)
        last = _last_indices(state, x)
        rows = torch.arange(x.shape[0], device=x.device)[applies]
        out[rows, last[applies]] += self.strength * direction
        self.last_name = name
        self.last_token_id = tok_id
        return out


class ActiveInferenceTokenGraft(TriggeredTokenDirectionGraft):
    """Projects an active-inference policy decision into the residual stream."""

    def __init__(self, agent, token_by_action: Mapping[str, int], *, trigger_ids: Sequence[int] | None = None, strength: float = 18.0):
        super().__init__(token_by_action, trigger_ids=trigger_ids, strength=strength)
        self.agent = agent
        self.last_decision = None

    def choose_name(self, state: dict) -> str | None:
        decision = self.agent.decide()
        self.last_decision = decision
        return decision.action_name


class CausalEffectTokenGraft(TriggeredTokenDirectionGraft):
    """Projects a Pearl-style intervention result into the residual stream."""

    def __init__(
        self,
        scm,
        *,
        treatment: str,
        outcome: str,
        outcome_value,
        positive_token: int,
        negative_token: int,
        treatment_values: tuple = (1, 0),
        trigger_ids: Sequence[int] | None = None,
        strength: float = 18.0,
    ):
        super().__init__({"helps": int(positive_token), "hurts": int(negative_token)}, trigger_ids=trigger_ids, strength=strength)
        self.scm = scm
        self.treatment = treatment
        self.outcome = outcome
        self.outcome_value = outcome_value
        self.treatment_values = treatment_values
        self.last_effects: dict[str, float] = {}

    def choose_name(self, state: dict) -> str:
        t_pos, t_neg = self.treatment_values
        p_pos = self.scm.probability({self.outcome: self.outcome_value}, interventions={self.treatment: t_pos})
        p_neg = self.scm.probability({self.outcome: self.outcome_value}, interventions={self.treatment: t_neg})
        self.last_effects = {"p_do_positive": p_pos, "p_do_negative": p_neg, "ate": p_pos - p_neg}
        return "helps" if p_pos >= p_neg else "hurts"


@torch.no_grad()
def extract_next_token_memory(
    model,
    tokenizer,
    prompt: str,
    target_token: str,
    *,
    slot: str = "final_hidden",
    query_mode: str = "sequence_mean",
    value_scale: float = 1.0,
):
    """Create an activation key and output-token direction for one association."""

    if target_token not in tokenizer.token_to_id:
        raise KeyError(f"target token {target_token!r} not in tokenizer vocabulary")
    batch = tokenizer.batch_encode([prompt], device=next(model.parameters()).device)
    model.eval()
    with model.grafts_enabled(False):
        _, cache = model(batch.ids, batch.attention_mask, return_cache=True)
    hidden = cache[f"{slot}.pre"]
    if query_mode == "sequence_mean":
        mask = batch.attention_mask.to(hidden.dtype).unsqueeze(-1)
        key = (hidden * mask).sum(dim=1)[0] / mask.sum(dim=1).clamp_min(1.0)[0]
    elif query_mode == "token":
        key = hidden[0, batch.lengths.item() - 1]
    else:
        raise ValueError("query_mode must be 'token' or 'sequence_mean'")
    target_id = tokenizer.token_to_id[target_token]
    value = F.normalize(model.lm_head.weight[target_id].detach(), dim=0) * value_scale
    metadata = {"prompt": prompt, "target": target_token, "slot": slot, "query_mode": query_mode}
    return key.detach().cpu(), value.detach().cpu(), metadata


@torch.no_grad()
def memorize_next_token(model, tokenizer, graft: KVMemoryGraft, prompt: str, target_token: str, *, slot: str = "final_hidden", value_scale: float = 1.0) -> None:
    key, value, metadata = extract_next_token_memory(
        model,
        tokenizer,
        prompt,
        target_token,
        slot=slot,
        query_mode=getattr(graft, "query_mode", "sequence_mean"),
        value_scale=value_scale,
    )
    graft.remember(key.reshape(1, -1), value.reshape(1, -1), metadata=metadata)


@torch.no_grad()
def memorize_persistent_next_token(store, model, tokenizer, prompt: str, target_token: str, *, namespace: str | None = None, kind: str = "fact", slot: str = "final_hidden", query_mode: str = "sequence_mean", value_scale: float = 1.0) -> int:
    key, value, metadata = extract_next_token_memory(
        model,
        tokenizer,
        prompt,
        target_token,
        slot=slot,
        query_mode=query_mode,
        value_scale=value_scale,
    )
    return store.write(key, value, metadata=metadata, namespace=namespace, kind=kind)


