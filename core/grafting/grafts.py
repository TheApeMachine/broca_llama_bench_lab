from __future__ import annotations

import logging
import math
from typing import Iterable, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..agent.active_inference import CoupledDecision, CoupledEFEAgent

logger = logging.getLogger(__name__)


def derived_residual_token_strength(d_model: int, n_outcomes: int) -> float:
    """Scale for injecting a discrete faculty direction into d_model-space.

    Matches the order of KVMemoryGraft writing magnitude: proportional to ``d_model``
    times a slowly-growing factor in the number of distinct outcomes (actions,
    binary causal labels, etc.).
    """

    dm = float(int(d_model))
    return dm * math.sqrt(math.log1p(float(max(1, int(n_outcomes)))))


DEFAULT_GRAFT_TARGET_SNR = 0.30


def host_rms(x: torch.Tensor) -> torch.Tensor:
    """Root-mean-square energy of the host residual stream over its feature axis.

    Returned shape preserves all leading dims and keeps the feature axis at size 1
    so the result broadcasts cleanly against the unit-norm direction added back in.
    """

    return x.detach().float().pow(2).mean(dim=-1, keepdim=True).sqrt().to(dtype=x.dtype)


def snr_magnitude(
    x: torch.Tensor, *, target_snr: float, confidence: float = 1.0, inertia: float = 1.0
) -> torch.Tensor:
    """Magnitude that injects a unit direction at ``target_snr`` × host RMS.

    Confidence and context-inertia are multiplicative on top: high-confidence
    substrate frames push proportionally harder, and the bias scales with the
    autoregressive prefix length so grafts can shout over an LLM that has built
    up momentum on a competing surface form.
    """

    ts = max(0.0, float(target_snr))
    return host_rms(x) * ts * float(max(0.0, confidence)) * float(max(0.0, inertia))


def _state_confidence(state: dict) -> float:
    val = state.get("substrate_confidence")
    try:
        return float(val) if val is not None else 1.0
    except (TypeError, ValueError):
        return 1.0


def _state_inertia(state: dict) -> float:
    val = state.get("substrate_inertia")
    try:
        return float(val) if val is not None else 1.0
    except (TypeError, ValueError):
        return 1.0


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
        return torch.full(
            (x.shape[0],), x.shape[1] - 1, device=x.device, dtype=torch.long
        )
    return mask.long().sum(dim=1).clamp_min(1).to(x.device) - 1


def _trigger_mask(
    token_ids: torch.Tensor, trigger_ids: Sequence[int] | None
) -> torch.Tensor:
    if not trigger_ids:
        return torch.ones(token_ids.shape[0], device=token_ids.device, dtype=torch.bool)
    mask = torch.zeros(token_ids.shape[0], device=token_ids.device, dtype=torch.bool)
    for tid in trigger_ids:
        mask |= (token_ids == int(tid)).any(dim=1)
    return mask


class KVMemoryGraft(BaseGraft):
    """Content-addressable memory grafted into a residual stream.

    Strength, temperature, and neighbor count derive from ``d_model`` and store
    size. Gating uses only geometry: (1) how peaked the query–key similarities
    are relative to their mean, and (2) whether the best key match exceeds the
    typical *inter-key* similarity of the loaded store—so prompts that sit
    outside the memory manifold are not driven by spurious nearest neighbors.
    """

    def __init__(
        self,
        d_model: int,
        *,
        max_items: int = 8192,
        query_mode: str = "sequence_mean",
        target_snr: float = DEFAULT_GRAFT_TARGET_SNR,
    ):
        super().__init__()
        if query_mode not in {"token", "sequence_mean"}:
            raise ValueError("query_mode must be 'token' or 'sequence_mean'")
        self.d_model = int(d_model)
        dm = float(self.d_model)
        log_store = math.log1p(float(max_items))
        self.target_snr = float(target_snr)
        self.temperature = math.sqrt(log_store) / max(dm, 1.0)
        self.max_items = int(max_items)
        self.query_mode = query_mode
        self.register_buffer("keys", torch.empty(0, self.d_model))
        self.register_buffer("values", torch.empty(0, self.d_model))
        self.metadata: list[dict] = []
        self.last_debug: dict = {}
        self.spread_matrix: torch.Tensor | None = None
        self._manifold_tau: float | None = None
        self._manifold_sigma: float | None = None

    def clear(self) -> None:
        self.keys = self.keys.new_empty(0, self.d_model)
        self.values = self.values.new_empty(0, self.d_model)
        self.metadata.clear()
        self.last_debug = {}
        self.spread_matrix = None
        self._manifold_tau = None
        self._manifold_sigma = None

    def set_spread_matrix(self, mat: torch.Tensor | None) -> None:
        """Row-stochastic operator indexed like ``remember`` order (from activation store)."""

        if mat is None or mat.numel() == 0:
            self.spread_matrix = None
            logger.debug("KVMemoryGraft.set_spread_matrix: cleared")
            return
        self.spread_matrix = mat.detach().float().clone()
        logger.debug(
            "KVMemoryGraft.set_spread_matrix: shape=%s", tuple(self.spread_matrix.shape)
        )

    @torch.no_grad()
    def remember(
        self, key: torch.Tensor, value: torch.Tensor, metadata: dict | None = None
    ) -> None:
        key = key.detach().reshape(-1, self.d_model).to(self.keys.device)
        value = value.detach().reshape(-1, self.d_model).to(self.values.device)
        if key.shape[0] != value.shape[0]:
            raise ValueError("key and value must contain the same number of rows")
        self.keys = torch.cat([self.keys, key], dim=0)[-self.max_items :]
        self.values = torch.cat([self.values, value], dim=0)[-self.max_items :]
        for _ in range(key.shape[0]):
            self.metadata.append(dict(metadata or {}))
        self.metadata = self.metadata[-self.max_items :]
        logger.debug(
            "KVMemoryGraft.remember: added_rows=%d store_size=%d d_model=%d",
            int(key.shape[0]),
            int(self.keys.shape[0]),
            self.d_model,
        )
        self._recompute_manifold_cache()

    def _recompute_manifold_cache(self) -> None:
        """Refresh inter-key manifold stats (O(n^2)); call only when the key store changes."""

        nk = int(self.keys.shape[0])
        if nk < 2:
            self._manifold_tau = None
            self._manifold_sigma = None
            return
        k = F.normalize(self.keys, dim=-1)
        gram = k @ k.T
        eye = torch.eye(nk, device=gram.device, dtype=torch.bool)
        masked = gram.masked_fill(eye, torch.finfo(gram.dtype).min)
        neighbor_max = masked.max(dim=-1).values
        tau = neighbor_max.median()
        sigma = neighbor_max.std(unbiased=False).clamp_min(1e-6)
        self._manifold_tau = float(tau.detach().cpu())
        self._manifold_sigma = float(sigma.detach().cpu())

    def _retrieve(
        self,
        queries: torch.Tensor,
        x: torch.Tensor,
        *,
        host_at_query: torch.Tensor,
        confidence: float = 1.0,
        inertia: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        q = F.normalize(queries, dim=-1)
        k = F.normalize(self.keys.to(x.device), dim=-1)
        raw_sims = q @ k.T
        nk = raw_sims.shape[-1]
        sqrt_d = math.sqrt(float(queries.shape[-1]))

        mean_raw = raw_sims.mean(dim=-1)
        std_raw = raw_sims.std(dim=-1, unbiased=False).clamp_min(1e-6)
        max_raw = raw_sims.max(dim=-1).values
        z_peak = (max_raw - mean_raw) / std_raw
        gate_peak = torch.sigmoid(z_peak * sqrt_d)

        if nk >= 2:
            tau_f = self._manifold_tau
            sigma_f = self._manifold_sigma
            if tau_f is None or sigma_f is None:
                self._recompute_manifold_cache()
                tau_f = self._manifold_tau
                sigma_f = self._manifold_sigma
            if tau_f is None or sigma_f is None:
                gate_manifold = torch.ones(
                    raw_sims.shape[0], device=x.device, dtype=raw_sims.dtype
                )
                manifold_dbg = {"tau_key_manifold": 0.0, "sigma_key_manifold": 1.0}
            else:
                gate_manifold = torch.sigmoid(
                    (max_raw - tau_f) / sigma_f * sqrt_d
                )
                manifold_dbg = {
                    "tau_key_manifold": tau_f,
                    "sigma_key_manifold": sigma_f,
                }
        else:
            gate_manifold = torch.ones(
                raw_sims.shape[0], device=x.device, dtype=raw_sims.dtype
            )
            manifold_dbg = {"tau_key_manifold": 0.0, "sigma_key_manifold": 1.0}

        sims = raw_sims.clone()
        eff_k = min(
            nk, max(2, int(math.ceil(math.sqrt(float(nk)) * math.log1p(float(nk)))))
        )
        if eff_k < nk:
            vals, idx = sims.topk(eff_k, dim=-1)
            masked_row = torch.full_like(sims, torch.finfo(sims.dtype).min)
            sims = masked_row.scatter(-1, idx, vals)
        weights = F.softmax(sims / max(self.temperature, 1e-6), dim=-1)
        sm = self.spread_matrix
        if sm is not None and sm.shape == (nk, nk):
            sm_dev = sm.to(device=weights.device, dtype=weights.dtype)
            weights = torch.matmul(weights, sm_dev)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        retrieved = weights @ self.values.to(x.device)
        gate = (gate_peak * gate_manifold).unsqueeze(-1)
        eps = 1e-12
        rnorm = torch.norm(retrieved, dim=-1, keepdim=True)
        direction = torch.where(
            rnorm < eps, torch.zeros_like(retrieved), retrieved / rnorm.clamp_min(eps)
        )
        magnitude = snr_magnitude(
            host_at_query,
            target_snr=self.target_snr,
            confidence=confidence,
            inertia=inertia,
        )
        delta = direction * magnitude * gate
        return delta, weights, gate.squeeze(-1), manifold_dbg

    def forward(self, x: torch.Tensor, state: dict) -> torch.Tensor:
        if not self.enabled or self.keys.numel() == 0:
            return x
        bsz, seq_len, d_model = x.shape
        nk = int(self.keys.shape[0])
        mask = state.get("attention_mask")
        if mask is None:
            mask = torch.ones(bsz, seq_len, device=x.device, dtype=torch.bool)
        confidence = _state_confidence(state)
        inertia = _state_inertia(state)
        if self.query_mode == "token":
            host_at_query = x.reshape(-1, d_model)
            delta, weights, gate, manifold_dbg = self._retrieve(
                host_at_query,
                x,
                host_at_query=host_at_query,
                confidence=confidence,
                inertia=inertia,
            )
            self.last_debug = {
                "weights": weights.detach().cpu(),
                "gate": gate.detach().cpu(),
                **manifold_dbg,
            }
            delta_view = delta.reshape(bsz, seq_len, d_model)
            gmax = float(gate.detach().max().item()) if gate.numel() else 0.0
            wmax = float(weights.detach().max().item()) if weights.numel() else 0.0
            logger.debug(
                "KVMemoryGraft.forward: mode=token bsz=%d seq=%d nk=%s gate_max=%.4f weight_max=%.4f dbg=%s",
                bsz,
                seq_len,
                nk,
                gmax,
                wmax,
                {
                    k: (float(v) if hasattr(v, "item") else v)
                    for k, v in manifold_dbg.items()
                },
            )
            return x + delta_view

        weights_for_mean = mask.to(x.dtype).unsqueeze(-1)
        lengths = weights_for_mean.sum(dim=1).clamp_min(1.0)
        queries = (x * weights_for_mean).sum(dim=1) / lengths
        last = _last_indices(state, x)
        host_at_last = x[torch.arange(bsz, device=x.device), last]
        delta, weights, gate, manifold_dbg = self._retrieve(
            queries,
            x,
            host_at_query=host_at_last,
            confidence=confidence,
            inertia=inertia,
        )
        out = x.clone()
        out[torch.arange(bsz, device=x.device), last] += delta
        self.last_debug = {
            "weights": weights.detach().cpu(),
            "gate": gate.detach().cpu(),
            **manifold_dbg,
        }
        gmax = float(gate.detach().max().item()) if gate.numel() else 0.0
        wmax = float(weights.detach().max().item()) if weights.numel() else 0.0
        logger.debug(
            "KVMemoryGraft.forward: mode=sequence_mean bsz=%d seq=%d nk=%s gate_max=%.4f weight_max=%.4f spread=%s dbg=%s",
            bsz,
            seq_len,
            nk,
            gmax,
            wmax,
            None if self.spread_matrix is None else tuple(self.spread_matrix.shape),
            {
                k: (float(v) if hasattr(v, "item") else v)
                for k, v in manifold_dbg.items()
            },
        )
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
        nn.init.kaiming_uniform_(self.A, a=5**0.5)

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
        if (
            name
            and isinstance(module, nn.Linear)
            and any(s in name for s in target_substrings)
        ):
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

    def __init__(
        self,
        d_features: int,
        d_model: int,
        *,
        trigger_ids: Sequence[int] | None = None,
        target_snr: float = DEFAULT_GRAFT_TARGET_SNR,
    ):
        super().__init__()
        self.d_features = int(d_features)
        self.d_model = int(d_model)
        self.trigger_ids = (
            tuple(int(x) for x in trigger_ids) if trigger_ids else tuple()
        )
        self.target_snr = float(target_snr)
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
            raise ValueError(
                f"expected faculty_features dim {self.d_features}, got {feats.shape[-1]}"
            )
        applies = _trigger_mask(state["token_ids"], self.trigger_ids)
        if not bool(applies.any()):
            return x
        confidence = _state_confidence(state)
        inertia = _state_inertia(state)
        last = _last_indices(state, x)
        rows = torch.arange(x.shape[0], device=x.device)[applies]
        last_apply = last[applies]
        host_at_last = x[rows, last_apply]
        direction = F.normalize(self.project(self.norm(feats[applies])), dim=-1)
        magnitude = snr_magnitude(
            host_at_last,
            target_snr=self.target_snr,
            confidence=confidence,
            inertia=inertia,
        )
        out = x.clone()
        out[rows, last_apply] += direction * magnitude
        return out


class TriggeredTokenDirectionGraft(BaseGraft):
    """Injects a selected token's embedding direction into the final hidden state.

    This is a compact bridge from a non-neural faculty into the neural host. The
    faculty computes a discrete answer/action, and the graft writes the
    corresponding token direction into the residual stream.
    """

    def __init__(
        self,
        token_by_name: Mapping[str, int],
        *,
        trigger_ids: Sequence[int] | None = None,
        target_snr: float = DEFAULT_GRAFT_TARGET_SNR,
    ):
        super().__init__()
        self.token_by_name = {str(k): int(v) for k, v in token_by_name.items()}
        self.trigger_ids = (
            tuple(int(x) for x in trigger_ids) if trigger_ids else tuple()
        )
        self.target_snr = float(target_snr)
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
        confidence = _state_confidence(state)
        inertia = _state_inertia(state)
        out = x.clone()
        model = state["model"]
        tok_id = self.token_by_name[name]
        direction = F.normalize(
            model.lm_head.weight[tok_id].detach().to(x.device, x.dtype), dim=0
        )
        last = _last_indices(state, x)
        rows = torch.arange(x.shape[0], device=x.device)[applies]
        last_apply = last[applies]
        host_at_last = x[rows, last_apply]
        magnitude = snr_magnitude(
            host_at_last,
            target_snr=self.target_snr,
            confidence=confidence,
            inertia=inertia,
        )
        out[rows, last_apply] += direction.unsqueeze(0) * magnitude
        self.last_name = name
        self.last_token_id = tok_id
        return out


class ActiveInferenceTokenGraft(TriggeredTokenDirectionGraft):
    """Projects an active-inference policy decision into the residual stream."""

    def __init__(
        self,
        agent,
        token_by_action: Mapping[str, int],
        *,
        trigger_ids: Sequence[int] | None = None,
        target_snr: float = DEFAULT_GRAFT_TARGET_SNR,
    ):
        super().__init__(
            token_by_action,
            trigger_ids=trigger_ids,
            target_snr=target_snr,
        )
        self.agent = agent
        self.last_decision = None

    def choose_name(self, state: dict) -> str | None:
        decision = self.agent.decide()
        self.last_decision = decision
        return decision.action_name


class CoupledActiveInferenceTokenGraft(TriggeredTokenDirectionGraft):
    """Arbitrates spatial vs causal active-inference faculties in one forward pass."""

    def __init__(
        self,
        coupled: CoupledEFEAgent,
        token_by_action: Mapping[str, int],
        *,
        trigger_ids: Sequence[int] | None = None,
        target_snr: float = DEFAULT_GRAFT_TARGET_SNR,
    ):
        super().__init__(
            token_by_action,
            trigger_ids=trigger_ids,
            target_snr=target_snr,
        )
        self.coupled = coupled
        self.last_coupled: CoupledDecision | None = None

    def choose_name(self, state: dict) -> str | None:
        decision = self.coupled.decide()
        self.last_coupled = decision
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
        target_snr: float = DEFAULT_GRAFT_TARGET_SNR,
    ):
        super().__init__(
            {"helps": int(positive_token), "hurts": int(negative_token)},
            trigger_ids=trigger_ids,
            target_snr=target_snr,
        )
        self.scm = scm
        self.treatment = treatment
        self.outcome = outcome
        self.outcome_value = outcome_value
        self.treatment_values = treatment_values
        self.last_effects: dict[str, float] = {}

    def choose_name(self, state: dict) -> str:
        t_pos, t_neg = self.treatment_values
        p_pos = self.scm.probability(
            {self.outcome: self.outcome_value},
            given={},
            interventions={self.treatment: t_pos},
        )
        p_neg = self.scm.probability(
            {self.outcome: self.outcome_value},
            given={},
            interventions={self.treatment: t_neg},
        )
        self.last_effects = {
            "p_do_positive": p_pos,
            "p_do_negative": p_neg,
            "ate": p_pos - p_neg,
        }
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
    if batch.attention_mask.shape[0] != 1:
        raise ValueError(
            f"extract_next_token_memory expects a single prompt (batch size 1); got batch={batch.attention_mask.shape[0]}"
        )
    model.eval()
    with model.grafts_enabled(False):
        _, cache = model(batch.ids, batch.attention_mask, return_cache=True)
    hidden = cache[f"{slot}.pre"]
    if query_mode == "sequence_mean":
        mask = batch.attention_mask.to(hidden.dtype).unsqueeze(-1)
        seq_len_sum = mask.sum(dim=1).clamp_min(1.0)
        key = (hidden * mask).sum(dim=1)[0] / seq_len_sum[0]
    elif query_mode == "token":
        last_pos = int(batch.lengths[0].item()) - 1
        key = hidden[0, last_pos]
    else:
        raise ValueError("query_mode must be 'token' or 'sequence_mean'")
    target_id = tokenizer.token_to_id[target_token]
    value = F.normalize(model.lm_head.weight[target_id].detach(), dim=0) * value_scale
    metadata = {
        "prompt": prompt,
        "target": target_token,
        "slot": slot,
        "query_mode": query_mode,
    }
    return key.detach().cpu(), value.detach().cpu(), metadata


@torch.no_grad()
def memorize_next_token(
    model,
    tokenizer,
    graft: KVMemoryGraft,
    prompt: str,
    target_token: str,
    *,
    slot: str = "final_hidden",
    value_scale: float = 1.0,
) -> None:
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
def memorize_persistent_next_token(
    store,
    model,
    tokenizer,
    prompt: str,
    target_token: str,
    *,
    namespace: str | None = None,
    kind: str = "fact",
    slot: str = "final_hidden",
    query_mode: str = "sequence_mean",
    value_scale: float = 1.0,
) -> int:
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
