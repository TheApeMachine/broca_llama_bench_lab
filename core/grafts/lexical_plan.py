"""LexicalPlanGraft — write the planned next-token direction into the residual stream.

This is the cleanest Broca analogy in the lab: the cognitive substrate decides
the lexical content; this graft turns the intended lexical sequence into
hidden-state directions the frozen language host can emit.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..grafting.grafts import (
    BaseGraft,
    DEFAULT_GRAFT_TARGET_SNR,
    snr_magnitude,
    state_confidence,
    state_inertia,
    state_target_snr_scale,
)


class LexicalPlanGraft(BaseGraft):
    """Writes a planned next word into the frozen host's residual stream.

    Reads from ``state``:

    * ``broca_plan_token_ids`` — ``LongTensor[B, T]`` (or 1-D, broadcast over
      batch) of planned token ids in canonical order.
    * ``broca_step`` — current generation step; clamps to the plan length.
    * ``model``, ``last_indices``, ``tokenizer`` — required surfaces from the
      host so the graft can read ``lm_head.weight`` and decode the picked id
      for diagnostic logging.
    """

    def __init__(self, *, target_snr: float = DEFAULT_GRAFT_TARGET_SNR):
        super().__init__()
        self.target_snr = float(target_snr)
        self.mixer_priority = 2.0
        self.last_token_id: int | None = None
        self.last_token: str | None = None

    def forward(self, x: torch.Tensor, state: dict) -> torch.Tensor:
        if not self.enabled or "broca_plan_token_ids" not in state:
            return x

        plan = state["broca_plan_token_ids"]
        
        if not isinstance(plan, torch.Tensor):
            plan = torch.tensor(plan, device=x.device, dtype=torch.long)
        
        plan = plan.to(x.device)
        
        if plan.ndim == 1:
            plan = plan.view(1, -1).expand(x.shape[0], -1)
        
        step = state.get("broca_step", 0)
        
        if not isinstance(step, torch.Tensor):
            step = torch.full((x.shape[0],), int(step), device=x.device, dtype=torch.long)
        
        step = step.to(x.device).long().view(-1)
        step = step.clamp_min(0).clamp_max(plan.shape[1] - 1)
        target_ids = plan[torch.arange(x.shape[0], device=x.device), step]
        host_model = state.get("model")
        last_raw = state.get("last_indices")
        
        if host_model is None or last_raw is None:
            missing = [
                k
                for k, v in (("model", host_model), ("last_indices", last_raw))
                if v is None
            ]
        
            raise ValueError(
                f"LexicalPlanGraft.forward: missing required state key(s): {', '.join(missing)}"
            )
        
        directions = F.normalize(
            host_model.lm_head.weight[target_ids].detach().to(x.device, x.dtype),
            dim=-1,
        )
        
        last = last_raw.to(x.device)
        rows = torch.arange(x.shape[0], device=x.device)
        host_at_last = x[rows, last]
        
        magnitude = snr_magnitude(
            host_at_last,
            target_snr=self.target_snr,
            confidence=state_confidence(state),
            inertia=state_inertia(state),
            substrate_scale=state_target_snr_scale(state),
        )
        
        out = x.clone()
        out[rows, last] += directions * magnitude
        self.last_token_id = int(target_ids[0].item())
        tok = getattr(state.get("tokenizer", None), "decode_id", None)
        self.last_token = tok(self.last_token_id) if callable(tok) else None
        
        return out
