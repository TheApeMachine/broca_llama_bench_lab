"""Motor learning: continuous online training of the Broca graft only.

The LLM stays frozen — catastrophic forgetting is something we explicitly avoid.
Instead this module trains the residual-stream bridges (``TrainableFeatureGraft``,
``FeatureVectorGraft``) so the substrate progressively learns *how* to inject
its frame into this specific host's geometry.

Design choices:

* Loss = next-token cross-entropy of a small "speech plan" (the lexical tokens
  the substrate would have liked the LLM to emit) computed under the host's
  full forward pass *with* the graft active. The graft's parameters are the
  only ones that receive gradient.
* Adam with weight decay; defaults are conservative (lr=1e-4, weight_decay=
  1e-2) so a single noisy training step can't destabilize a graft that already
  works.
* Optional gradient clipping by global norm (default 1.0) to bound the worst
  per-step move.
* All host parameters are frozen explicitly *inside* the trainer regardless of
  caller state, so a misconfigured environment can't accidentally leak grads
  into the LLM.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class MotorLearningConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    max_replay_per_tick: int = 16
    min_replay_for_step: int = 4
    target_keys: Sequence[str] = field(default_factory=lambda: ("speech_plan_tokens",))


def freeze_all_but(
    parameters_to_train: Iterable[nn.Parameter], host_root: nn.Module
) -> set[int]:
    """Freeze every parameter under ``host_root`` except the ones in ``parameters_to_train``.

    Returns the set of object ids that were left trainable so callers can
    restore prior requires_grad state if desired.
    """

    keep = {id(p) for p in parameters_to_train}
    for p in host_root.parameters():
        p.requires_grad = id(p) in keep
    return keep


class GraftMotorTrainer:
    """Online trainer for the substrate's Broca grafts.

    Replay items have the shape::

        {
            "messages": [{"role": "user", "content": ...}, ...],
            "broca_features": Tensor [d_features],
            "<target_key>": LongTensor [k]    # plan tokens (keys from ``MotorLearningConfig.target_keys``)
        }

    The trainer assembles the host's chat template, forwards through the host
    *with* the grafts active, and minimizes cross-entropy of the speech-plan
    tokens at the appended-position. The same machinery doubles as a
    self-distillation loop: ``speech_plan_tokens`` can be the LLM's own
    generation under graft bias, in which case the trainer is effectively
    teaching the graft to reproduce its own (now-validated) output.
    """

    def __init__(
        self,
        host: nn.Module,
        tokenizer: Any,
        graft_modules: Sequence[nn.Module],
        *,
        config: MotorLearningConfig | None = None,
    ):
        self.host = host
        self.tokenizer = tokenizer
        self.grafts = list(graft_modules)
        self.config = config or MotorLearningConfig()
        params = [p for graft in self.grafts for p in graft.parameters()]
        freeze_all_but(parameters_to_train=params, host_root=self.host)
        self.params = params
        self.optimizer = torch.optim.AdamW(
            params,
            lr=float(self.config.learning_rate),
            weight_decay=float(self.config.weight_decay),
        )
        self.steps = 0
        self.last_loss: float | None = None

    def _plan_tensor_from_item(self, item: dict[str, Any]) -> torch.Tensor | None:
        for key in self.config.target_keys:
            plan = item.get(key)
            if plan is None:
                continue
            if isinstance(plan, torch.Tensor):
                return plan
            return torch.tensor(plan, dtype=torch.long)
        return None

    def _build_inputs(
        self, messages: Sequence[dict[str, str]], plan_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hf_tok = getattr(self.tokenizer, "inner", None)
        if hf_tok is None or not callable(getattr(hf_tok, "apply_chat_template", None)):
            raise RuntimeError(
                "motor learning requires a chat-template tokenizer at .tokenizer.inner"
            )
        device = next(self.host.parameters()).device
        prompt = hf_tok.apply_chat_template(
            list(messages), add_generation_prompt=True, return_tensors="pt"
        )
        if not isinstance(prompt, torch.Tensor):
            prompt = prompt["input_ids"]
        prompt = prompt.to(device)
        if prompt.ndim == 1:
            prompt = prompt.view(1, -1)
        plan = plan_tokens.to(device).long().view(-1)
        if plan.numel() == 0:
            raise ValueError("plan token tensor must be non-empty for a training step")
        # Sequence: prompt + plan_tokens. We supervise the plan positions.
        full = torch.cat([prompt, plan.view(1, -1)], dim=1)
        return prompt, full

    def step(self, replay: Sequence[dict[str, Any]]) -> dict[str, Any]:
        items = [r for r in replay if self._plan_tensor_from_item(r) is not None]
        if len(items) < self.config.min_replay_for_step:
            return {"skipped": True, "reason": "insufficient_replay", "n": len(items)}

        self.host.train()
        for graft in self.grafts:
            graft.train()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = torch.zeros(
            1, device=next(self.host.parameters()).device, dtype=torch.float32
        )
        contributions = 0
        for item in items[: self.config.max_replay_per_tick]:
            messages = item["messages"]
            plan = self._plan_tensor_from_item(item)
            try:
                prompt, full = self._build_inputs(messages, plan)
            except (RuntimeError, ValueError):
                logger.debug(
                    "GraftMotorTrainer.step: skipping replay item (build failed)",
                    exc_info=True,
                )
                continue
            mask = torch.ones_like(full, dtype=torch.bool)
            extra = {"tokenizer": self.tokenizer, "motor_prompt_len": int(prompt.shape[1])}
            features = item.get("broca_features")
            if isinstance(features, torch.Tensor):
                extra["broca_features"] = features.to(full.device)
            substrate_confidence = float(item.get("substrate_confidence", 1.0))
            extra["substrate_confidence"] = substrate_confidence
            extra["substrate_inertia"] = float(item.get("substrate_inertia", 1.0))
            logits = self.host(full, mask, extra_state=extra)
            # Supervise positions corresponding to plan tokens. Logits at index
            # i predict token i+1, so the supervision target for position
            # ``prompt_len + j - 1`` is plan[j].
            prompt_len = prompt.shape[1]
            plan_len = plan.numel()
            target_positions = torch.arange(
                prompt_len - 1, prompt_len - 1 + plan_len, device=full.device
            )
            preds = logits[0, target_positions]  # [plan_len, V]
            targets = plan.view(-1).to(full.device)
            loss = F.cross_entropy(preds, targets, reduction="mean")
            total_loss = total_loss + loss
            contributions += 1

        if contributions == 0:
            return {"skipped": True, "reason": "no_valid_items", "n": len(items)}

        total_loss = total_loss / float(contributions)
        total_loss.backward()
        if self.config.grad_clip is not None and self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.params, max_norm=float(self.config.grad_clip)
            )
        self.optimizer.step()
        self.steps += 1
        self.last_loss = float(total_loss.detach().item())
        for graft in self.grafts:
            graft.eval()
        self.host.eval()
        logger.info(
            "GraftMotorTrainer.step: steps=%d loss=%.4f items=%d/%d clip=%.3f",
            self.steps,
            self.last_loss,
            contributions,
            len(items),
            float(self.config.grad_clip or 0.0),
        )
        return {
            "skipped": False,
            "steps": self.steps,
            "loss": self.last_loss,
            "items": contributions,
        }
