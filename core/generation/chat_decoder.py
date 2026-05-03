"""Autoregressive chat decoder."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import torch

from ..grafts.chat_plan import ChatGraftPlan
from ..numeric import Sampling, SequenceGrowth


class ChatDecoder:
    """Runs the host decode loop for one planned chat reply."""

    def __init__(self, *, host: Any, tokenizer: Any) -> None:
        self._host = host
        self._tokenizer = tokenizer
        self.sampling = Sampling()
        self.sequence = SequenceGrowth()

    def stream(
        self,
        messages: Sequence[dict[str, str]],
        *,
        plan: ChatGraftPlan,
        max_new_tokens: int,
        do_sample: bool,
        top_p: float,
        on_token: Callable[[str], None] | None,
    ) -> tuple[str, list[int], float]:
        hf_tok = getattr(self._tokenizer, "inner", None)
        if hf_tok is None or not callable(getattr(hf_tok, "apply_chat_template", None)):
            raise RuntimeError(
                "ChatDecoder.stream requires a HuggingFace chat-template tokenizer at .tokenizer.inner"
            )

        device = next(self._host.parameters()).device
        prompt = hf_tok.apply_chat_template(
            list(messages), add_generation_prompt=True, return_tensors="pt"
        )
        if not isinstance(prompt, torch.Tensor):
            prompt = prompt["input_ids"]
        prompt = prompt.to(device)
        if prompt.ndim == 1:
            prompt = prompt.view(1, -1)

        eos_id = getattr(hf_tok, "eos_token_id", None)
        current = prompt[0].tolist()
        generated: list[int] = []
        feature_tensor = (
            plan.broca_features.to(device) if plan.broca_features is not None else None
        )
        attract_tokens = {
            str(name): [int(t) for t in tids]
            for name, tids in (plan.concept_token_ids or {}).items()
        }
        repel_tokens = {
            str(name): [int(t) for t in tids]
            for name, tids in (plan.repulsion_token_ids or {}).items()
        }

        past_key_values = None
        with torch.no_grad():
            for _step in range(max(1, int(max_new_tokens))):
                extra_state: dict[str, Any] = {
                    "tokenizer": self._tokenizer,
                    "substrate_confidence": float(plan.confidence),
                    "substrate_inertia": self.sequence.inertia(len(current)),
                    "substrate_target_snr_scale": float(plan.derived_target_snr_scale),
                    "return_past_key_values": True,
                }
                if feature_tensor is not None:
                    extra_state["broca_features"] = feature_tensor
                if attract_tokens:
                    extra_state["broca_concept_token_ids"] = attract_tokens
                if repel_tokens:
                    extra_state["broca_repulsion_token_ids"] = repel_tokens
                if past_key_values is not None:
                    extra_state["past_key_values"] = past_key_values

                if past_key_values is not None:
                    row_t = torch.tensor([[current[-1]]], device=device, dtype=torch.long)
                    mask_t = torch.ones((1, len(current)), dtype=torch.bool, device=device)
                else:
                    row_t = torch.tensor([current], device=device, dtype=torch.long)
                    mask_t = torch.ones_like(row_t, dtype=torch.bool)

                out = self._host(row_t, mask_t, extra_state=extra_state)
                if not isinstance(out, tuple):
                    raise RuntimeError(
                        "LlamaBrocaHost.forward expected (logits, past_key_values) when return_past_key_values is set"
                    )
                logits, past_key_values = out
                logits_row = logits[0, logits.shape[1] - 1].float()
                pred = self.sampling.next_token(
                    logits_row,
                    do_sample=do_sample,
                    temperature=plan.effective_temperature,
                    top_p=top_p,
                )
                if eos_id is not None and pred == int(eos_id):
                    break

                generated.append(pred)
                current.append(pred)
                if on_token is not None:
                    piece = hf_tok.decode(
                        [pred],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    if piece:
                        on_token(piece)

        reply = hf_tok.decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return reply, generated, self.sequence.inertia(len(current))
