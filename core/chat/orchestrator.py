"""Chat orchestrator."""

from __future__ import annotations

from typing import Callable, Sequence

import torch

from ..frame import CognitiveFrame


class ChatOrchestrator:
    """Conversation boundary for one chat reply.

    Each chat turn runs through three substrate stages:

    1. ``comprehend`` — comprehension pipeline produces the cognitive frame
       and publishes per-organ slots into the substrate working memory.
    2. ``recursion_controller.run`` — the closed-form latent collaboration
       loop refines the SWM ``r`` rounds, leaving the converged thought in
       the canonical ``active.thought`` slot that the
       :class:`SWMResidualGraft` reads during decode.
    3. ``chat_turn.reply`` — Llama emits text with the substrate's recursive
       thought biasing the residual stream via the closed-form
       :class:`SWMToInputProjection`.
    """

    def __init__(self, mind: "SubstrateController") -> None:
        self._mind = mind

    def run(
        self,
        messages: Sequence[dict[str, str]],
        *,
        max_new_tokens: int = 256,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9,
        on_token: Callable[[str], None] | None = None,
    ) -> tuple[CognitiveFrame, str]:
        mind = self._mind
        msgs = [dict(m) for m in messages]

        if not msgs or msgs[-1].get("role") != "user":
            raise ValueError("ChatOrchestrator.run expects messages ending with a user turn")

        user_text = str(msgs[-1].get("content", "")).strip()
        frame = mind.comprehend(user_text)

        recursion_input_ids = self._build_recursion_input_ids(msgs)
        trace = mind.recursion_controller.run(input_ids=recursion_input_ids)
        frame.evidence = {
            **dict(frame.evidence or {}),
            "recursion": {
                "rounds": trace.rounds,
                "final_thought_slot": trace.final_thought_slot,
                "final_llama_slot": trace.final_llama_slot,
                "halts": [
                    {
                        "halt": h.halt,
                        "reason": h.reason,
                        "cosine_to_previous": h.cosine_to_previous,
                        "rounds_completed": h.rounds_completed,
                    }
                    for h in trace.halts
                ],
            },
        }

        text = mind.chat_turn.reply(
            frame,
            msgs,
            max_new_tokens=int(max_new_tokens),
            do_sample=bool(do_sample),
            temperature=float(temperature),
            top_p=float(top_p),
            on_token=on_token,
        )

        return frame, text

    def _build_recursion_input_ids(self, messages: Sequence[dict[str, str]]) -> torch.Tensor:
        """Tokenize ``messages`` with the host tokenizer's chat template.

        The recursion controller drives latent rollout from this prompt; the
        graft injects the SWM thought slot so the LLM "thinks" about the
        full conversation, not just the comprehension frame.
        """

        mind = self._mind
        hf_tok = getattr(mind.tokenizer, "inner", None)

        if hf_tok is None or not callable(getattr(hf_tok, "apply_chat_template", None)):
            raise RuntimeError(
                "ChatOrchestrator: substrate tokenizer must expose .inner with apply_chat_template"
            )

        device = next(mind.host.parameters()).device
        prompt = hf_tok.apply_chat_template(
            list(messages), add_generation_prompt=True, return_tensors="pt"
        )

        if not isinstance(prompt, torch.Tensor):
            prompt = prompt["input_ids"]

        prompt = prompt.to(device)

        if prompt.ndim == 1:
            prompt = prompt.view(1, -1)

        return prompt
