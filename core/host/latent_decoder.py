"""LatentMAS / Coconut-style m-step latent rollout for the Llama host.

After the initial prompt forward, the decoder generates ``m`` continuous
"thought" steps inside Llama's latent space — no token decoding between
steps. Each step:

1. Take the previous step's last-position hidden state ``h_{t}``.
2. Project it back into Llama's input embedding distribution via the
   closed-form :class:`RidgeAlignment` (LatentMAS Wₐ).
3. Append the projected embedding as the next position; extend the
   attention mask; re-run the host's :meth:`latent_forward`.
4. Read the new ``h_{t+1}``; repeat.

Layer-post grafts continue to fire during latent rollout (substrate
contributions reach the LLM the same way they do in token-level forward).
After ``m`` steps the final hidden state is returned; callers can either
project it through ``lm_head`` for text decode or write it back into the
SWM for further substrate algebra.

LatentMAS empirically validates ``m ∈ [40, 80]`` as the productive range
when the closed-form Wₐ is in place. We default to ``m=40`` so a single
rollout adds 40 forward passes per turn — costly but bounded.
"""

from __future__ import annotations

from typing import Any

import torch

from ..grafting.alignment import RidgeAlignment
from ..workspace import WorkspacePublisher


DEFAULT_M_LATENT_STEPS: int = 40


class LatentDecoder:
    """Run ``m``-step latent rollout against a frozen Llama host."""

    def __init__(self, *, host: Any, m_latent_steps: int = DEFAULT_M_LATENT_STEPS) -> None:
        if int(m_latent_steps) <= 0:
            raise ValueError(f"LatentDecoder.m_latent_steps must be positive, got {m_latent_steps}")

        self._host = host
        self._m = int(m_latent_steps)
        self._alignment = self._build_alignment(host)

    @property
    def host(self) -> Any:
        return self._host

    @property
    def m_latent_steps(self) -> int:
        return self._m

    @property
    def alignment(self) -> RidgeAlignment:
        return self._alignment

    @torch.no_grad()
    def think(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        extra_state: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, Any]:
        """Run prompt + ``m`` latent thoughts; return ``(last_hidden[:, -1:, :], past_kv)``.

        ``last_hidden`` shape is ``[batch, 1, d_model]`` so callers can hand it
        straight back to :meth:`LlamaBrocaHost.latent_forward` for another
        round, or project to vocab via ``lm_head`` for text decode.
        """

        if input_ids.ndim != 2:
            raise ValueError(f"LatentDecoder.think requires input_ids [batch, seq], got {tuple(input_ids.shape)}")

        if not callable(getattr(self._host, "latent_forward", None)):
            raise TypeError(
                f"LatentDecoder.think: host must expose latent_forward(), got {type(self._host).__name__}"
            )

        device = next(self._host.parameters()).device
        ids = input_ids.to(device)
        mask = (
            attention_mask.to(device).bool()
            if attention_mask is not None
            else torch.ones_like(ids, dtype=torch.bool, device=device)
        )

        prompt_embeds = self._host.llm.get_input_embeddings()(ids)
        prompt_len = int(prompt_embeds.shape[1])
        full_mask_len = prompt_len + self._m

        # Pre-allocate the full attention mask once; sequential ``torch.cat``
        # on every think step is a known MPS hot path that crashes inside
        # ``at::native::cat_out_mps`` for m≳20. All positions stay attended
        # (latent thoughts are non-padded), so a precomputed all-ones mask is
        # mathematically identical to the iterative cat.
        full_mask = torch.ones(
            (mask.shape[0], full_mask_len), dtype=torch.bool, device=mask.device
        )
        full_mask[:, :prompt_len] = mask

        WorkspacePublisher.emit(
            "latent.think.start",
            {
                "m_latent_steps": self._m,
                "prompt_seq_len": prompt_len,
                "batch_size": int(prompt_embeds.shape[0]),
            },
        )

        hidden, past_kv = self._host.latent_forward(
            inputs_embeds=prompt_embeds,
            attention_mask=full_mask[:, :prompt_len],
            extra_state=extra_state,
            past_key_values=None,
        )
        last_hidden = hidden[:, -1:, :]

        for step in range(self._m):
            next_embed = self._alignment.apply(last_hidden.to(torch.float32)).to(prompt_embeds.dtype)
            hidden, past_kv = self._host.latent_forward(
                inputs_embeds=next_embed,
                attention_mask=full_mask[:, : prompt_len + step + 1],
                extra_state=extra_state,
                past_key_values=past_kv,
            )
            last_hidden = hidden[:, -1:, :]

        WorkspacePublisher.emit(
            "latent.think.complete",
            {
                "m_latent_steps": self._m,
                "final_seq_len": full_mask_len,
                "last_hidden_norm": float(last_hidden.detach().to(torch.float32).norm().item()),
            },
        )

        return last_hidden, past_kv

    @staticmethod
    def _build_alignment(host: Any) -> RidgeAlignment:
        get_in = getattr(host.llm, "get_input_embeddings", None)

        if not callable(get_in):
            raise RuntimeError(
                "LatentDecoder: host.llm must expose get_input_embeddings() (HF causal LM contract)"
            )

        embed_module = get_in()

        if embed_module is None or not hasattr(embed_module, "weight"):
            raise RuntimeError(
                "LatentDecoder: get_input_embeddings() returned a module without a .weight tensor"
            )

        w_in = embed_module.weight.detach()
        lm_head = host.lm_head

        if not hasattr(lm_head, "weight"):
            raise RuntimeError("LatentDecoder: host.lm_head has no .weight; expected nn.Linear")

        w_out = lm_head.weight.detach()

        return RidgeAlignment(name="llama.inner_latent", w_in=w_in, w_out=w_out)
