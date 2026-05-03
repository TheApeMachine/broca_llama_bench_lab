"""ChatOrchestrator — substrate-biased free-form chat reply.

The largest single block of behavior the substrate controller used to hold:
the user's last message routes through :meth:`SubstrateController.comprehend`
to obtain a cognitive frame, the frame's continuous features feed
:class:`TrainableFeatureGraft`, a derived logit-bias dict over the answer's
content subwords feeds :class:`SubstrateLogitBiasGraft`, and the LLM then
decodes a free-form reply through its own chat template — surface form,
fluency, and ordering are entirely the LLM's choice.

This file owns the orchestration. The controller's ``chat_reply`` becomes
a one-liner: ``return ChatOrchestrator(self).run(messages, ...)``.
"""

from __future__ import annotations

import logging
import math
import time
from typing import TYPE_CHECKING, Any, Callable, Sequence

import torch

from ..agent.active_inference import entropy as belief_entropy
from ..dmn import DMNConfig
from ..frame import CognitiveFrame
from .derived_strength import DerivedStrength, StrengthInputs


if TYPE_CHECKING:
    from .substrate import SubstrateController


logger = logging.getLogger(__name__)


class ChatOrchestrator:
    """Run a substrate-biased chat turn against the controller's faculties."""

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

        confidence = max(0.0, min(1.0, float(frame.confidence)))
        derived_scale = self._derived_target_snr_scale(frame)
        if derived_scale <= 0.0:
            broca_features = None
            logit_bias: dict[int, float] = {}
        else:
            broca_features = (
                mind.broca_features_from_frame(frame) if frame.intent != "unknown" else None
            )
            logit_bias = self._content_logit_bias(frame)
        eff_temperature = max(
            1e-3,
            float(temperature) * self._substrate_temperature_scale(frame, confidence),
        )
        bias_top: list[dict[str, Any]] = self._bias_preview(logit_bias)

        mind._last_chat_meta = {
            "intent": frame.intent,
            "subject": frame.subject,
            "answer": frame.answer,
            "confidence": float(confidence),
            "eff_temperature": float(eff_temperature),
            "bias_token_count": len(logit_bias),
            "bias_top": bias_top,
            "has_broca_features": broca_features is not None,
            "derived_target_snr_scale": float(derived_scale),
            "ts": time.time(),
        }
        try:
            mind.event_bus.publish("chat.start", dict(mind._last_chat_meta))
        except Exception:
            logger.exception("ChatOrchestrator.run: chat.start publish failed")

        text, gen_ids, sub_inertia = self._stream(
            msgs,
            broca_features=broca_features,
            logit_bias=logit_bias,
            max_new_tokens=int(max_new_tokens),
            do_sample=bool(do_sample),
            temperature=eff_temperature,
            top_p=float(top_p),
            on_token=on_token,
            substrate_confidence=confidence,
            substrate_target_snr_scale=float(derived_scale),
        )
        self._record_motor_replay(
            msgs,
            generated_token_ids=gen_ids,
            broca_features=broca_features,
            substrate_confidence=confidence,
            substrate_inertia=sub_inertia,
        )
        self._record_assistant_affect(text, frame, confidence)
        return frame, text

    # -- private helpers ------------------------------------------------------

    def _bias_preview(self, logit_bias: dict[int, float]) -> list[dict[str, Any]]:
        preview: list[dict[str, Any]] = []
        try:
            hf_tok = getattr(self._mind.tokenizer, "inner", None)
            if hf_tok is not None and logit_bias:
                ranked = sorted(logit_bias.items(), key=lambda kv: kv[1], reverse=True)[:8]
                for tid, val in ranked:
                    try:
                        piece = hf_tok.decode(
                            [int(tid)],
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )
                    except Exception:
                        piece = f"<{tid}>"
                    preview.append(
                        {"token_id": int(tid), "token": piece, "bias": float(val)}
                    )
        except Exception:
            logger.exception("ChatOrchestrator: bias preview extraction failed")
        return preview

    def _record_assistant_affect(
        self, text: str, frame: CognitiveFrame, confidence: float
    ) -> None:
        mind = self._mind
        assistant_affect = mind.affect_encoder.detect(text)
        if mind._last_affect is None:
            raise RuntimeError(
                "ChatOrchestrator: cannot align affect before user affect has been recorded"
            )
        affect_alignment = mind.affect_trace.alignment(mind._last_affect, assistant_affect)
        assistant_affect_trace_id = mind.affect_trace.record(
            role="assistant",
            text=text,
            affect=assistant_affect,
            response_to_id=mind._last_user_affect_trace_id,
            alignment=affect_alignment,
        )
        from .affect_evidence import AffectEvidence

        mind._last_chat_meta = {
            **mind._last_chat_meta,
            "assistant_affect": AffectEvidence.as_dict(assistant_affect),
            "affect_alignment": affect_alignment,
            "assistant_affect_trace_id": int(assistant_affect_trace_id),
            "user_affect_trace_id": mind._last_user_affect_trace_id,
        }
        try:
            mind.event_bus.publish(
                "chat.complete",
                {
                    "intent": frame.intent,
                    "confidence": float(confidence),
                    "affect_alignment": float(affect_alignment["alignment"]),
                    "reply_chars": len(text),
                    "reply_preview": text[:200],
                },
            )
        except Exception:
            logger.exception("ChatOrchestrator: chat.complete publish failed")

    def _substrate_temperature_scale(self, frame: CognitiveFrame, confidence: float) -> float:
        """Sampling temperature multiplier derived from substrate posterior entropy."""

        if frame.intent == "unknown":
            return 1.0
        try:
            coupled = self._mind.unified_agent.decide()
        except (RuntimeError, ValueError, IndexError):
            return max(1e-3, 1.0 - 0.6 * float(confidence))
        if coupled.faculty == "spatial":
            posterior = list(coupled.spatial_decision.posterior_over_policies)
        else:
            posterior = list(coupled.causal_decision.posterior_over_policies)
        n = len(posterior)
        if n < 2:
            return max(1e-3, 1.0 - 0.6 * float(confidence))
        h_q = belief_entropy(posterior)
        h_max = math.log(n)
        if h_max <= 1e-9:
            return max(1e-3, 1.0 - 0.6 * float(confidence))
        normalized_uncertainty = max(0.0, min(1.0, h_q / h_max))
        return max(1e-3, normalized_uncertainty * (1.0 - 0.6 * float(confidence)))

    def _content_logit_bias(self, frame: CognitiveFrame) -> dict[int, float]:
        """Map substrate content (subject / predicate / answer) to subword token ids."""

        if frame.intent == "unknown":
            return {}
        targets: list[str] = []
        if frame.subject:
            targets.append(str(frame.subject))
        if frame.answer and frame.answer.lower() != "unknown":
            targets.append(str(frame.answer))
        pred = (frame.evidence or {}).get("predicate") or (frame.evidence or {}).get(
            "predicate_surface"
        )
        if isinstance(pred, str) and pred:
            targets.append(pred)
        if not targets:
            return {}
        hf_tok = getattr(self._mind.tokenizer, "inner", None)
        bias: dict[int, float] = {}
        for surface in targets:
            surface = surface.strip()
            if not surface:
                continue
            ids: list[int] = []
            if hf_tok is not None and callable(getattr(hf_tok, "encode", None)):
                ids.extend(int(t) for t in hf_tok.encode(surface, add_special_tokens=False))
                ids.extend(
                    int(t) for t in hf_tok.encode(" " + surface, add_special_tokens=False)
                )
            else:
                ids.extend(int(t) for t in self._mind.tokenizer.encode(surface))
            for tid in set(ids):
                if tid < 0:
                    continue
                bias[tid] = max(bias.get(tid, 0.0), 1.0)
        return bias

    def _derived_target_snr_scale(self, frame: CognitiveFrame) -> float:
        """Compose intent / memory / conformal / affect into a graft-strength scale in ``[0, 1]``."""

        from .affect_evidence import AffectEvidence

        evidence = frame.evidence or {}
        is_actionable = bool(evidence.get("is_actionable", frame.intent != "unknown"))
        actionability = 1.0 if is_actionable else 0.0
        memory_confidence = max(0.0, min(1.0, float(frame.confidence)))
        conformal_set_size = int(evidence.get("conformal_set_size", 0) or 0)
        certainty = AffectEvidence.certainty(self._mind._last_affect)
        return float(
            DerivedStrength.compute(
                StrengthInputs(
                    intent_actionability=actionability,
                    memory_confidence=memory_confidence,
                    conformal_set_size=conformal_set_size,
                    affect_certainty=certainty,
                )
            )
        )

    def _record_motor_replay(
        self,
        messages: Sequence[dict[str, str]],
        *,
        generated_token_ids: Sequence[int],
        broca_features: torch.Tensor | None,
        substrate_confidence: float,
        substrate_inertia: float,
    ) -> None:
        """Append one training target for REM-time :class:`GraftMotorTrainer`."""

        if len(generated_token_ids) == 0:
            return
        mind = self._mind
        cap = DMNConfig().sleep_max_replay
        snap = (
            broca_features.detach().cpu().clone() if broca_features is not None else None
        )
        item: dict[str, Any] = {
            "messages": [dict(m) for m in messages],
            "speech_plan_tokens": torch.tensor(list(generated_token_ids), dtype=torch.long),
            "substrate_confidence": float(substrate_confidence),
            "substrate_inertia": float(substrate_inertia),
        }
        if snap is not None:
            item["broca_features"] = snap
        with mind._cognitive_state_lock:
            mind.motor_replay.append(item)
            if len(mind.motor_replay) > cap:
                mind.motor_replay[:] = mind.motor_replay[-cap:]

    def _stream(
        self,
        messages: Sequence[dict[str, str]],
        *,
        broca_features: torch.Tensor | None,
        logit_bias: dict[int, float],
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        on_token: Callable[[str], None] | None,
        substrate_confidence: float = 1.0,
        substrate_target_snr_scale: float = 1.0,
    ) -> tuple[str, list[int], float]:
        mind = self._mind
        hf_tok = getattr(mind.tokenizer, "inner", None)
        if hf_tok is None or not callable(getattr(hf_tok, "apply_chat_template", None)):
            raise RuntimeError(
                "ChatOrchestrator._stream requires a HuggingFace chat-template tokenizer at .tokenizer.inner"
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

        eos_id = getattr(hf_tok, "eos_token_id", None)
        current = prompt[0].tolist()
        generated: list[int] = []
        bias_active = bool(logit_bias)
        feature_tensor = broca_features.to(device) if broca_features is not None else None
        target_token_set = {int(t) for t in logit_bias.keys()} if bias_active else set()
        target_emitted = False

        past_key_values = None
        with torch.no_grad():
            for _step in range(max(1, int(max_new_tokens))):
                inertia = math.log1p(float(len(current)))
                extra: dict[str, Any] = {
                    "tokenizer": mind.tokenizer,
                    "substrate_confidence": float(substrate_confidence),
                    "substrate_inertia": float(inertia),
                    "substrate_target_snr_scale": float(substrate_target_snr_scale),
                    "return_past_key_values": True,
                }
                if feature_tensor is not None:
                    extra["broca_features"] = feature_tensor
                if bias_active:
                    semantic_decay = 0.15 if target_emitted else 1.0
                    extra["broca_logit_bias"] = logit_bias
                    extra["broca_logit_bias_decay"] = semantic_decay
                if past_key_values is not None:
                    extra["past_key_values"] = past_key_values
                    row_t = torch.tensor([[current[-1]]], device=device, dtype=torch.long)
                    mask_t = torch.ones((1, len(current)), dtype=torch.bool, device=device)
                else:
                    row_t = torch.tensor([current], device=device, dtype=torch.long)
                    mask_t = torch.ones_like(row_t, dtype=torch.bool)
                out = mind.host(row_t, mask_t, extra_state=extra)
                if isinstance(out, tuple):
                    logits, past_key_values = out
                else:
                    raise RuntimeError(
                        "LlamaBrocaHost.forward expected (logits, past_key_values) when return_past_key_values is set"
                    )
                last_pos = logits.shape[1] - 1
                logits_row = logits[0, last_pos].float()
                if do_sample:
                    scaled = logits_row / max(temperature, 1e-5)
                    probs = torch.softmax(scaled, dim=-1)
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cdf = torch.cumsum(sorted_probs, dim=-1)
                    over = (cdf > top_p).nonzero(as_tuple=False)
                    keep = int(over[0, 0].item()) + 1 if over.numel() > 0 else int(probs.numel())
                    keep = max(1, keep)
                    kept_probs = sorted_probs[:keep]
                    kept_idx = sorted_idx[:keep]
                    kept_probs = kept_probs / kept_probs.sum().clamp_min(1e-12)
                    pick = int(torch.multinomial(kept_probs, num_samples=1).item())
                    pred = int(kept_idx[pick].item())
                else:
                    pred = int(logits_row.argmax().item())
                if eos_id is not None and pred == int(eos_id):
                    break
                generated.append(pred)
                current.append(pred)
                if bias_active and not target_emitted and pred in target_token_set:
                    target_emitted = True
                if on_token is not None:
                    piece = hf_tok.decode(
                        [pred], skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )
                    if piece:
                        on_token(piece)
        reply = hf_tok.decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        inertia_tail = math.log1p(float(len(current)))
        return reply, generated, inertia_tail
