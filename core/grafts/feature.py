"""Frame-shaped tensors and token biases for substrate-owned grafts."""

from __future__ import annotations

from typing import Any

import torch

from ..affect.evidence import AffectEvidence
from ..frame import CognitiveFrame
from ..numeric import Probability
from .chat_plan import ChatGraftPlan
from .strength import DerivedStrength, StrengthEvidence, StrengthInputs
from .token_bias import TokenBias


class FrameGraftProjection:
    """Sketch / pack Broca vectors and derive content logit boosts from a frame."""

    def __init__(self, mind: Any) -> None:
        self._mind = mind
        self.probability = Probability()

    def broca_features(self, frame: CognitiveFrame) -> torch.Tensor:
        mind = self._mind
        vsa_vec: torch.Tensor | None = None
        if frame.subject and frame.answer and str(frame.answer).lower() not in {"", "unknown"}:
            pr = str((frame.evidence or {}).get("predicate", frame.intent))
            vsa_vec = mind.encode_triple_vsa(str(frame.subject), pr, str(frame.answer))
        return mind.frame_packer.broca(
            frame.intent,
            frame.subject,
            frame.answer,
            float(frame.confidence),
            frame.evidence,
            vsa_bundle=vsa_vec,
            vsa_projection_seed=int(mind.seed),
        )

    def content_logit_bias(self, frame: CognitiveFrame) -> dict[int, float]:
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
        mind = self._mind
        hf_tok = getattr(mind.tokenizer, "inner", None)
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
                ids.extend(int(t) for t in mind.tokenizer.encode(surface))
            for tid in set(ids):
                if tid < 0:
                    continue
                bias[tid] = max(bias.get(tid, 0.0), 1.0)
        return bias

    def chat_plan(
        self, frame: CognitiveFrame, *, requested_temperature: float
    ) -> ChatGraftPlan:
        confidence = self.probability.unit_interval(frame.confidence)
        derived_scale = self.derived_target_snr_scale(frame)

        if derived_scale <= 0.0:
            broca_features = None
            logit_bias: dict[int, float] = {}
        else:
            broca_features = (
                self.broca_features(frame) if frame.intent != "unknown" else None
            )
            logit_bias = self.content_logit_bias(frame)

        effective_temperature = max(
            1e-3,
            float(requested_temperature) * self.substrate_temperature_scale(frame, confidence),
        )

        return ChatGraftPlan(
            frame=frame,
            confidence=confidence,
            effective_temperature=effective_temperature,
            broca_features=broca_features,
            logit_bias=logit_bias,
            bias_top=self.bias_preview(logit_bias),
            derived_target_snr_scale=derived_scale,
        )

    def bias_preview(self, logit_bias: dict[int, float]) -> list[TokenBias]:
        if not logit_bias:
            return []

        hf_tok = getattr(self._mind.tokenizer, "inner", None)
        if hf_tok is None:
            raise RuntimeError("FrameGraftProjection.bias_preview requires tokenizer.inner")

        preview: list[TokenBias] = []
        ranked = sorted(logit_bias.items(), key=lambda kv: kv[1], reverse=True)[:8]
        for token_id, bias in ranked:
            piece = hf_tok.decode(
                [int(token_id)],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            preview.append(
                TokenBias(token_id=int(token_id), token=piece, bias=float(bias))
            )
        return preview

    def derived_target_snr_scale(self, frame: CognitiveFrame) -> float:
        evidence = StrengthEvidence.from_frame(frame)
        memory_confidence = self.probability.unit_interval(frame.confidence)
        certainty = AffectEvidence.certainty(self._mind.session.last_affect)
        return float(
            DerivedStrength.compute(
                StrengthInputs(
                    intent_actionability=evidence.actionability,
                    memory_confidence=memory_confidence,
                    conformal_set_size=evidence.conformal_set_size,
                    affect_certainty=certainty,
                )
            )
        )

    def substrate_temperature_scale(
        self, frame: CognitiveFrame, confidence: float
    ) -> float:
        if frame.intent == "unknown":
            return 1.0

        coupled = self._mind.unified_agent.decide()
        if coupled.faculty == "spatial":
            posterior = list(coupled.spatial_decision.posterior_over_policies)
        else:
            posterior = list(coupled.causal_decision.posterior_over_policies)

        return self.probability.temperature_scale(
            confidence=confidence,
            posterior=posterior,
        )
