"""Frame-shaped tensors and token biases for substrate-owned grafts."""

from __future__ import annotations

from typing import Any

import torch

from ..frame import CognitiveFrame


class FrameGraftProjection:
    """Sketch / pack Broca vectors and derive content logit boosts from a frame."""

    def __init__(self, mind: Any) -> None:
        self._mind = mind

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
