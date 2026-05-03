"""Frame-shaped tensors and continuous concept directions for substrate-owned grafts."""

from __future__ import annotations

from typing import Any

import torch

from ..frame import CognitiveFrame, FrameDimensions
from ..numeric import Probability
from .chat_plan import ChatGraftPlan
from .strength import DerivedStrength, StrengthInputs


class FrameGraftProjection:
    """Pack Broca features and resolve concept token IDs for the concept graft.

    Promoted ontological axes from the registry are overlaid into the broca
    feature vector so the TrainableFeatureGraft receives the orthogonalized
    geometry of any concept the substrate has Hebbian-promoted, rather than the
    raw text-encoder sketch.
    """

    def __init__(self, mind: Any) -> None:
        self._mind = mind
        self.probability = Probability()

    def broca_features(self, frame: CognitiveFrame) -> torch.Tensor:
        mind = self._mind
        vsa_vec: torch.Tensor | None = None
        if frame.subject and frame.answer and str(frame.answer).lower() not in {"", "unknown"}:
            pr = str((frame.evidence or {}).get("predicate", frame.intent))
            vsa_vec = mind.encode_triple_vsa(str(frame.subject), pr, str(frame.answer))
        base = mind.frame_packer.broca(
            frame.intent,
            frame.subject,
            frame.answer,
            float(frame.confidence),
            frame.evidence,
            vsa_bundle=vsa_vec,
            vsa_projection_seed=int(mind.seed),
        )

        return self._overlay_orthogonal_axes(base, frame)

    def _overlay_orthogonal_axes(
        self, base: torch.Tensor, frame: CognitiveFrame
    ) -> torch.Tensor:
        ontology = getattr(self._mind, "ontology", None)

        if ontology is None:
            return base

        sketch_dim = FrameDimensions.SKETCH_DIM
        slots = (
            (str(frame.intent or ""), 0),
            (str(frame.subject or ""), sketch_dim),
            (str(frame.answer or ""), 2 * sketch_dim),
        )

        out = base.clone()

        for name, offset in slots:
            if not name or name.lower() == "unknown":
                continue

            if not ontology.is_promoted(name):
                continue

            sketch_view = base[offset : offset + sketch_dim]
            axis = ontology.vector_for(name, sketch_view).to(dtype=base.dtype, device=base.device)

            if axis.numel() != sketch_dim:
                raise ValueError(
                    f"FrameGraftProjection: ontology axis for {name!r} has dim {axis.numel()}, "
                    f"expected {sketch_dim}"
                )

            out[offset : offset + sketch_dim] = axis

        return out

    def concept_token_ids(self, frame: CognitiveFrame) -> dict[str, list[int]]:
        """Token-ID groups for the concepts the host should attract toward."""

        if frame.intent == "unknown":
            return {}

        names: list[str] = []
        if frame.subject:
            names.append(str(frame.subject))
        if frame.answer and frame.answer.lower() != "unknown":
            names.append(str(frame.answer))
        pred = (frame.evidence or {}).get("predicate") or (frame.evidence or {}).get(
            "predicate_surface"
        )
        if isinstance(pred, str) and pred:
            names.append(pred)

        return self._resolve_groups(names)

    def repulsion_token_ids(self, frame: CognitiveFrame) -> dict[str, list[int]]:
        """Token-ID groups for concepts the host should be pushed away from.

        The frame may carry banned concepts under ``evidence['banned_concepts']``
        as a sequence of strings. Empty until the substrate emits suppression
        candidates — the graft is no-op without input.
        """

        evidence = frame.evidence or {}
        banned = evidence.get("banned_concepts")

        if not banned:
            return {}

        names = [str(b) for b in banned if isinstance(b, str) and b.strip()]

        return self._resolve_groups(names)

    def _resolve_groups(self, names: list[str]) -> dict[str, list[int]]:
        mind = self._mind
        hf_tok = getattr(mind.tokenizer, "inner", None)
        groups: dict[str, list[int]] = {}

        for surface in names:
            surface = surface.strip()
            if not surface:
                continue

            ids: list[int] = []

            if hf_tok is not None and callable(getattr(hf_tok, "encode", None)):
                ids.extend(int(t) for t in hf_tok.encode(surface, add_special_tokens=False))
                ids.extend(int(t) for t in hf_tok.encode(" " + surface, add_special_tokens=False))
            else:
                ids.extend(int(t) for t in mind.tokenizer.encode(surface))

            unique = sorted({tid for tid in ids if tid >= 0})
            if unique:
                groups[surface] = unique

        return groups

    def chat_plan(
        self, frame: CognitiveFrame, *, requested_temperature: float
    ) -> ChatGraftPlan:
        confidence = self.probability.unit_interval(frame.confidence)
        derived_scale = self.derived_target_snr_scale(frame)

        if derived_scale <= 0.0:
            broca_features = None
            attract: dict[str, list[int]] = {}
            repel: dict[str, list[int]] = {}
        else:
            broca_features = (
                self.broca_features(frame) if frame.intent != "unknown" else None
            )
            attract = self.concept_token_ids(frame)
            repel = self.repulsion_token_ids(frame)

        effective_temperature = max(
            1e-3,
            float(requested_temperature) * self.substrate_temperature_scale(frame, confidence),
        )

        return ChatGraftPlan(
            frame=frame,
            confidence=confidence,
            effective_temperature=effective_temperature,
            broca_features=broca_features,
            concept_token_ids=attract,
            repulsion_token_ids=repel,
            concept_preview=self.concept_preview(attract, repel),
            derived_target_snr_scale=derived_scale,
        )

    def concept_preview(
        self,
        attract: dict[str, list[int]],
        repel: dict[str, list[int]],
    ) -> list[dict[str, int | str | float]]:
        if not attract and not repel:
            return []

        hf_tok = getattr(self._mind.tokenizer, "inner", None)
        if hf_tok is None:
            raise RuntimeError("FrameGraftProjection.concept_preview requires tokenizer.inner")

        preview: list[dict[str, int | str | float]] = []

        for kind, groups in (("attract", attract), ("repel", repel)):
            for name, token_ids in list(groups.items())[:4]:
                pieces = hf_tok.decode(
                    [int(t) for t in token_ids[:4]],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                preview.append(
                    {
                        "kind": kind,
                        "concept": str(name),
                        "tokens_preview": str(pieces),
                        "token_count": int(len(token_ids)),
                    }
                )

        return preview

    def derived_target_snr_scale(self, frame: CognitiveFrame) -> float:
        return float(
            DerivedStrength.compute(
                StrengthInputs.from_frame(frame, self._mind.session.last_affect)
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
