"""CognitiveRouter — always-on faculty router over memory, action, and causal candidates.

Each faculty receives every utterance and emits a scored latent frame. The
selected frame is the highest-precision candidate above a low relevance
floor; otherwise the workspace still receives an ``unknown`` frame with the
candidate traces attached.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import TYPE_CHECKING, Sequence

from ..cognition.constants import SEMANTIC_CONFIDENCE_FLOOR
from ..cognition.intent_gate import UtteranceIntent
from ..cognition.predictive_coding import lexical_surprise_gap
from ..frame import CognitiveFrame, FacultyCandidate, ParsedClaim, ParsedQuery
from .memory_query_parser import MemoryQueryParser
from .scm_target_picker import SCMTargetPicker
from .text_relevance import TextRelevance
from .tokens import LexicalTokens


if TYPE_CHECKING:
    from ..substrate.controller import SubstrateController


logger = logging.getLogger(__name__)


class CognitiveRouter:
    """Always-on faculty router."""

    def __init__(self, *, extractor, relevance_floor: float = 0.28):
        self.relevance_floor = float(relevance_floor)
        self.extractor = extractor

    def route(
        self,
        mind: "SubstrateController",
        utterance: str,
        toks: Sequence[str],
        *,
        utterance_intent: UtteranceIntent,
    ) -> CognitiveFrame:
        candidates: list[FacultyCandidate] = []
        query = MemoryQueryParser.parse(
            toks,
            utterance=utterance,
            known_subjects=mind.memory.subjects(),
            records_for_subject=mind.memory.records_for_subject,
            text_encoder=mind.text_encoder,
        )

        if utterance_intent.allows_storage:
            candidates.append(
                FacultyCandidate(
                    "memory_ingest_pending",
                    1.45,
                    lambda: self._memory_ingest_pending(utterance, toks),
                )
            )
        if query is not None:
            candidates.append(
                FacultyCandidate(
                    "semantic_query",
                    1.35,
                    lambda query=query: self._memory_query(mind, utterance, toks, query),
                )
            )

        active_frame = self._active_action(mind)
        active_score = 0.1 + TextRelevance.frame(
            utterance, toks, active_frame, mind.text_encoder
        )
        candidates.append(
            FacultyCandidate(
                "active_inference",
                active_score,
                lambda active_frame=active_frame: active_frame,
            )
        )

        causal_frame = self._causal_effect(mind)
        causal_score = 0.1 + TextRelevance.frame(
            utterance, toks, causal_frame, mind.text_encoder
        )
        candidates.append(
            FacultyCandidate(
                "causal_effect",
                causal_score,
                lambda causal_frame=causal_frame: causal_frame,
            )
        )

        ranked = sorted(candidates, key=lambda c: c.score, reverse=True)
        selected = (
            ranked[0] if ranked and ranked[0].score >= self.relevance_floor else None
        )
        frame = (
            selected.build()
            if selected is not None
            else CognitiveFrame(
                "unknown",
                answer="unknown",
                confidence=0.0,
                evidence={"route": "none"},
            )
        )
        frame.evidence = {
            **dict(frame.evidence),
            "router_selected": selected.name if selected is not None else "unknown",
            "router_candidates": [
                {"name": c.name, "score": round(float(c.score), 6)} for c in ranked
            ],
            "intrinsic_cues": [asdict(cue) for cue in mind.workspace.intrinsic_cues],
        }
        logger.debug(
            "CognitiveRouter.route: selected=%s intent=%s scores=%s utterance_preview=%r",
            selected.name if selected is not None else "unknown",
            frame.intent,
            [(c.name, round(float(c.score), 4)) for c in ranked],
            (utterance[:160] + "…") if len(utterance) > 160 else utterance,
        )
        return frame

    def _memory_ingest_pending(self, utterance: str, toks: Sequence[str]) -> CognitiveFrame:
        return CognitiveFrame(
            "memory_ingest_pending",
            answer="pending",
            confidence=0.0,
            evidence={
                "route": "deferred_relation_ingest",
                "deferred_relation_ingest": True,
                "utterance": utterance,
                "source_words": list(LexicalTokens.words(toks)),
                "source": "observed_utterance",
                "instruments": ["runtime_observation", "dmn_relation_ingest"],
            },
        )

    def _memory_write(
        self, mind: "SubstrateController", utterance: str, claim: ParsedClaim
    ) -> CognitiveFrame:
        ev = {
            "source": "observed_utterance",
            "utterance": utterance,
            "predicate": claim.predicate,
            "instruments": ["runtime_observation"],
            **dict(claim.evidence),
        }
        existing = mind.memory.get(claim.subject, claim.predicate)
        if existing is not None and existing[0] != claim.obj:
            gap = self._claim_prediction_gap(mind, utterance, claim)
            if gap is not None:
                ev["prediction_gap"] = float(gap)
        observed = mind.memory.observe_claim(
            claim.subject,
            claim.predicate,
            claim.obj,
            confidence=float(claim.confidence),
            evidence=ev,
        )
        if observed["status"] == "conflict":
            return CognitiveFrame(
                "memory_conflict",
                subject=claim.subject,
                answer=str(observed["current_object"]),
                confidence=float(observed.get("current_confidence", 0.0)),
                evidence={
                    "predicate": claim.predicate,
                    "claimed_answer": claim.obj,
                    "claim_id": observed["claim_id"],
                    "claim_status": observed["status"],
                    "counterfactual": observed["counterfactual"],
                    "source": "observed_utterance",
                    "instruments": [
                        "runtime_observation",
                        "counterfactual_belief_update",
                    ],
                },
            )
        return CognitiveFrame(
            "memory_write",
            subject=claim.subject,
            answer=claim.obj,
            confidence=float(claim.confidence),
            evidence={
                "predicate": claim.predicate,
                "claim_id": observed["claim_id"],
                "claim_status": observed["status"],
                "source": "observed_utterance",
                "instruments": ["runtime_observation"],
            },
        )

    def _claim_prediction_gap(
        self,
        mind: "SubstrateController",
        utterance: str,
        claim: ParsedClaim,
    ) -> float | None:
        try:
            broca_features = mind.frame_packer.broca(
                "memory_write",
                claim.subject,
                claim.obj,
                float(claim.confidence),
                claim.evidence,
                vsa_bundle=mind.encode_triple_vsa(
                    claim.subject, claim.predicate, claim.obj
                ),
                vsa_projection_seed=int(mind.seed),
            )
            _ce_g, _ce_p, gap = lexical_surprise_gap(
                mind.host,
                mind.tokenizer,
                utterance=utterance,
                plan_words=[claim.subject, claim.predicate, claim.obj, "."],
                broca_features=broca_features,
            )
            return float(gap)
        except (
            AttributeError,
            RuntimeError,
            TypeError,
            ValueError,
            StopIteration,
            IndexError,
        ):
            logger.debug(
                "CognitiveRouter._claim_prediction_gap: unavailable host path utterance=%r",
                utterance[:200],
            )
            return None

    def _memory_query(
        self,
        mind: "SubstrateController",
        utterance: str,
        toks: Sequence[str],
        query: ParsedQuery,
    ) -> CognitiveFrame:
        if not query.subject or not str(query.subject).strip():
            return CognitiveFrame(
                "unknown",
                subject="",
                answer="unknown",
                confidence=0.0,
                evidence={
                    "missing": "semantic_subject",
                    "predicate": query.predicate,
                    **dict(query.evidence),
                },
            )
        rec = mind.memory.get(query.subject, query.predicate)
        if rec is None:
            return CognitiveFrame(
                "unknown",
                subject=query.subject,
                answer="unknown",
                confidence=0.0,
                evidence={
                    "missing": "semantic_memory",
                    "predicate": query.predicate,
                    **dict(query.evidence),
                },
            )

        obj, conf, ev = rec
        frame = CognitiveFrame(
            "memory_lookup",
            subject=query.subject,
            answer=obj,
            confidence=conf,
            evidence=dict(ev),
        )
        mu_pop = mind.memory.mean_confidence()
        frame.evidence["semantic_mean_confidence"] = max(
            SEMANTIC_CONFIDENCE_FLOOR, float(mu_pop or 0.0)
        )
        frame.evidence["predicate"] = query.predicate

        known_objects = mind.memory.distinct_objects_for_predicate(query.predicate)
        mentioned_objects = [t for t in toks if t in known_objects]
        conflicting = bool(mentioned_objects and mentioned_objects[-1] != obj.lower())
        if not conflicting:
            return frame

        plan_words = frame.speech_plan()
        broca_features = mind.broca_features_from_frame(frame)
        ce_g, ce_p, gap = lexical_surprise_gap(
            mind.host,
            mind.tokenizer,
            utterance=utterance,
            plan_words=plan_words,
            broca_features=broca_features,
        )
        frame.evidence["prediction_ce_graft"] = ce_g
        frame.evidence["prediction_ce_plain"] = ce_p
        frame.evidence["prediction_gap"] = gap
        if gap <= 0.0:
            return frame

        coupled = mind.unified_agent.decide()
        return CognitiveFrame(
            "prediction_error",
            subject=query.subject,
            answer=obj,
            confidence=conf,
            evidence={
                **dict(frame.evidence),
                "delta_ce": gap,
                "coupled_faculty": coupled.faculty,
                "wake_action": coupled.action_name,
                "spatial_min_G": coupled.spatial_min_G,
                "causal_min_G": coupled.causal_min_G,
            },
        )

    def _active_action(self, mind: "SubstrateController") -> CognitiveFrame:
        coupled = mind.unified_agent.decide()
        posterior_spatial = {
            mind.pomdp.action_names[i]: float(p)
            for i, p in enumerate(
                coupled.spatial_decision.posterior_over_policies[
                    : len(mind.pomdp.action_names)
                ]
            )
        }
        causal_names = mind.causal_pomdp.action_names
        posterior_causal = {
            causal_names[i]: float(p)
            for i, p in enumerate(
                coupled.causal_decision.posterior_over_policies[: len(causal_names)]
            )
        }
        conf = (
            max(coupled.spatial_decision.posterior_over_policies)
            if coupled.faculty == "spatial"
            else max(coupled.causal_decision.posterior_over_policies)
        )
        return CognitiveFrame(
            "active_action",
            answer=coupled.action_name,
            confidence=float(conf),
            evidence={
                "coupled_faculty": coupled.faculty,
                "spatial_min_G": coupled.spatial_min_G,
                "causal_min_G": coupled.causal_min_G,
                "expected_free_energy_spatial": min(
                    ev.expected_free_energy for ev in coupled.spatial_decision.policies
                ),
                "expected_free_energy_causal": min(
                    ev.expected_free_energy for ev in coupled.causal_decision.policies
                ),
                "policy_posterior": posterior_spatial,
                "causal_policy_posterior": posterior_causal,
            },
        )

    def _causal_effect(self, mind: "SubstrateController") -> CognitiveFrame:
        scm = mind.scm
        labels = getattr(scm, "labels", {}) or {}
        t_name, y_name = SCMTargetPicker.pick(scm, labels)
        dom_t = scm.domains.get(t_name, (0, 1))
        dom_y = scm.domains.get(y_name, (0, 1))
        t_lo = 0 if 0 in dom_t else dom_t[0]
        t_hi = 1 if 1 in dom_t else (dom_t[1] if len(dom_t) > 1 else dom_t[0])
        y_hi = 1 if 1 in dom_y else dom_y[-1]
        p1 = scm.probability({y_name: y_hi}, given={}, interventions={t_name: t_hi})
        p0 = scm.probability({y_name: y_hi}, given={}, interventions={t_name: t_lo})
        ate = p1 - p0
        answer_key = "positive_effect" if ate >= 0 else "negative_effect"
        return CognitiveFrame(
            "causal_effect",
            subject=str(labels.get(t_name, t_name)),
            answer=str(labels.get(answer_key, answer_key)),
            confidence=float(min(1.0, abs(ate))),
            evidence={
                "treatment_var": t_name,
                "outcome_var": y_name,
                "p_do_positive": p1,
                "p_do_negative": p0,
                "p_do_high": p1,
                "p_do_low": p0,
                "ate": ate,
                "labels": labels,
            },
        )
