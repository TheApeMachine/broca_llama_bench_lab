"""ComprehensionPipeline — substrate-side end-to-end utterance comprehension.

The substrate controller used to inline the entire utterance → frame
pipeline plus its post-commit side effects. That cluster (~300 lines, 13
methods) lives here. The controller's :meth:`comprehend` becomes a
two-line delegation, and the remaining perceive_* / commit / scan methods
follow the same shape.
"""

from __future__ import annotations

import logging
import math
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Sequence

from ..agent.active_inference import entropy as belief_entropy
from ..affect.evidence import AffectEvidence
from ..cognition.constants import SEMANTIC_CONFIDENCE_FLOOR
from ..cognition.intent_gate import UtteranceIntent
from ..cognition.observation import CognitiveObservation
from ..encoders.affect import AffectState
from ..frame import CognitiveFrame, SubwordProjector
from ..host.tokenizer import utterance_words
from ..swm import SWMSource
from ..workspace import IntrinsicCue


if TYPE_CHECKING:
    from ..substrate.controller import SubstrateController


logger = logging.getLogger(__name__)
_SUBWORD = SubwordProjector()


class ComprehensionPipeline:
    """Substrate-side façade over the comprehend / perceive_* / commit_frame surface."""

    def __init__(self, mind: "SubstrateController") -> None:
        self._mind = mind

    # -- foreground ------------------------------------------------------------

    def comprehend(self, utterance: str) -> CognitiveFrame:
        mind = self._mind
        toks = utterance_words(utterance)
        intent, affect = self.perceive_utterance(utterance)
        with mind.session.cognitive_state_lock:
            self.intrinsic_scan(toks)
            mind.session.last_intent = intent
            mind.session.last_affect = affect
            if not intent.is_actionable:
                frame = self.non_actionable_frame(intent, affect)
            else:
                frame = mind.router.route(mind, utterance, toks, utterance_intent=intent)
                self.attach_perception(frame, intent, affect)
            out = self.commit_frame(utterance, toks, frame)
            self.publish_to_swm(intent=intent, frame=out)
            if bool((out.evidence or {}).get("deferred_relation_ingest")):
                journal_id = (out.evidence or {}).get("journal_id")
                if journal_id is None:
                    raise RuntimeError(
                        "deferred relation ingest frame is missing journal_id"
                    )
                mind.deferred_relations.enqueue(
                    utterance,
                    toks,
                    intent,
                    journal_id=int(journal_id),
                )
            mind.session.last_user_affect_trace_id = mind.affect_trace.record(
                role="user",
                text=utterance,
                affect=affect,
                journal_id=(out.evidence or {}).get("journal_id"),
            )
        self.after_frame_commit(out, utterance, event_topic="frame.comprehend")
        return out

    def perceive_utterance(
        self, utterance: str
    ) -> tuple[UtteranceIntent, AffectState]:
        mind = self._mind
        with ThreadPoolExecutor(max_workers=2) as executor:
            intent_future = executor.submit(mind.intent_gate.classify, utterance)
            affect_future = executor.submit(mind.affect_encoder.detect, utterance)
            return intent_future.result(), affect_future.result()

    # -- frame committing ------------------------------------------------------

    def commit_frame(
        self, utterance: str, toks: Sequence[str], frame: CognitiveFrame
    ) -> CognitiveFrame:
        import time

        mind = self._mind
        commit_ts = time.time()
        trace = mind.hawkes.trace(t=commit_ts)
        frame.evidence = {**dict(frame.evidence or {}), "hawkes_trace": trace}
        jid = mind.journal.append(utterance, frame, ts=commit_ts)
        frame.evidence = {**frame.evidence, "journal_id": jid}
        if mind.session.last_journal_id is not None:
            mind.episode_graph.bump(mind.session.last_journal_id, jid)
        mind.session.last_journal_id = jid
        out = mind.workspace.post_frame(frame)
        predicate = str((out.evidence or {}).get("predicate", ""))
        if out.intent == "memory_write" and out.subject and predicate:
            mind.memory.merge_epistemic_evidence(
                out.subject, predicate, out.evidence
            )
        for tail in mind.workspace.frames:
            pred = str((tail.evidence or {}).get("predicate", ""))
            if tail.intent == "synthesis_bundle" and tail.subject and pred:
                mind.memory.merge_epistemic_evidence(
                    tail.subject, pred, tail.evidence
                )
        return out

    def after_frame_commit(
        self,
        out: CognitiveFrame,
        utterance: str,
        *,
        event_topic: str,
    ) -> None:
        mind = self._mind
        try:
            mind.hawkes.observe(str(out.intent or "unknown"))
        except Exception:
            logger.exception("ComprehensionPipeline.after_frame_commit: hawkes observe failed")

        if mind.session.background_worker is not None:
            mind.session.background_worker.mark_user_active()

        self.observe_frame_concepts(out)
        self.remember_declarative_binding(out, utterance)

        try:
            payload = {
                "intent": out.intent,
                "subject": out.subject,
                "answer": out.answer,
                "confidence": float(out.confidence),
                "journal_id": (out.evidence or {}).get("journal_id"),
                "utterance": utterance[:200],
            }
            if event_topic == "frame.perception":
                payload.update(
                    {
                        "modality": (out.evidence or {}).get("modality"),
                        "source": (out.evidence or {}).get("source"),
                        "feature_dim": (out.evidence or {}).get("feature_dim"),
                    }
                )
            mind.event_bus.publish(event_topic, payload)
        except Exception:
            logger.exception(
                "ComprehensionPipeline.after_frame_commit: event publish failed"
            )

    def observe_frame_concepts(self, out: CognitiveFrame) -> None:
        mind = self._mind
        for concept in (out.subject, out.answer):
            if isinstance(concept, str) and concept and concept != "unknown":
                mind.ontology.observe(concept)
                base = _SUBWORD.encode(concept)
                mind.ontology.maybe_promote(concept, base)

    def remember_declarative_binding(
        self, out: CognitiveFrame, utterance: str
    ) -> None:
        mind = self._mind
        if out.subject and out.answer and out.intent in {"memory_write", "memory_lookup"}:
            try:
                pr_bind = str((out.evidence or {}).get("predicate", out.intent))
                mind.vsa.encode_triple(out.subject, pr_bind, out.answer)
                ut_sk = _SUBWORD.encode(utterance[:512])
                trip_sk = _SUBWORD.encode(f"{out.subject}|{pr_bind}|{out.answer}")
                mind.remember_hopfield(
                    ut_sk,
                    trip_sk,
                    metadata={"kind": "declarative_binding", "intent": out.intent},
                )
            except Exception:
                logger.exception(
                    "ComprehensionPipeline.remember_declarative_binding: VSA/Hopfield binding failed"
                )

    # -- multimodal observation ------------------------------------------------

    def commit_observation(
        self, observation: CognitiveObservation
    ) -> CognitiveFrame:
        mind = self._mind
        source_text = f"[{observation.modality}:{observation.source}] {observation.answer}"
        frame = self.frame_from_observation(observation)
        with mind.session.cognitive_state_lock:
            out = self.commit_frame(source_text, utterance_words(source_text), frame)
            mind.vsa.encode_triple(
                observation.modality, "observed_as", observation.answer
            )
            mind.remember_hopfield(
                _SUBWORD.encode(source_text[:512]),
                observation.features,
                metadata={
                    "kind": "multimodal_observation",
                    "modality": observation.modality,
                    "source": observation.source,
                    "intent": out.intent,
                    "journal_id": (out.evidence or {}).get("journal_id"),
                },
            )
        self.after_frame_commit(out, source_text, event_topic="frame.perception")
        return out

    @staticmethod
    def frame_from_observation(observation: CognitiveObservation) -> CognitiveFrame:
        return CognitiveFrame(
            f"perception_{observation.modality}",
            subject=observation.subject,
            answer=observation.answer,
            confidence=float(observation.confidence),
            evidence={
                **observation.frame_evidence(),
                "is_actionable": True,
                "allows_storage": False,
                "intent_label": f"perception_{observation.modality}",
                "intent_confidence": float(observation.confidence),
            },
        )

    def perceive_image(self, image: Any, *, source: str = "image") -> CognitiveFrame:
        return self.commit_observation(
            self._mind.multimodal_perception.perceive_image(image, source=source)
        )

    def perceive_video(self, frames: Any, *, source: str = "video") -> CognitiveFrame:
        return self.commit_observation(
            self._mind.multimodal_perception.perceive_video(frames, source=source)
        )

    def perceive_audio(
        self,
        audio: Any,
        *,
        sampling_rate: int = 16000,
        source: str = "audio",
        language: str | None = None,
    ) -> CognitiveFrame:
        mind = self._mind
        observation = mind.multimodal_perception.perceive_audio(
            audio,
            sampling_rate=int(sampling_rate),
            source=source,
            language=language,
        )
        out = self.commit_observation(observation)
        transcription = str((observation.evidence or {}).get("transcription") or "").strip()
        if transcription:
            transcription_frame = self.comprehend(transcription)
            try:
                mind.event_bus.publish(
                    "frame.perception.transcription",
                    {
                        "audio_journal_id": (out.evidence or {}).get("journal_id"),
                        "transcription_journal_id": (
                            transcription_frame.evidence or {}
                        ).get("journal_id"),
                        "transcription": transcription[:200],
                    },
                )
            except Exception:
                logger.exception(
                    "ComprehensionPipeline.perceive_audio: transcription event publish failed"
                )
        return out

    # -- routing helpers -------------------------------------------------------

    def intrinsic_scan(self, toks: list[str]) -> None:
        mind = self._mind
        mind.workspace.intrinsic_cues.clear()
        mu_pop = mind.memory.mean_confidence()
        confidence_floor = (
            SEMANTIC_CONFIDENCE_FLOOR
            if mu_pop is None
            else max(SEMANTIC_CONFIDENCE_FLOOR, float(mu_pop))
        )
        toks_set = set(toks)
        for ent in mind.memory.subjects():
            if ent not in toks_set:
                continue
            records = mind.memory.records_for_subject(ent)
            if not records:
                mind.workspace.intrinsic_cues.append(
                    IntrinsicCue(1.0, "memory_gap", {"subject": ent})
                )
                continue
            best_pred, _obj, best_conf, _ev = max(records, key=lambda row: row[2])
            if best_conf < confidence_floor:
                mind.workspace.intrinsic_cues.append(
                    IntrinsicCue(
                        float(confidence_floor - best_conf),
                        "memory_low_confidence",
                        {"subject": ent, "predicate": best_pred, "confidence": best_conf},
                    )
                )
        cq = mind.causal_agent.qs
        if cq is not None and len(cq) >= 2:
            max_ent = math.log(len(cq))
            h_q = belief_entropy(cq)
            if max_ent > 1e-9 and h_q > 0.5 * max_ent:
                mind.workspace.intrinsic_cues.append(
                    IntrinsicCue(
                        float(h_q / max_ent), "causal_uncertain", {"entropy": h_q}
                    )
                )
        try:
            for cue in mind.workspace.intrinsic_cues:
                mind.event_bus.publish(
                    "intrinsic_cue",
                    {
                        "urgency": float(cue.urgency),
                        "faculty": cue.faculty,
                        "evidence": dict(cue.evidence) if isinstance(cue.evidence, dict) else {},
                    },
                )
        except Exception:
            logger.exception("ComprehensionPipeline.intrinsic_scan: event publish failed")

    @staticmethod
    def non_actionable_frame(
        intent: UtteranceIntent, affect: AffectState
    ) -> CognitiveFrame:
        evidence = {
            "route": "intent_gate",
            "intent_label": intent.label,
            "intent_confidence": float(intent.confidence),
            "intent_scores": dict(intent.scores),
            "is_actionable": False,
            "allows_storage": intent.allows_storage,
            "affect": AffectEvidence.as_dict(affect),
        }
        return CognitiveFrame(
            "unknown",
            answer="unknown",
            confidence=0.0,
            evidence=evidence,
        )

    # -- SWM publication ------------------------------------------------------

    def publish_to_swm(self, *, intent: UtteranceIntent, frame: CognitiveFrame) -> None:
        """Publish encoder outputs (structured + hidden) into the substrate working memory.

        Hidden-state slots come from the captured ``last_hidden`` on each
        encoder (only published if the encoder has been called and exposes a
        captured tensor). Structured slots come from the comprehension frame
        and the intent classification.
        """

        mind = self._mind
        publisher = mind.swm_publisher

        self._publish_encoder_hidden(publisher, mind.extraction_encoder, SWMSource.GLINER2)
        self._publish_encoder_hidden(publisher, mind.classification_encoder, SWMSource.GLICLASS)

        publisher.publish_classifications(
            source=SWMSource.GLICLASS,
            labels=[intent.label] if intent.label else [],
        )

        evidence = dict(frame.evidence or {})
        alternatives = evidence.get("alternative_relations") or []
        triples = [
            (rel.get("subject", ""), rel.get("predicate", ""), rel.get("object", ""))
            for rel in alternatives
            if isinstance(rel, dict)
        ]

        primary_predicate = str(evidence.get("predicate", "")).strip()
        primary_subject = str(getattr(frame, "subject", "")).strip()
        primary_answer = str(getattr(frame, "answer", "")).strip()

        if primary_subject and primary_predicate and primary_answer:
            triples.append((primary_subject, primary_predicate, primary_answer))

        publisher.publish_relations(source=SWMSource.GLINER2, triples=triples)

        if primary_subject:
            publisher.publish_entities(
                source=SWMSource.GLINER2,
                entities=[("subject", primary_subject)],
            )

    @staticmethod
    def _publish_encoder_hidden(publisher: Any, encoder: Any, source: SWMSource) -> None:
        if encoder is None or not getattr(encoder, "has_captured_hidden", False):
            return

        # Encoders with no native confidence reporting are treated as unit
        # confidence; the recorded prediction error therefore collapses to 0
        # and the joint EFE leaves them out of the active organ-error budget.
        publisher.publish_hidden(source=source, hidden=encoder.last_hidden, confidence=1.0)

    @staticmethod
    def attach_perception(
        frame: CognitiveFrame, intent: UtteranceIntent, affect: AffectState
    ) -> None:
        frame.evidence = {
            **dict(frame.evidence or {}),
            "intent_label": intent.label,
            "intent_confidence": float(intent.confidence),
            "intent_scores": dict(intent.scores),
            "is_actionable": True,
            "allows_storage": intent.allows_storage,
            "affect": AffectEvidence.as_dict(affect),
        }
