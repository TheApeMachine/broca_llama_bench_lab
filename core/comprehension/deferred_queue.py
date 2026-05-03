"""DeferredRelationQueue — defer relation extraction past the foreground turn.

When :class:`CognitiveRouter` decides a storable utterance should be parsed
later (foreground has higher-priority work), it enqueues a
:class:`DeferredRelationIngest`. The DMN drains the queue between turns by
calling :meth:`DeferredRelationQueue.process_all`.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Sequence

from ..cognition.intent_gate import UtteranceIntent
from .deferred_relation_ingest import DeferredRelationIngest

if TYPE_CHECKING:
    from .pipeline import ComprehensionPipeline

from .claim_refiner import ClaimRefiner


logger = logging.getLogger(__name__)


class DeferredRelationQueue:
    """Queue + worker for relation-extraction jobs deferred to the DMN."""

    def __init__(
        self,
        *,
        router: Any,
        event_bus: Any,
        hawkes: Any,
        claims: ClaimRefiner,
        substrate: Any,
        session: Any,
    ) -> None:
        self._router = router
        self._event_bus = event_bus
        self._hawkes = hawkes
        self._claims = claims
        self._substrate = substrate
        self._session = session
        self._comprehension: ComprehensionPipeline | None = None

    def bind_comprehension(self, pipe: ComprehensionPipeline) -> None:
        self._comprehension = pipe

    def is_online(self) -> bool:
        worker = self._session.background_worker
        return worker is not None and worker.running

    def count(self) -> int:
        return len(self._session.deferred_relation_jobs)

    def enqueue(
        self,
        utterance: str,
        toks: Sequence[str],
        intent: UtteranceIntent,
        *,
        journal_id: int,
    ) -> DeferredRelationIngest:
        if not intent.allows_storage:
            raise ValueError(f"cannot defer non-storable intent: {intent.label}")

        sess = self._session
        job = DeferredRelationIngest(
            job_id=int(sess.next_deferred_relation_job_id),
            utterance=str(utterance),
            tokens=tuple(str(t) for t in toks),
            intent=intent,
            journal_id=int(journal_id),
            queued_at=time.time(),
        )
        sess.next_deferred_relation_job_id += 1
        sess.deferred_relation_jobs.append(job)

        self._event_bus.publish(
            "deferred_relation_ingest.queued",
            {
                "job_id": job.job_id,
                "journal_id": job.journal_id,
                "intent_label": intent.label,
                "intent_confidence": float(intent.confidence),
                "pending": len(sess.deferred_relation_jobs),
                "utterance": job.utterance[:200],
            },
        )

        worker = sess.background_worker
        if worker is not None:
            worker.notify_work()
        return job

    def process_all(self) -> list[dict[str, Any]]:
        sess = self._session
        with sess.cognitive_state_lock:
            reflections: list[dict[str, Any]] = []
            while sess.deferred_relation_jobs:
                job = sess.deferred_relation_jobs.popleft()
                reflections.append(self._process(job))
            return reflections

    def _process(self, job: DeferredRelationIngest) -> dict[str, Any]:
        sess = self._session
        claim = self._router.extractor.extract_claim(
            job.utterance, job.tokens, utterance_intent=job.intent
        )
        if claim is None:
            reflection = {
                "kind": "deferred_relation_ingest",
                "status": "no_relation",
                "job_id": job.job_id,
                "journal_id": job.journal_id,
                "utterance": job.utterance[:200],
                "intent_label": job.intent.label,
                "pending": len(sess.deferred_relation_jobs),
            }
            self._event_bus.publish("deferred_relation_ingest.processed", reflection)
            return reflection

        refined = self._claims.refine(job.utterance, job.tokens, claim)
        frame = self._router._memory_write(self._substrate, job.utterance, refined)
        frame.evidence = {
            **dict(frame.evidence or {}),
            "deferred_relation_job_id": job.job_id,
            "source_journal_id": job.journal_id,
            "queued_at": job.queued_at,
            "processed_at": time.time(),
        }
        self._substrate.workspace.post_frame(frame)
        self._after_commit(frame, job)

        reflection = {
            "kind": "deferred_relation_ingest",
            "status": frame.intent,
            "job_id": job.job_id,
            "journal_id": job.journal_id,
            "subject": frame.subject,
            "answer": frame.answer,
            "confidence": float(frame.confidence),
            "evidence": dict(frame.evidence),
            "pending": len(sess.deferred_relation_jobs),
        }
        self._event_bus.publish("deferred_relation_ingest.processed", reflection)
        return reflection

    def _after_commit(self, frame: Any, job: DeferredRelationIngest) -> None:
        try:
            self._hawkes.observe(str(frame.intent or "unknown"))
        except Exception:
            logger.exception(
                "DeferredRelationQueue._after_commit: hawkes observe failed"
            )
        pipe = self._comprehension
        if pipe is None:
            raise RuntimeError(
                "DeferredRelationQueue: comprehension pipeline must be bound before processing jobs"
            )
        pipe.observe_frame_concepts(frame)
        pipe.remember_declarative_binding(frame, job.utterance)
