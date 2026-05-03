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
from ..comprehension import DeferredRelationIngest


if TYPE_CHECKING:
    from .substrate import SubstrateController


logger = logging.getLogger(__name__)


class DeferredRelationQueue:
    """Queue + worker for relation-extraction jobs deferred to the DMN."""

    def __init__(self, mind: "SubstrateController") -> None:
        self._mind = mind

    def is_online(self) -> bool:
        worker = self._mind._background_worker
        return worker is not None and worker.running

    def count(self) -> int:
        return len(self._mind._deferred_relation_jobs)

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

        mind = self._mind
        job = DeferredRelationIngest(
            job_id=int(mind._next_deferred_relation_job_id),
            utterance=str(utterance),
            tokens=tuple(str(t) for t in toks),
            intent=intent,
            journal_id=int(journal_id),
            queued_at=time.time(),
        )
        mind._next_deferred_relation_job_id += 1
        mind._deferred_relation_jobs.append(job)

        mind.event_bus.publish(
            "deferred_relation_ingest.queued",
            {
                "job_id": job.job_id,
                "journal_id": job.journal_id,
                "intent_label": intent.label,
                "intent_confidence": float(intent.confidence),
                "pending": len(mind._deferred_relation_jobs),
                "utterance": job.utterance[:200],
            },
        )

        worker = mind._background_worker
        if worker is not None:
            worker.notify_work()
        return job

    def process_all(self) -> list[dict[str, Any]]:
        mind = self._mind
        with mind._cognitive_state_lock:
            reflections: list[dict[str, Any]] = []
            while mind._deferred_relation_jobs:
                job = mind._deferred_relation_jobs.popleft()
                reflections.append(self._process(job))
            return reflections

    def _process(self, job: DeferredRelationIngest) -> dict[str, Any]:
        mind = self._mind
        claim = mind.router.extractor.extract_claim(
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
                "pending": len(mind._deferred_relation_jobs),
            }
            mind.event_bus.publish("deferred_relation_ingest.processed", reflection)
            return reflection

        refined = mind.refine_extracted_claim(job.utterance, job.tokens, claim)
        frame = mind.router._memory_write(mind, job.utterance, refined)
        frame.evidence = {
            **dict(frame.evidence or {}),
            "deferred_relation_job_id": job.job_id,
            "source_journal_id": job.journal_id,
            "queued_at": job.queued_at,
            "processed_at": time.time(),
        }
        mind.workspace.post_frame(frame)
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
            "pending": len(mind._deferred_relation_jobs),
        }
        mind.event_bus.publish("deferred_relation_ingest.processed", reflection)
        return reflection

    def _after_commit(
        self, frame: Any, job: DeferredRelationIngest
    ) -> None:
        mind = self._mind
        try:
            mind.hawkes.observe(str(frame.intent or "unknown"))
        except Exception:
            logger.exception(
                "DeferredRelationQueue._after_commit: hawkes observe failed"
            )
        mind._observe_frame_concepts(frame)
        mind._remember_declarative_binding(frame, job.utterance)
