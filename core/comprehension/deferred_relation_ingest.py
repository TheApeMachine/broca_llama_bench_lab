"""DeferredRelationIngest — record of a relation-extraction job queued for the DMN.

When :class:`CognitiveRouter` decides a storable utterance should be parsed
later (because the foreground has higher-priority work), it enqueues one of
these. The DMN's relation-ingest worker drains the queue between turns.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..cognition.intent_gate import UtteranceIntent


@dataclass(frozen=True)
class DeferredRelationIngest:
    job_id: int
    utterance: str
    tokens: tuple[str, ...]
    intent: UtteranceIntent
    journal_id: int
    queued_at: float
