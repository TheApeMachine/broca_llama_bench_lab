"""Mutable coordination shared across comprehension, chat, DMN, and deferred jobs."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SubstrateSessionState:
    cognitive_state_lock: threading.RLock = field(default_factory=threading.RLock)
    deferred_relation_jobs: deque[Any] = field(default_factory=deque)
    next_deferred_relation_job_id: int = 1
    last_chat_meta: dict[str, Any] = field(default_factory=dict)
    last_intent: Any = None
    last_affect: Any = None
    last_user_affect_trace_id: Any = None
    last_journal_id: Any = None
    background_worker: Any = None
    self_improve_worker: Any = None
