"""EventBus — bounded-ring pub/sub with per-subscriber drop-oldest policy.

Subscribers register on one or more topics (or the wildcard ``"*"``); every
:meth:`publish` enqueues an :class:`Event` into a bounded deque per
subscriber. Consumers pull by calling :meth:`drain` (clears the queue) or
:meth:`peek` (snapshot copy without clearing).

Deques drop oldest events when full, so a slow UI cannot stall any
publishing concern. The default queue size is 1024 — plenty for a 5 Hz UI
tick at debug-log volume.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Iterable

from .event import Event


WILDCARD = "*"
_DEFAULT_QUEUE_SIZE = 1024


class EventBus:
    """Thread-safe topic pub/sub with per-subscriber bounded queues."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next_id = 0
        self._subs: dict[int, tuple[frozenset[str], deque[Event]]] = {}

    def subscribe(
        self,
        topics: Iterable[str] | str = WILDCARD,
        *,
        queue_size: int = _DEFAULT_QUEUE_SIZE,
    ) -> int:
        if isinstance(topics, str):
            topic_set = frozenset({topics})
        else:
            topic_set = frozenset(topics) or frozenset({WILDCARD})
        size = max(16, int(queue_size))
        with self._lock:
            sub_id = self._next_id
            self._next_id += 1
            self._subs[sub_id] = (topic_set, deque(maxlen=size))
        return sub_id

    def unsubscribe(self, sub_id: int) -> None:
        with self._lock:
            self._subs.pop(sub_id, None)

    def publish(self, topic: str, payload: object = None) -> None:
        ev = Event(topic=topic, payload=payload)
        with self._lock:
            for topics, q in self._subs.values():
                if WILDCARD in topics or topic in topics:
                    q.append(ev)

    def drain(self, sub_id: int) -> list[Event]:
        with self._lock:
            entry = self._subs.get(sub_id)
            if entry is None:
                raise KeyError(sub_id)
            _, q = entry
            out = list(q)
            q.clear()
            return out

    def peek(self, sub_id: int) -> list[Event]:
        with self._lock:
            entry = self._subs.get(sub_id)
            if entry is None:
                raise KeyError(sub_id)
            _, q = entry
            return list(q)
