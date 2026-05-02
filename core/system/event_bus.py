"""Thread-safe pub/sub event bus for live UI / debugger feeds.

The bus is intentionally minimal: subscribers register on one or more topics
(or the wildcard ``"*"``), every ``publish`` enqueues an :class:`Event` into a
bounded deque per subscriber, and consumers pull by calling :meth:`drain`.
Deques drop oldest events when full so a slow UI cannot stall the substrate.

Used by the Textual TUI to render activity feeds, sparklines, and status
changes without polluting the substrate code with UI concerns. The companion
:class:`LogToBusHandler` adapts the standard ``logging`` machinery into a bus
publisher, so existing ``logger.info(...)`` calls light up the UI for free.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable

WILDCARD = "*"

# A bounded ring per subscriber means a stalled UI drops oldest events instead
# of growing memory unbounded. 1024 is plenty for a 5Hz UI tick at debug level.
_DEFAULT_QUEUE_SIZE = 1024


@dataclass(frozen=True)
class Event:
    topic: str
    payload: Any
    ts: float = field(default_factory=time.time)


class EventBus:
    """Thread-safe topic pub/sub with per-subscriber bounded queues."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next_id = 0
        # sub_id -> (topics_set, deque)
        self._subs: dict[int, tuple[frozenset[str], deque[Event]]] = {}

    def subscribe(self, topics: Iterable[str] | str = WILDCARD, *, queue_size: int = _DEFAULT_QUEUE_SIZE) -> int:
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

    def publish(self, topic: str, payload: Any = None) -> None:
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
        """Return a snapshot copy without clearing."""

        with self._lock:
            entry = self._subs.get(sub_id)
            if entry is None:
                raise KeyError(sub_id)
            _, q = entry
            return list(q)


class LogToBusHandler(logging.Handler):
    """``logging.Handler`` that forwards records to an :class:`EventBus`.

    Records are published on the topic ``log.<levelname.lower()>`` with a
    payload of ``{"name", "level", "msg", "ts"}``. The TUI activity feed
    subscribes to ``log.*`` (or specific levels) and renders the stream
    without changes to any subsystem.
    """

    def __init__(self, bus: EventBus, *, level: int = logging.INFO) -> None:
        super().__init__(level=level)
        self.bus = bus

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
        except Exception:
            msg = record.msg if isinstance(record.msg, str) else repr(record.msg)
        payload = {
            "name": record.name,
            "level": record.levelname,
            "msg": msg,
            "ts": record.created,
        }
        topic = f"log.{record.levelname.lower()}"
        try:
            self.bus.publish(topic, payload)
        except Exception:
            # A logging handler must never raise into the calling code.
            self.handleError(record)


# Process-wide default bus. Subsystems can publish into the default bus
# without taking a reference; the TUI grabs the same instance via
# ``get_default_bus()`` when wiring subscribers.
_DEFAULT_BUS: EventBus | None = None
_DEFAULT_BUS_LOCK = threading.Lock()


def get_default_bus() -> EventBus:
    global _DEFAULT_BUS
    with _DEFAULT_BUS_LOCK:
        if _DEFAULT_BUS is None:
            _DEFAULT_BUS = EventBus()
        return _DEFAULT_BUS


def _reset_default_bus() -> None:
    """Test helper: drop the process-wide bus so the next call creates a fresh one."""

    global _DEFAULT_BUS
    with _DEFAULT_BUS_LOCK:
        _DEFAULT_BUS = None
