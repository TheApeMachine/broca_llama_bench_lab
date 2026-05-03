"""LogToBusHandler — adapt the standard ``logging`` machinery into a workspace publisher.

Records are published on the topic ``log.<levelname.lower()>`` with a payload
of ``{"name", "level", "msg", "ts"}``. The TUI activity feed subscribes to
``log.*`` and renders the stream without the substrate code knowing the UI
exists.
"""

from __future__ import annotations

import logging

from .event_bus import EventBus


class LogToBusHandler(logging.Handler):
    """``logging.Handler`` that forwards records to an :class:`EventBus`."""

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
            self.handleError(record)
