"""Event — the immutable record of a single workspace publish call."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Event:
    """One pub/sub message: a topic name, an opaque payload, and a timestamp."""

    topic: str
    payload: Any
    ts: float = field(default_factory=time.time)
