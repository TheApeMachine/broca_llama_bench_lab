"""BaseWorkspace — the pub/sub + blackboard contract every workspace specialist implements.

Concerns outside this package only see :class:`BaseWorkspace`. They publish
events with :meth:`publish`, post cognitive frames to the blackboard with
:meth:`post_frame`, raise interrupts with :meth:`raise_cue`, and consume
events by :meth:`subscribe` + :meth:`drain`.

The workspace makes no guarantee about delivery order across subscribers —
each subscriber sees its own events in publish order, but two subscribers
may interleave differently. Subscribers that fall behind drop oldest events
silently (bounded ring per subscriber); a slow consumer cannot stall the
substrate.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from ..frame import CognitiveFrame
from .event import Event
from .intrinsic_cue import IntrinsicCue


WILDCARD = "*"


class BaseWorkspace(ABC):
    """Abstract pub/sub bus + global blackboard interface."""

    @abstractmethod
    def publish(self, topic: str, payload: object = None) -> None: ...

    @abstractmethod
    def subscribe(self, topics: Iterable[str] | str = WILDCARD, *, queue_size: int = 1024) -> int: ...

    @abstractmethod
    def unsubscribe(self, sub_id: int) -> None: ...

    @abstractmethod
    def drain(self, sub_id: int) -> list[Event]: ...

    @abstractmethod
    def peek(self, sub_id: int) -> list[Event]: ...

    @abstractmethod
    def post_frame(self, frame: CognitiveFrame) -> CognitiveFrame: ...

    @abstractmethod
    def raise_cue(self, cue: IntrinsicCue) -> None: ...

    @property
    @abstractmethod
    def latest_frame(self) -> CognitiveFrame | None: ...

    @property
    @abstractmethod
    def working_frames(self) -> list[CognitiveFrame]: ...

    @property
    @abstractmethod
    def cues(self) -> list[IntrinsicCue]: ...
