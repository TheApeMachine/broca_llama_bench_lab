"""CompositeWorkspace — the only :class:`BaseWorkspace` implementation.

Composes :class:`EventBus` for pub/sub and :class:`GlobalWorkspace` for the
cognitive frame buffer. There is one composite per substrate session;
``WorkspaceBuilder.build()`` returns it.
"""

from __future__ import annotations

from typing import Iterable

from ..frame import CognitiveFrame
from .base import BaseWorkspace, WILDCARD
from .event import Event
from .event_bus import EventBus
from .global_workspace import GlobalWorkspace
from .intrinsic_cue import IntrinsicCue


class CompositeWorkspace(BaseWorkspace):
    """The substrate's pub/sub bus + global blackboard, fused."""

    def __init__(self, *, event_bus: EventBus, blackboard: GlobalWorkspace) -> None:
        self._bus = event_bus
        self._blackboard = blackboard

    @property
    def event_bus(self) -> EventBus:
        return self._bus

    def publish(self, topic: str, payload: object = None) -> None:
        self._bus.publish(topic, payload)

    def subscribe(self, topics: Iterable[str] | str = WILDCARD, *, queue_size: int = 1024) -> int:
        return self._bus.subscribe(topics, queue_size=queue_size)

    def unsubscribe(self, sub_id: int) -> None:
        self._bus.unsubscribe(sub_id)

    def drain(self, sub_id: int) -> list[Event]:
        return self._bus.drain(sub_id)

    def peek(self, sub_id: int) -> list[Event]:
        return self._bus.peek(sub_id)

    def post_frame(self, frame: CognitiveFrame) -> CognitiveFrame:
        return self._blackboard.post_frame(frame)

    def raise_cue(self, cue: IntrinsicCue) -> None:
        self._blackboard.raise_cue(cue)

    @property
    def latest_frame(self) -> CognitiveFrame | None:
        return self._blackboard.latest

    @property
    def working_frames(self) -> list[CognitiveFrame]:
        return list(self._blackboard.working)

    @property
    def cues(self) -> list[IntrinsicCue]:
        return list(self._blackboard.intrinsic_cues)
