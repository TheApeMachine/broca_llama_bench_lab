"""Mutable holder for :class:`~core.agent.active_inference.ToolForagingAgent`."""


from __future__ import annotations

from typing import Any


class ToolForagingSlot:
    """Native-tool synthesis reads/writes this slot instead of ad hoc attrs."""

    __slots__ = ("agent",)

    def __init__(self, agent: Any) -> None:
        self.agent = agent
