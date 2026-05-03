"""WorkspaceBuilder — single canonical entry point for constructing workspaces.

Every concern that needs to publish or subscribe asks the builder for a
:class:`BaseWorkspace`. Tests build a fresh one per case; production code
uses :meth:`process_default` to share the singleton.
"""

from __future__ import annotations

import threading

from .base import BaseWorkspace
from .composite import CompositeWorkspace
from .event_bus import EventBus
from .global_workspace import GlobalWorkspace


class WorkspaceBuilder:
    """Construct a :class:`BaseWorkspace` (fresh) or fetch the process-wide one."""

    _process_lock = threading.Lock()
    _process_default: BaseWorkspace | None = None

    def build(self) -> BaseWorkspace:
        return CompositeWorkspace(event_bus=EventBus(), blackboard=GlobalWorkspace())

    def process_default(self) -> BaseWorkspace:
        cls = type(self)
        with cls._process_lock:
            if cls._process_default is None:
                cls._process_default = self.build()
            return cls._process_default

    @classmethod
    def reset_process_default(cls) -> None:
        """Test helper: drop the process-wide bus so the next call creates a fresh one."""

        with cls._process_lock:
            cls._process_default = None
