"""WorkspacePublisher — single canonical wrapper that publishes to the process-wide workspace.

Replaces the nine duplicated module-level ``_publish`` helpers that used to
live in :mod:`core.grafts.strength`, ``encoder_relation_extractor``,
``intent_gate``, ``predictive_coding``, ``top_down_control``, and four
``core.encoders`` modules.  Every concern that needs to emit a workspace
event imports this class instead of writing its own.
"""

from __future__ import annotations

from .builder import WorkspaceBuilder


class WorkspacePublisher:
    """Stateless convenience wrapper around the process-default workspace."""

    @classmethod
    def emit(cls, topic: str, payload: dict) -> None:
        """Publish ``payload`` on ``topic`` to the process-default workspace.

        Lookups go through :meth:`WorkspaceBuilder.process_default` on every
        call so tests can call :meth:`WorkspaceBuilder.reset_process_default`
        between cases without leaving stale references behind.
        """

        WorkspaceBuilder().process_default().publish(topic, payload)
