"""Removed. The substrate's pub/sub layer now lives in :mod:`core.workspace`.

Use:

  from core.workspace import (
      BaseWorkspace,
      GlobalWorkspace,
      IntrinsicCue,
      WorkspaceBuilder,
      WorkspacePublisher,
  )
  from core.workspace.event_bus import EventBus
  from core.workspace.log_handler import LogToBusHandler

Importing this module raises so any caller not yet migrated fails loudly.
"""

from __future__ import annotations

raise ImportError(
    "core.system.event_bus has been removed. Import from core.workspace "
    "(BaseWorkspace, WorkspaceBuilder, WorkspacePublisher) or "
    "core.workspace.event_bus / core.workspace.log_handler instead."
)
