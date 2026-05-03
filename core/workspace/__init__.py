"""Workspace — the substrate's pub/sub layer and global blackboard.

Every concern in the system publishes events through one process-wide
:class:`BaseWorkspace`. Subscribers register on topics (or the wildcard
``"*"``); each publish enqueues an event in every matching subscriber's
bounded ring. The :class:`GlobalWorkspace` holds the per-turn cognitive
frames and the :class:`IntrinsicCue` signals the DMN raises when the
substrate notices something worth interrupting itself for.

The workspace owns no persistent state — the journal of past frames lives
in :mod:`core.memory.episodic`, not here. Workspace is the *live* layer;
journal is the *recorded* one.

Public surface: :class:`BaseWorkspace`, :class:`WorkspaceBuilder`,
:class:`IntrinsicCue`. Everything else is internal.
"""

from __future__ import annotations

from .base import BaseWorkspace
from .builder import WorkspaceBuilder
from .global_workspace import GlobalWorkspace
from .intrinsic_cue import IntrinsicCue
from .publisher import WorkspacePublisher

__all__ = [
    "BaseWorkspace",
    "GlobalWorkspace",
    "IntrinsicCue",
    "WorkspaceBuilder",
    "WorkspacePublisher",
]
