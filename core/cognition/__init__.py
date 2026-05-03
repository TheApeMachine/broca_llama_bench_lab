"""Cognition — intent routing, semantics, perception observations, and related constants.

Import :class:`~core.substrate.controller.SubstrateController` from :mod:`core.substrate.controller`
(or the compatibility alias :mod:`core.cognition.substrate`).
"""

from __future__ import annotations

from .constants import (
    BELIEF_REVISION_LOG_ODDS_THRESHOLD,
    BELIEF_REVISION_MIN_CLAIMS,
    DEFAULT_CHAT_MODEL_ID,
    SEMANTIC_CONFIDENCE_FLOOR,
)

__all__ = [
    "BELIEF_REVISION_LOG_ODDS_THRESHOLD",
    "BELIEF_REVISION_MIN_CLAIMS",
    "DEFAULT_CHAT_MODEL_ID",
    "SEMANTIC_CONFIDENCE_FLOOR",
]
