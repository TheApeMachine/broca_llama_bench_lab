"""Cognitive substrate package: constants and the stacked implementation in :mod:`core.cognition.substrate`.

Import the controller with::

    from core.cognition.substrate import SubstrateController
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
