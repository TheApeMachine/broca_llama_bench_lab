"""A candidate cognitive frame proposed by a single faculty for the same turn.

When more than one faculty (memory lookup, causal effect, active inference,
…) produces a plausible response to the user, each emits a
:class:`FacultyCandidate`. The router picks the highest-scoring candidate
and materializes its frame via the ``build`` thunk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .cognitive_frame import CognitiveFrame


@dataclass
class FacultyCandidate:
    """Lazy-built CognitiveFrame plus router score and per-faculty evidence."""

    name: str
    score: float
    build: Callable[[], CognitiveFrame]
    evidence: dict[str, Any] = field(default_factory=dict)
