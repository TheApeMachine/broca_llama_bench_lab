"""A parsed memory-lookup query: subject/predicate plus provenance.

Produced by :class:`core.comprehension.QueryResolver` when a user utterance
is interpreted as a question whose answer should come from semantic memory;
consumed by :class:`core.memory.SymbolicMemory.lookup`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParsedQuery:
    """Immutable record of one (subject, predicate) lookup intent."""

    subject: str
    predicate: str
    confidence: float
    evidence: dict[str, Any]
