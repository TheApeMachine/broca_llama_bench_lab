"""A parsed declarative claim: subject/predicate/object plus provenance.

Produced by :class:`core.comprehension.ClaimExtractor` from a user utterance
or an ingested page; consumed by :class:`core.memory.SymbolicMemory.store`
and by :class:`core.consolidation.BeliefRevisionConsolidation`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParsedClaim:
    """Immutable record of one extracted (subject, predicate, object) triple.

    ``confidence`` is in ``[0, 1]`` and reflects extractor confidence at the
    point of parsing — not consolidated belief, which is computed later by
    :class:`core.consolidation.BeliefRevisionConsolidation`.

    ``evidence`` is an open-ended dict for the parser's notes (which model
    extracted it, the surface form of the predicate, the source utterance,
    ensemble disagreement notes, etc.).
    """

    subject: str
    predicate: str
    obj: str
    confidence: float
    evidence: dict[str, Any]
