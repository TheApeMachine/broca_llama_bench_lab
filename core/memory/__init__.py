"""Memory — symbolic facts, claim consolidation, activation, Hopfield retrieval.

The substrate's memory layer holds:

* :class:`SymbolicMemory` — SQLite-backed semantic-fact store, claims log,
  reflections (was ``SymbolicMemory`` in the substrate monolith).
* :class:`ClaimTrust` — prediction-error-weighted scoring used by
  :meth:`SymbolicMemory.consolidate_claims_once` and the DMN.
* :class:`SQLiteActivationMemory` — key/value blob store for graft
  activations.
* :class:`HopfieldAssociativeMemory` — Modern Continuous Hopfield retrieval
  for substrate-side associative memory.

Public surface: the classes named in :data:`__all__`. Internals (schema
table names, JSON column shapes, lock implementation) are implementation
details and may change without bumping the public contract.
"""

from __future__ import annotations

from .claim_trust import ClaimTrust
from .episodic import WorkspaceJournal
from .hopfield import HopfieldAssociativeMemory, derived_inverse_temperature, hopfield_update
from .memory import ActivationMemoryGraftProtocol, MemoryRecord, SQLiteActivationMemory
from .symbolic import (
    BELIEF_REVISION_LOG_ODDS_THRESHOLD,
    BELIEF_REVISION_MIN_CLAIMS,
    SymbolicMemory,
)

__all__ = [
    "ActivationMemoryGraftProtocol",
    "BELIEF_REVISION_LOG_ODDS_THRESHOLD",
    "BELIEF_REVISION_MIN_CLAIMS",
    "ClaimTrust",
    "HopfieldAssociativeMemory",
    "MemoryRecord",
    "SQLiteActivationMemory",
    "SymbolicMemory",
    "WorkspaceJournal",
    "derived_inverse_temperature",
    "hopfield_update",
]
