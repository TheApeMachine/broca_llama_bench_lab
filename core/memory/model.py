from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass(slots=True)
class ActivationMemory:
    """Plain row object for persisted activation-memory entries."""

    namespace: str
    kind: str
    dim: int
    key_blob: bytes
    value_blob: bytes
    metadata_json: str
    confidence: float
    created_at: float
    updated_at: float
    id: int | None = None
    access_count: int = 0

    table_name: ClassVar[str] = "activation_memory"
    create_statement: ClassVar[str] = """
        CREATE TABLE IF NOT EXISTS activation_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            namespace TEXT NOT NULL,
            kind TEXT NOT NULL,
            dim INTEGER NOT NULL,
            key_blob BLOB NOT NULL,
            value_blob BLOB NOT NULL,
            metadata_json TEXT NOT NULL,
            confidence REAL NOT NULL,
            access_count INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        )
    """
    index_statements: ClassVar[tuple[str, ...]] = (
        "CREATE INDEX IF NOT EXISTS idx_activation_namespace_kind ON activation_memory(namespace, kind)",
    )


@dataclass(slots=True)
class ActivationAssociation:
    """Plain row object for symmetric activation co-occurrence weights."""

    lo: int
    hi: int
    weight: float
    updated_at: float

    table_name: ClassVar[str] = "activation_association"
    create_statement: ClassVar[str] = """
        CREATE TABLE IF NOT EXISTS activation_association (
            lo INTEGER NOT NULL,
            hi INTEGER NOT NULL,
            weight REAL NOT NULL,
            updated_at REAL NOT NULL,
            PRIMARY KEY (lo, hi)
        )
    """
    index_statements: ClassVar[tuple[str, ...]] = (
        "CREATE INDEX IF NOT EXISTS idx_activation_assoc_lo ON activation_association(lo)",
        "CREATE INDEX IF NOT EXISTS idx_activation_assoc_hi ON activation_association(hi)",
    )
