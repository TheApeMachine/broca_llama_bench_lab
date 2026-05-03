"""Repository for persisted activation-memory rows."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from core.persistence.sqlite_engine import SqliteEngine

from .model import ActivationAssociation, ActivationMemory


class Repository:
    """Tiny SQLite repository kept for compatibility with older imports."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        with self._connection() as connection:
            SqliteEngine.create_tables(connection, ActivationMemory, ActivationAssociation)

    def insert_activation_memory(self, memory: ActivationMemory) -> None:
        with self._connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO activation_memory(
                    namespace, kind, dim, key_blob, value_blob, metadata_json,
                    confidence, access_count, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory.namespace,
                    memory.kind,
                    int(memory.dim),
                    memory.key_blob,
                    memory.value_blob,
                    memory.metadata_json,
                    float(memory.confidence),
                    int(memory.access_count),
                    float(memory.created_at),
                    float(memory.updated_at),
                ),
            )
            connection.commit()

        if memory.id is None:
            memory.id = int(cursor.lastrowid)

    def insert_activation_association(self, association: ActivationAssociation) -> None:
        lo, hi = sorted((int(association.lo), int(association.hi)))
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO activation_association(lo, hi, weight, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(lo, hi)
                DO UPDATE SET weight = excluded.weight,
                              updated_at = excluded.updated_at
                """,
                (lo, hi, float(association.weight), float(association.updated_at)),
            )
            connection.commit()

    def _connection(self) -> closing[sqlite3.Connection]:
        return closing(
            SqliteEngine.create(
                self.path,
                timeout_seconds=SqliteEngine.CONNECT_TIMEOUT_SECONDS,
            )
        )
