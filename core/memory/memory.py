from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, Sequence

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ActivationMemoryGraftProtocol(Protocol):
    """Required surface for loading activation memory into a graft."""

    def remember(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        metadata: Optional[dict] = None,
    ) -> None: ...


@dataclass
class MemoryRecord:
    id: int
    namespace: str
    kind: str
    key: torch.Tensor
    value: torch.Tensor
    metadata: dict
    confidence: float
    access_count: int


class TensorBlobCodec:
    """Encode tensors into deterministic little-endian float32 SQLite blobs."""

    def to_blob(self, tensor: torch.Tensor) -> tuple[bytes, int]:
        array = np.ascontiguousarray(tensor.detach().cpu().numpy().astype("<f4", copy=False))

        return array.tobytes(), int(array.size)

    def to_tensor(self, blob: bytes, dim: int) -> torch.Tensor:
        raw = np.frombuffer(blob, dtype=np.dtype("<f4"))
        expected = int(dim)

        if raw.size != expected:
            raise ValueError(f"blob size {raw.size} != declared dim {expected}")

        return torch.from_numpy(np.array(raw)).float()


class SQLiteActivationSchema:
    """Own the activation-memory schema and SQL statements."""

    def initialize(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            """
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
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS activation_association (
                lo INTEGER NOT NULL,
                hi INTEGER NOT NULL,
                weight REAL NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (lo, hi)
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_activation_namespace_kind ON activation_memory(namespace, kind)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_activation_assoc_lo ON activation_association(lo)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_activation_assoc_hi ON activation_association(hi)"
        )


class SQLiteActivationConnection:
    """Connection factory with WAL and busy-timeout configuration."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def open(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=30.0, check_same_thread=False)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=60000")

        return connection


class SQLiteActivationMemory:
    """Persistent activation-space memory backed by SQLite.

    Records are hidden-state keys and hidden-state value directions.  Loading
    this store into a graft changes activations directly; it does not paste
    facts into prompts.
    """

    def __init__(self, path: str | Path, *, default_namespace: str = "main") -> None:
        self.path = Path(path)
        self.default_namespace = default_namespace
        self.codec = TensorBlobCodec()
        self.schema = SQLiteActivationSchema()
        self.connection = SQLiteActivationConnection(self.path)
        self._lock = threading.RLock()
        self._init_schema()

    def bump_association(self, id_a: int, id_b: int, *, delta: float = 1.0) -> None:
        """Symmetric co-activation counter between activation-memory rows."""

        left, right = int(id_a), int(id_b)

        if left == right:
            return

        lo, hi = (left, right) if left < right else (right, left)
        now = time.time()

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO activation_association(lo, hi, weight, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(lo, hi)
                DO UPDATE SET weight = weight + excluded.weight,
                              updated_at = excluded.updated_at
                """,
                (lo, hi, float(delta), now),
            )
            row = connection.execute(
                "SELECT weight FROM activation_association WHERE lo=? AND hi=?",
                (lo, hi),
            ).fetchone()

        if row is None:
            raise RuntimeError(f"association row ({lo},{hi}) missing after upsert")

        logger.debug(
            "SQLiteActivationMemory.bump_association: pair=(%s,%s) weight=%s",
            lo,
            hi,
            float(row[0]),
        )

    def normalized_spread_matrix(self, record_ids: list[int]) -> torch.Tensor:
        """Row-stochastic spread operator over ordered graft slots."""

        slot_count = len(record_ids)

        if slot_count == 0:
            return torch.empty(0, 0)

        index_of = {int(record_id): index for index, record_id in enumerate(record_ids)}
        accum = torch.eye(slot_count, dtype=torch.float32)

        if slot_count < 2:
            return accum

        ids = tuple(sorted(index_of))
        placeholders = ",".join("?" for _ in ids)

        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT lo, hi, weight
                FROM activation_association
                WHERE lo IN ({placeholders}) AND hi IN ({placeholders})
                """,
                ids + ids,
            ).fetchall()

        for lo, hi, weight in rows:
            i = index_of.get(int(lo))
            j = index_of.get(int(hi))

            if i is None or j is None:
                continue

            value = float(weight)
            accum[i, j] += value
            accum[j, i] += value

        row_sums = accum.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        normed = accum / row_sums
        logger.debug(
            "SQLiteActivationMemory.normalized_spread_matrix: nk=%d shape=%s row_sum_range=(%.6f,%.6f)",
            slot_count,
            tuple(normed.shape),
            float(row_sums.min().item()),
            float(row_sums.max().item()),
        )

        return normed

    def clear(self, *, namespace: Optional[str] = None, kind: Optional[str] = None) -> None:
        namespace_value = namespace or self.default_namespace

        with self._connect() as connection:
            ids = self._matching_ids(connection, namespace=namespace_value, kind=kind)
            self._delete_associations(connection, ids)

            if kind is None:
                connection.execute(
                    "DELETE FROM activation_memory WHERE namespace=?",
                    (namespace_value,),
                )
            else:
                connection.execute(
                    "DELETE FROM activation_memory WHERE namespace=? AND kind=?",
                    (namespace_value, kind),
                )

        logger.debug("SQLiteActivationMemory.clear: namespace=%s kind=%s", namespace_value, kind)

    def delete_records(self, record_ids: Sequence[int]) -> int:
        """Remove activation-memory rows and association edges by primary key."""

        ids = [int(record_id) for record_id in record_ids]

        if not ids:
            return 0

        with self._connect() as connection:
            self._delete_associations(connection, ids)
            placeholders = ",".join("?" for _ in ids)
            cursor = connection.execute(
                f"DELETE FROM activation_memory WHERE id IN ({placeholders})",
                ids,
            )
            deleted = int(cursor.rowcount) if cursor.rowcount is not None and cursor.rowcount >= 0 else len(ids)

        logger.debug("SQLiteActivationMemory.delete_records: n_ids=%s deleted=%s", len(ids), deleted)

        return deleted

    def count(self, *, namespace: Optional[str] = None, kind: Optional[str] = None) -> int:
        namespace_value = namespace or self.default_namespace

        with self._connect() as connection:
            if kind is None:
                row = connection.execute(
                    "SELECT COUNT(*) FROM activation_memory WHERE namespace=?",
                    (namespace_value,),
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT COUNT(*) FROM activation_memory WHERE namespace=? AND kind=?",
                    (namespace_value, kind),
                ).fetchone()

        count = int(row[0]) if row is not None else 0
        logger.debug("SQLiteActivationMemory.count: namespace=%s kind=%s n=%s", namespace_value, kind, count)

        return count

    def write(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        metadata: Optional[dict] = None,
        namespace: Optional[str] = None,
        kind: str = "fact",
        confidence: float = 1.0,
    ) -> int:
        namespace_value = namespace or self.default_namespace
        key_blob, key_dim = self.codec.to_blob(key)
        value_blob, value_dim = self.codec.to_blob(value)

        if key_dim != value_dim:
            raise ValueError(f"key dim {key_dim} != value dim {value_dim}")

        now = time.time()
        metadata_json = json.dumps(metadata or {}, sort_keys=True)

        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO activation_memory(
                    namespace, kind, dim, key_blob, value_blob, metadata_json,
                    confidence, access_count, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    namespace_value,
                    kind,
                    key_dim,
                    key_blob,
                    value_blob,
                    metadata_json,
                    float(confidence),
                    0,
                    now,
                    now,
                ),
            )
            record_id = int(cursor.lastrowid)

        if record_id <= 0:
            raise RuntimeError("activation_memory insert did not produce a primary key")

        logger.debug(
            "SQLiteActivationMemory.write: id=%s ns=%s kind=%s dim=%s conf=%s meta_keys=%s",
            record_id,
            namespace_value,
            kind,
            key_dim,
            float(confidence),
            sorted((metadata or {}).keys()),
        )

        return record_id

    def load(
        self,
        *,
        namespace: Optional[str] = None,
        kind: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[MemoryRecord]:
        namespace_value = namespace or self.default_namespace
        sql = [
            "SELECT id, namespace, kind, dim, key_blob, value_blob, metadata_json, confidence, access_count",
            "FROM activation_memory WHERE namespace=?",
        ]
        args: list[object] = [namespace_value]

        if kind is not None:
            sql.append("AND kind=?")
            args.append(kind)

        sql.append("ORDER BY id")

        if limit is not None:
            sql.append("LIMIT ?")
            args.append(int(limit))

        with self._connect() as connection:
            rows = connection.execute(" ".join(sql), args).fetchall()

        records = [self._row_to_record(row) for row in rows]
        logger.debug(
            "SQLiteActivationMemory.load: namespace=%s kind=%s n_records=%d",
            namespace_value,
            kind,
            len(records),
        )

        return records

    def retrieve(
        self,
        query: torch.Tensor,
        *,
        namespace: Optional[str] = None,
        kind: Optional[str] = None,
        top_k: int = 3,
        sim_chunk_rows: int = 512,
    ) -> list[tuple[MemoryRecord, float]]:
        """Return top cosine-similar records and bump access_count on matches."""

        records = self.load(namespace=namespace, kind=kind)

        if not records:
            return []

        q = F.normalize(query.detach().cpu().float().reshape(1, -1), dim=-1)
        chunk_size = max(1, int(sim_chunk_rows))
        sim_parts: list[torch.Tensor] = []

        for offset in range(0, len(records), chunk_size):
            batch = records[offset : offset + chunk_size]
            keys = F.normalize(torch.stack([record.key.float() for record in batch], dim=0), dim=-1)
            sim_parts.append((q @ keys.T).squeeze(0))

        sims = torch.cat(sim_parts, dim=0)
        values, indices = sims.topk(min(top_k, len(records)))
        ids = [records[int(index)].id for index in indices]

        self._bump_access(ids)
        pairs: list[tuple[MemoryRecord, float]] = []

        for value, tensor_index in zip(values, indices):
            record = records[int(tensor_index)]
            record.access_count += 1
            pairs.append((record, float(value)))

        logger.debug(
            "SQLiteActivationMemory.retrieve: namespace=%s kind=%s pool=%d top_k=%d tops=%s",
            namespace,
            kind,
            len(records),
            len(indices),
            [(record.id, score) for record, score in pairs],
        )

        return pairs

    def load_into_graft(
        self,
        graft: ActivationMemoryGraftProtocol,
        *,
        namespace: Optional[str] = None,
        kind: str = "fact",
        clear_first: bool = True,
    ) -> int:
        remember = getattr(graft, "remember", None)

        if not callable(remember):
            raise TypeError(
                "SQLiteActivationMemory.load_into_graft requires graft.remember to be callable; "
                f"got graft type={type(graft).__name__!r}",
            )

        records = self.load(namespace=namespace, kind=kind)

        if clear_first and hasattr(graft, "clear"):
            clear = getattr(graft, "clear", None)

            if not callable(clear):
                raise TypeError(
                    "SQLiteActivationMemory.load_into_graft: graft declares clear but graft.clear "
                    "is not callable — provide a callable clear(), or pass clear_first=False",
                )

            clear()

        for record in records:
            metadata = dict(record.metadata)
            metadata["memory_id"] = record.id
            metadata["confidence"] = record.confidence
            remember(record.key.reshape(1, -1), record.value.reshape(1, -1), metadata=metadata)

        spread = self.normalized_spread_matrix([record.id for record in records])
        setter = getattr(graft, "set_spread_matrix", None)

        if setter is not None:
            if not callable(setter):
                raise TypeError(
                    "SQLiteActivationMemory.load_into_graft: graft.set_spread_matrix must be callable when present "
                    f"(graft={type(graft).__name__!r})",
                )

            setter(spread if spread.numel() else None)

        logger.debug(
            "SQLiteActivationMemory.load_into_graft: ns=%s kind=%s n_loaded=%s spread_shape=%s",
            namespace,
            kind,
            len(records),
            tuple(spread.shape) if spread.numel() else None,
        )

        return len(records)

    def _init_schema(self) -> None:
        with self._connect() as connection:
            self.schema.initialize(connection)

    def _connect(self) -> sqlite3.Connection:
        return SQLiteActivationContext(self.connection.open(), self._lock)

    def _matching_ids(
        self,
        connection: sqlite3.Connection,
        *,
        namespace: str,
        kind: Optional[str],
    ) -> list[int]:
        if kind is None:
            rows = connection.execute(
                "SELECT id FROM activation_memory WHERE namespace=?",
                (namespace,),
            ).fetchall()
        else:
            rows = connection.execute(
                "SELECT id FROM activation_memory WHERE namespace=? AND kind=?",
                (namespace, kind),
            ).fetchall()

        return [int(row[0]) for row in rows]

    def _delete_associations(self, connection: sqlite3.Connection, record_ids: Sequence[int]) -> None:
        ids = [int(record_id) for record_id in record_ids]

        if not ids:
            return

        placeholders = ",".join("?" for _ in ids)
        connection.execute(
            f"DELETE FROM activation_association WHERE lo IN ({placeholders}) OR hi IN ({placeholders})",
            ids + ids,
        )

    def _row_to_record(self, row: tuple) -> MemoryRecord:
        return MemoryRecord(
            id=int(row[0]),
            namespace=str(row[1]),
            kind=str(row[2]),
            key=self.codec.to_tensor(row[4], int(row[3])),
            value=self.codec.to_tensor(row[5], int(row[3])),
            metadata=json.loads(row[6]),
            confidence=float(row[7]),
            access_count=int(row[8]),
        )

    def _bump_access(self, record_ids: Sequence[int]) -> None:
        ids = [int(record_id) for record_id in record_ids]

        if not ids:
            return

        placeholders = ",".join("?" for _ in ids)
        now = time.time()

        with self._connect() as connection:
            connection.execute(
                f"""
                UPDATE activation_memory
                SET access_count = access_count + 1, updated_at = ?
                WHERE id IN ({placeholders})
                """,
                [now] + ids,
            )


class SQLiteActivationContext:
    """Context manager that serializes SQLite writes through a shared lock."""

    def __init__(self, connection: sqlite3.Connection, lock: threading.RLock) -> None:
        self.connection = connection
        self.lock = lock

    def __enter__(self) -> sqlite3.Connection:
        self.lock.acquire()

        return self.connection.__enter__()

    def __exit__(self, exc_type, exc, tb) -> bool | None:
        try:
            return self.connection.__exit__(exc_type, exc, tb)
        finally:
            self.connection.close()
            self.lock.release()
