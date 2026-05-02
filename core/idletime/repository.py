"""SQLite persistence for the ontological registry (counts + promoted axes)."""

from __future__ import annotations

import base64
import binascii
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


def encode_fp32_blob(tensor: torch.Tensor) -> str:
    """Base64 of contiguous little-endian fp32 bytes for one flattened vector."""

    arr = np.ascontiguousarray(
        tensor.detach().cpu().numpy().astype("<f4", copy=False).reshape(-1)
    )
    return base64.standard_b64encode(arr.tobytes()).decode("ascii")


def decode_fp32_blob(serialized: str, *, expected_nelem: int, field: str) -> torch.Tensor:
    """Decode base64 fp32 blob; raises ``ValueError`` if length or format is wrong."""

    s = serialized.strip()

    if expected_nelem <= 0:
        raise ValueError(f"{field}: expected_nelem must be positive, got {expected_nelem}")

    try:
        raw = base64.standard_b64decode(s.encode("ascii"))
    except (binascii.Error, ValueError) as e:
        raise ValueError(
            f"{field}: expected base64-encoded fp32 blob ({expected_nelem * 4} bytes "
            f"for dim={expected_nelem})",
        ) from e

    want = expected_nelem * 4
    if len(raw) != want:
        raise ValueError(
            f"{field}: decoded {len(raw)} bytes but expected {want} for dim={expected_nelem}",
        )

    vec = torch.from_numpy(np.frombuffer(raw, dtype=np.dtype("<f4")).copy()).float()
    return vec.reshape(expected_nelem).contiguous()


@dataclass(frozen=True)
class PromotedPersistRow:
    """One promoted concept row ready for INSERT (columns historically named ``*_json``)."""

    name: str
    axis_b64: str
    base_sketch_b64: str
    promoted_at: float
    access_count: int


class OntologicalRepository:
    """Connection, schema, and SQL for ``ontological_registry`` / ``ontological_counts``."""

    def __init__(self, path: str | Path, *, namespace: str = "main") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.namespace = namespace
        self._conn: sqlite3.Connection | None = None
        self._conn_lock = threading.RLock()

    def _open_if_needed_unlocked(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.path), timeout=30.0, check_same_thread=False
            )
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.isolation_level = None
        return self._conn

    def close(self) -> None:
        with self._conn_lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def init_schema(self) -> None:
        with self._conn_lock:
            con = self._open_if_needed_unlocked()
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS ontological_registry (
                    namespace TEXT NOT NULL,
                    name TEXT NOT NULL,
                    axis_json TEXT NOT NULL,
                    base_sketch_json TEXT NOT NULL,
                    promoted_at REAL NOT NULL,
                    access_count INTEGER NOT NULL,
                    PRIMARY KEY(namespace, name)
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS ontological_counts (
                    namespace TEXT NOT NULL,
                    name TEXT NOT NULL,
                    count INTEGER NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY(namespace, name)
                )
                """
            )

    def replace_all(
        self,
        *,
        promoted: Sequence[PromotedPersistRow],
        access_counts: dict[str, int],
    ) -> None:
        """Sync namespace to match ``promoted`` and ``access_counts`` (delete stale keys, upsert)."""

        now = time.time()
        promoted_names = {row.name for row in promoted}
        access_names = set(access_counts.keys())
        ns = self.namespace

        with self._conn_lock:
            con = self._open_if_needed_unlocked()

            if promoted_names:
                placeholders = ",".join("?" * len(promoted_names))
                con.execute(
                    f"DELETE FROM ontological_registry WHERE namespace=? AND name NOT IN ({placeholders})",
                    (ns, *promoted_names),
                )
            else:
                con.execute("DELETE FROM ontological_registry WHERE namespace=?", (ns,))

            if access_names:
                placeholders = ",".join("?" * len(access_names))
                con.execute(
                    f"DELETE FROM ontological_counts WHERE namespace=? AND name NOT IN ({placeholders})",
                    (ns, *access_names),
                )
            else:
                con.execute("DELETE FROM ontological_counts WHERE namespace=?", (ns,))

            for row in promoted:
                con.execute(
                    """
                    INSERT INTO ontological_registry(namespace, name, axis_json, base_sketch_json, promoted_at, access_count)
                    VALUES (?,?,?,?,?,?)
                    ON CONFLICT(namespace, name) DO UPDATE SET
                        axis_json=excluded.axis_json,
                        base_sketch_json=excluded.base_sketch_json,
                        promoted_at=excluded.promoted_at,
                        access_count=excluded.access_count
                    """,
                    (
                        ns,
                        row.name,
                        row.axis_b64,
                        row.base_sketch_b64,
                        float(row.promoted_at),
                        int(row.access_count),
                    ),
                )

            for name, count in access_counts.items():
                con.execute(
                    """
                    INSERT INTO ontological_counts(namespace, name, count, updated_at)
                    VALUES (?,?,?,?)
                    ON CONFLICT(namespace, name) DO UPDATE SET count=excluded.count, updated_at=excluded.updated_at
                    """,
                    (ns, name, int(count), now),
                )

    def fetch_counts(self) -> list[tuple[str, int]]:
        with self._conn_lock:
            con = self._open_if_needed_unlocked()
            rows = con.execute(
                "SELECT name, count FROM ontological_counts WHERE namespace=?",
                (self.namespace,),
            ).fetchall()
        return [(str(name), int(count)) for name, count in rows]

    def fetch_promoted(self) -> list[PromotedPersistRow]:
        with self._conn_lock:
            con = self._open_if_needed_unlocked()
            rows = con.execute(
                "SELECT name, axis_json, base_sketch_json, promoted_at, access_count "
                "FROM ontological_registry WHERE namespace=?",
                (self.namespace,),
            ).fetchall()
        return [
            PromotedPersistRow(
                name=str(name),
                axis_b64=str(axis_b64),
                base_sketch_b64=str(base_b64),
                promoted_at=float(promoted_at),
                access_count=int(access_count),
            )
            for name, axis_b64, base_b64, promoted_at, access_count in rows
        ]
