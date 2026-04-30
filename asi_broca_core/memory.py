from __future__ import annotations

import json
import sqlite3
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F


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


def _tensor_to_blob(t: torch.Tensor) -> tuple[bytes, int]:
    vec = t.detach().cpu().float().reshape(-1)
    data = struct.pack(f"<{vec.numel()}f", *[float(x) for x in vec.tolist()])
    return data, int(vec.numel())


def _blob_to_tensor(blob: bytes, dim: int) -> torch.Tensor:
    vals = struct.unpack(f"<{dim}f", blob)
    return torch.tensor(vals, dtype=torch.float32)


class SQLiteActivationMemory:
    """Persistent activation-space memory.

    Records are stored as hidden-state keys and hidden-state value directions.
    Loading this store into a graft changes model activations directly; it does
    not concatenate facts into prompts.
    """

    def __init__(self, path: str | Path, *, default_namespace: str = "main"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.default_namespace = default_namespace
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path)
        con.execute("PRAGMA journal_mode=WAL")
        return con

    def _init_schema(self) -> None:
        with self._connect() as con:
            con.execute(
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
            con.execute("CREATE INDEX IF NOT EXISTS idx_activation_namespace_kind ON activation_memory(namespace, kind)")
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS activation_association (
                    lo INTEGER NOT NULL,
                    hi INTEGER NOT NULL,
                    weight REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY(lo, hi)
                )
                """
            )
            con.execute("CREATE INDEX IF NOT EXISTS idx_activation_assoc_lo ON activation_association(lo)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_activation_assoc_hi ON activation_association(hi)")

    def bump_association(self, id_a: int, id_b: int, *, delta: float = 1.0) -> None:
        """Symmetric co-activation counter between activation_memory rows.

        Call this when two records are jointly recruited by the substrate (not for
        arbitrary insertion order); edges drive one spreading step in KVMemory.
        """

        ia, ib = int(id_a), int(id_b)
        if ia == ib:
            return
        lo, hi = (ia, ib) if ia < ib else (ib, ia)
        now = time.time()
        with self._connect() as con:
            row = con.execute(
                "SELECT weight FROM activation_association WHERE lo=? AND hi=?",
                (lo, hi),
            ).fetchone()
            w = float(row[0]) + float(delta) if row else float(delta)
            con.execute(
                """
                INSERT INTO activation_association(lo, hi, weight, updated_at)
                VALUES (?,?,?,?)
                ON CONFLICT(lo, hi) DO UPDATE SET weight=excluded.weight, updated_at=excluded.updated_at
                """,
                (lo, hi, w, now),
            )

    def normalized_spread_matrix(self, record_ids: list[int]) -> torch.Tensor:
        """Row-stochastic spread operator over ordered graft slots (derived self-loop + co-access mass)."""

        nk = len(record_ids)
        if nk == 0:
            return torch.empty(0, 0)
        index_of = {int(rid): i for i, rid in enumerate(record_ids)}
        accum = torch.eye(nk, dtype=torch.float32)
        if nk < 2:
            return accum

        ids_tuple = tuple(sorted(index_of.keys()))
        placeholders = ",".join("?" for _ in ids_tuple)
        sql = (
            f"SELECT lo, hi, weight FROM activation_association WHERE lo IN ({placeholders}) AND hi IN ({placeholders})"
        )
        args = ids_tuple + ids_tuple
        with self._connect() as con:
            rows = con.execute(sql, args).fetchall()
        for lo, hi, w in rows:
            i = index_of.get(int(lo))
            j = index_of.get(int(hi))
            if i is None or j is None:
                continue
            wf = float(w)
            accum[i, j] += wf
            accum[j, i] += wf
        row_sums = accum.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return accum / row_sums

    def clear(self, *, namespace: Optional[str] = None, kind: Optional[str] = None) -> None:
        namespace = namespace or self.default_namespace
        with self._connect() as con:
            if kind is None:
                con.execute("DELETE FROM activation_memory WHERE namespace=?", (namespace,))
            else:
                con.execute("DELETE FROM activation_memory WHERE namespace=? AND kind=?", (namespace, kind))

    def count(self, *, namespace: Optional[str] = None, kind: Optional[str] = None) -> int:
        namespace = namespace or self.default_namespace
        with self._connect() as con:
            if kind is None:
                row = con.execute("SELECT COUNT(*) FROM activation_memory WHERE namespace=?", (namespace,)).fetchone()
            else:
                row = con.execute("SELECT COUNT(*) FROM activation_memory WHERE namespace=? AND kind=?", (namespace, kind)).fetchone()
        return int(row[0])

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
        namespace = namespace or self.default_namespace
        key_blob, key_dim = _tensor_to_blob(key)
        value_blob, value_dim = _tensor_to_blob(value)
        if key_dim != value_dim:
            raise ValueError(f"key dim {key_dim} != value dim {value_dim}")
        now = time.time()
        with self._connect() as con:
            cur = con.execute(
                """
                INSERT INTO activation_memory
                (namespace, kind, dim, key_blob, value_blob, metadata_json, confidence, access_count, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                """,
                (namespace, kind, key_dim, key_blob, value_blob, json.dumps(metadata or {}, sort_keys=True), float(confidence), now, now),
            )
            return int(cur.lastrowid)

    def load(self, *, namespace: Optional[str] = None, kind: Optional[str] = None, limit: Optional[int] = None) -> list[MemoryRecord]:
        namespace = namespace or self.default_namespace
        sql = "SELECT id, namespace, kind, dim, key_blob, value_blob, metadata_json, confidence, access_count FROM activation_memory WHERE namespace=?"
        args: list[object] = [namespace]
        if kind is not None:
            sql += " AND kind=?"
            args.append(kind)
        sql += " ORDER BY id ASC"
        if limit is not None:
            sql += " LIMIT ?"
            args.append(int(limit))
        out: list[MemoryRecord] = []
        with self._connect() as con:
            for row in con.execute(sql, tuple(args)):
                rid, ns, kd, dim, kb, vb, mj, conf, access = row
                out.append(
                    MemoryRecord(
                        id=int(rid),
                        namespace=str(ns),
                        kind=str(kd),
                        key=_blob_to_tensor(kb, int(dim)),
                        value=_blob_to_tensor(vb, int(dim)),
                        metadata=json.loads(mj),
                        confidence=float(conf),
                        access_count=int(access),
                    )
                )
        return out

    def retrieve(
        self,
        query: torch.Tensor,
        *,
        namespace: Optional[str] = None,
        kind: Optional[str] = None,
        top_k: int = 3,
    ) -> list[tuple[MemoryRecord, float]]:
        records = self.load(namespace=namespace, kind=kind)
        if not records:
            return []
        q = F.normalize(query.detach().cpu().float().reshape(1, -1), dim=-1)
        keys = F.normalize(torch.stack([r.key.float() for r in records], dim=0), dim=-1)
        sims = (q @ keys.T).squeeze(0)
        vals, idxs = sims.topk(min(top_k, len(records)))
        ids = [records[int(i)].id for i in idxs]
        if ids:
            placeholders = ",".join("?" for _ in ids)
            now = time.time()
            with self._connect() as con:
                con.execute(
                    f"UPDATE activation_memory SET access_count=access_count+1, updated_at=? WHERE id IN ({placeholders})",
                    (now, *ids),
                )
        return [(records[int(i)], float(v)) for v, i in zip(vals, idxs)]

    def load_into_graft(self, graft, *, namespace: Optional[str] = None, kind: str = "fact", clear_first: bool = True) -> int:
        records = self.load(namespace=namespace, kind=kind)
        if clear_first and hasattr(graft, "clear"):
            graft.clear()
        for rec in records:
            meta = dict(rec.metadata)
            meta["memory_id"] = rec.id
            meta["confidence"] = rec.confidence
            graft.remember(rec.key.reshape(1, -1), rec.value.reshape(1, -1), metadata=meta)
        ids = [rec.id for rec in records]
        spread = self.normalized_spread_matrix(ids)
        setter = getattr(graft, "set_spread_matrix", None)
        if callable(setter):
            setter(spread if spread.numel() else None)
        return len(records)


