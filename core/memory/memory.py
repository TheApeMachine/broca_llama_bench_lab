from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, Sequence

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ActivationMemoryGraftProtocol(Protocol):
    """Required surface for :meth:`SQLiteActivationMemory.load_into_graft`.

    Many graft classes also expose ``clear()`` and ``set_spread_matrix``; those are
    optional and are invoked only when :meth:`~SQLiteActivationMemory.load_into_graft`
    is configured to use them (see runtime checks beside ``remember``, ``clear``,
    and ``set_spread_matrix`` there).
    """

    def remember(
        self, key: torch.Tensor, value: torch.Tensor, *, metadata: Optional[dict] = None
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


def _tensor_to_blob(t: torch.Tensor) -> tuple[bytes, int]:
    arr = np.ascontiguousarray(t.detach().cpu().numpy().astype("<f4", copy=False))
    return arr.tobytes(), int(arr.size)


def _blob_to_tensor(blob: bytes, dim: int) -> torch.Tensor:
    raw = np.frombuffer(blob, dtype=np.dtype("<f4"))
    expected = int(dim)
    if raw.size != expected:
        raise ValueError(f"blob size {raw.size} != declared dim {expected}")
    return torch.from_numpy(np.array(raw)).float()


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
        con = sqlite3.connect(self.path, timeout=5.0)
        row = con.execute("PRAGMA journal_mode=WAL").fetchone()
        mode_raw = row[0] if row else None
        mode = str(mode_raw).lower() if mode_raw is not None else ""
        if mode != "wal":
            logger.warning(
                "SQLiteActivationMemory(%s): expected journal_mode wal, got %r",
                self.path,
                mode_raw,
            )
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
            con.execute(
                """
                INSERT INTO activation_association(lo, hi, weight, updated_at)
                VALUES (?,?,?,?)
                ON CONFLICT(lo, hi) DO UPDATE SET
                    weight = activation_association.weight + excluded.weight,
                    updated_at = excluded.updated_at
                """,
                (lo, hi, float(delta), now),
            )
            row = con.execute(
                "SELECT weight FROM activation_association WHERE lo=? AND hi=?",
                (lo, hi),
            ).fetchone()
            w = float(row[0]) if row else float(delta)

            logger.debug("SQLiteActivationMemory.bump_association: pair=(%s,%s) weight=%s", lo, hi, w)

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
        normed = accum / row_sums
        logger.debug(
            "SQLiteActivationMemory.normalized_spread_matrix: nk=%d shape=%s row_sum_range=(%.6f,%.6f)",
            nk,
            tuple(normed.shape),
            float(row_sums.min().item()),
            float(row_sums.max().item()),
        )
        return normed

    def clear(self, *, namespace: Optional[str] = None, kind: Optional[str] = None) -> None:
        namespace = namespace or self.default_namespace

        with self._connect() as con:
            if kind is None:
                ids_subsel = "(SELECT id FROM activation_memory WHERE namespace=?)"
                assoc_params = (namespace, namespace)
            else:
                ids_subsel = "(SELECT id FROM activation_memory WHERE namespace=? AND kind=?)"
                assoc_params = (namespace, kind, namespace, kind)
            con.execute(
                f"DELETE FROM activation_association WHERE lo IN {ids_subsel} OR hi IN {ids_subsel}",
                assoc_params,
            )
            if kind is None:
                con.execute("DELETE FROM activation_memory WHERE namespace=?", (namespace,))
            else:
                con.execute("DELETE FROM activation_memory WHERE namespace=? AND kind=?", (namespace, kind))

        logger.debug("SQLiteActivationMemory.clear: namespace=%s kind=%s", namespace, kind)

    def delete_records(self, record_ids: Sequence[int]) -> int:
        """Remove activation_memory rows (and association edges touching them) by primary key."""

        ids = [int(i) for i in record_ids]
        if not ids:
            return 0
        placeholders = ",".join("?" for _ in ids)
        id_tuple = tuple(ids)
        with self._connect() as con:
            con.execute(
                f"DELETE FROM activation_association WHERE lo IN ({placeholders}) OR hi IN ({placeholders})",
                id_tuple + id_tuple,
            )
            cur = con.execute(f"DELETE FROM activation_memory WHERE id IN ({placeholders})", id_tuple)
            deleted = int(cur.rowcount) if cur.rowcount is not None and cur.rowcount >= 0 else len(ids)
        logger.debug("SQLiteActivationMemory.delete_records: n_ids=%s deleted=%s", len(ids), deleted)
        return deleted

    def count(self, *, namespace: Optional[str] = None, kind: Optional[str] = None) -> int:
        namespace = namespace or self.default_namespace

        with self._connect() as con:
            if kind is None:
                row = con.execute("SELECT COUNT(*) FROM activation_memory WHERE namespace=?", (namespace,)).fetchone()
            else:
                row = con.execute("SELECT COUNT(*) FROM activation_memory WHERE namespace=? AND kind=?", (namespace, kind)).fetchone()

        n = int(row[0])
        logger.debug("SQLiteActivationMemory.count: namespace=%s kind=%s n=%s", namespace, kind, n)
        return n

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

            rid = int(cur.lastrowid)
            meta = metadata or {}
            logger.debug(
                "SQLiteActivationMemory.write: id=%s ns=%s kind=%s dim=%s conf=%s meta_keys=%s",
                rid,
                namespace,
                kind,
                key_dim,
                float(confidence),
                sorted(meta.keys()),
            )
            return rid

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

        logger.debug("SQLiteActivationMemory.load: namespace=%s kind=%s n_records=%d", namespace, kind, len(out))
        return out

    def retrieve(
        self,
        query: torch.Tensor,
        *,
        namespace: Optional[str] = None,
        kind: Optional[str] = None,
        top_k: int = 3,
        sim_chunk_rows: int = 512,
    ) -> list[tuple[MemoryRecord, float]]:
        """Return top cosine-similar records and bump ``access_count`` on matches.

        Loads all matching SQLite rows then scores keys in-process. Similarity runs
        in batches over ``sim_chunk_rows`` vectors to bound peak RAM; for millions
        of rows per namespace, use an external vector index instead.
        """

        records = self.load(namespace=namespace, kind=kind)
        if not records:
            return []
        q = F.normalize(query.detach().cpu().float().reshape(1, -1), dim=-1)
        chunk = max(1, int(sim_chunk_rows))
        sim_parts: list[torch.Tensor] = []
        for off in range(0, len(records), chunk):
            batch = records[off : off + chunk]
            keys_b = F.normalize(
                torch.stack([r.key.float() for r in batch], dim=0),
                dim=-1,
            )
            sim_parts.append((q @ keys_b.T).squeeze(0))
        sims = torch.cat(sim_parts, dim=0)
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

        tops = [(int(records[int(i)].id), float(v)) for v, i in zip(vals, idxs)]
        logger.debug(
            "SQLiteActivationMemory.retrieve: namespace=%s kind=%s pool=%d top_k=%d tops=%s",
            namespace,
            kind,
            len(records),
            len(idxs),
            tops,
        )
        pairs: list[tuple[MemoryRecord, float]] = []
        for v, ti in zip(vals, idxs):
            ix = int(ti)
            rec = records[ix]
            rec.access_count += 1
            pairs.append((rec, float(v)))
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
            clr = getattr(graft, "clear", None)
            if clr is None or not callable(clr):
                raise TypeError(
                    "SQLiteActivationMemory.load_into_graft: graft declares clear but graft.clear "
                    "is not callable — provide a callable clear(), or pass clear_first=False",
                )
            clr()
        for rec in records:
            meta = dict(rec.metadata)
            meta["memory_id"] = rec.id
            meta["confidence"] = rec.confidence
            remember(rec.key.reshape(1, -1), rec.value.reshape(1, -1), metadata=meta)
        spread = self.normalized_spread_matrix([rec.id for rec in records])
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
