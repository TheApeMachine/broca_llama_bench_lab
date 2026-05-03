from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sqlalchemy import delete, func, or_
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlmodel import Session, col, select

from core.persistence.sqlite_engine import SqliteEngine

from .model import ActivationMemory, ActivationAssociation

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
        self.default_namespace = default_namespace
        self._engine = SqliteEngine.create(
            self.path,
            timeout_seconds=SqliteEngine.CONNECT_TIMEOUT_SECONDS,
        )
        self._init_schema()

    def _init_schema(self) -> None:
        SqliteEngine.create_tables(self._engine, ActivationMemory, ActivationAssociation)

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
        table = ActivationAssociation.__table__
        stmt = sqlite_insert(table).values(
            lo=lo,
            hi=hi,
            weight=float(delta),
            updated_at=now,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[table.c.lo, table.c.hi],
            set_={
                "weight": table.c.weight + stmt.excluded.weight,
                "updated_at": stmt.excluded.updated_at,
            },
        )

        with Session(self._engine) as session:
            session.exec(stmt)
            session.flush()
            row = session.get(ActivationAssociation, (lo, hi))
            if row is None:
                raise RuntimeError(f"association row ({lo},{hi}) missing after upsert")
            w = float(row.weight)
            session.commit()

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

        stmt = select(ActivationAssociation).where(
            col(ActivationAssociation.lo).in_(ids_tuple),
            col(ActivationAssociation.hi).in_(ids_tuple),
        )
        rows: list[tuple[int, int, float]] = []
        with Session(self._engine) as session:
            for assoc in session.exec(stmt).all():
                rows.append((int(assoc.lo), int(assoc.hi), float(assoc.weight)))
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

        with Session(self._engine) as session:
            id_subselect = (
                select(col(ActivationMemory.id)).where(ActivationMemory.namespace == namespace)
                if kind is None
                else select(col(ActivationMemory.id)).where(
                    ActivationMemory.namespace == namespace,
                    ActivationMemory.kind == kind,
                )
            )

            assoc_del = delete(ActivationAssociation).where(
                or_(
                    col(ActivationAssociation.lo).in_(id_subselect),
                    col(ActivationAssociation.hi).in_(id_subselect),
                ),
            )
        
            session.exec(assoc_del)
        
            row_del = (
                delete(ActivationMemory).where(ActivationMemory.namespace == namespace)
                if kind is None
                else delete(ActivationMemory).where(
                    ActivationMemory.namespace == namespace,
                    ActivationMemory.kind == kind,
                )
            )
        
            session.exec(row_del)
            session.commit()

        logger.debug("SQLiteActivationMemory.clear: namespace=%s kind=%s", namespace, kind)

    def delete_records(self, record_ids: Sequence[int]) -> int:
        """Remove activation_memory rows (and association edges touching them) by primary key."""

        ids = [int(i) for i in record_ids]
        
        if not ids:
            return 0
        
        with Session(self._engine) as session:
            assoc_del = delete(ActivationAssociation).where(
                or_(
                    col(ActivationAssociation.lo).in_(ids),
                    col(ActivationAssociation.hi).in_(ids),
                ),
            )
        
            session.exec(assoc_del)
            row_del = delete(ActivationMemory).where(col(ActivationMemory.id).in_(ids))
            cur = session.exec(row_del)
            session.commit()
            deleted = int(cur.rowcount) if getattr(cur, "rowcount", None) is not None and cur.rowcount >= 0 else len(ids)
        
        logger.debug("SQLiteActivationMemory.delete_records: n_ids=%s deleted=%s", len(ids), deleted)
        return deleted

    def count(self, *, namespace: Optional[str] = None, kind: Optional[str] = None) -> int:
        namespace = namespace or self.default_namespace

        stmt = (
            select(func.count())
            .select_from(ActivationMemory)
            .where(
                ActivationMemory.namespace == namespace,
            )
        
            if kind is None
            else select(func.count()).select_from(ActivationMemory).where(
                ActivationMemory.namespace == namespace,
                ActivationMemory.kind == kind,
            )
        )

        with Session(self._engine) as session:
            n = int(session.exec(stmt).one())

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
        row_in = ActivationMemory(
            namespace=namespace,
            kind=kind,
            dim=key_dim,
            key_blob=key_blob,
            value_blob=value_blob,
            metadata_json=json.dumps(metadata or {}, sort_keys=True),
            confidence=float(confidence),
            access_count=0,
            created_at=now,
            updated_at=now,
        )

        with Session(self._engine) as session:
            session.add(row_in)
            session.commit()
            session.refresh(row_in)

        rid = int(row_in.id) if row_in.id is not None else 0
        
        if rid <= 0:
            raise RuntimeError("activation_memory insert did not produce a primary key")

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
        stmt = select(ActivationMemory).where(ActivationMemory.namespace == namespace)
        
        if kind is not None:
            stmt = stmt.where(ActivationMemory.kind == kind)
        
        stmt = stmt.order_by(col(ActivationMemory.id))
        
        if limit is not None:
            stmt = stmt.limit(int(limit))

        out: list[MemoryRecord] = []
        
        with Session(self._engine) as session:
            for row_db in session.exec(stmt).all():
                out.append(
                    MemoryRecord(
                        id=int(row_db.id),
                        namespace=str(row_db.namespace),
                        kind=str(row_db.kind),
                        key=_blob_to_tensor(row_db.key_blob, int(row_db.dim)),
                        value=_blob_to_tensor(row_db.value_blob, int(row_db.dim)),
                        metadata=json.loads(row_db.metadata_json),
                        confidence=float(row_db.confidence),
                        access_count=int(row_db.access_count),
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

        now = time.time()
        
        if ids:
            with Session(self._engine) as session:
                stm = select(ActivationMemory).where(col(ActivationMemory.id).in_(ids))
        
                for bump_row in session.exec(stm).all():
                    bump_row.access_count = int(bump_row.access_count) + 1
                    bump_row.updated_at = now
                    session.add(bump_row)
        
                session.commit()

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
