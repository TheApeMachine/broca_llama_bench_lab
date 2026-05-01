"""Persistent associative structure between substrate episodes (journal ids).

Edges accumulate when successive cognitive episodes are processed; strengths are
counts (no manual affinity knobs).
"""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class EpisodeAssociationGraph:
    """SQLite-backed symmetric edge weights between workspace_journal row ids."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path)
        con.execute("PRAGMA journal_mode=WAL")
        return con

    def _init_schema(self) -> None:
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS episode_association (
                    lo INTEGER NOT NULL,
                    hi INTEGER NOT NULL,
                    weight REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY(lo, hi)
                )
                """
            )
            con.execute("CREATE INDEX IF NOT EXISTS idx_episode_assoc_lo ON episode_association(lo)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_episode_assoc_hi ON episode_association(hi)")

    def bump(self, episode_id_a: int, episode_id_b: int, *, delta: float = 1.0) -> None:
        ia, ib = int(episode_id_a), int(episode_id_b)
        if ia == ib:
            return
        lo, hi = (ia, ib) if ia < ib else (ib, ia)
        now = time.time()
        with self._connect() as con:
            row = con.execute(
                "SELECT weight FROM episode_association WHERE lo=? AND hi=?",
                (lo, hi),
            ).fetchone()
            w = float(row[0]) + float(delta) if row else float(delta)
            con.execute(
                """
                INSERT INTO episode_association(lo, hi, weight, updated_at)
                VALUES (?,?,?,?)
                ON CONFLICT(lo, hi) DO UPDATE SET weight=excluded.weight, updated_at=excluded.updated_at
                """,
                (lo, hi, w, now),
            )
            logger.debug("EpisodeAssociationGraph.bump: lo=%s hi=%s weight=%s", lo, hi, w)

    def weight(self, episode_id_a: int, episode_id_b: int) -> float:
        ia, ib = int(episode_id_a), int(episode_id_b)
        if ia == ib:
            return 0.0
        lo, hi = (ia, ib) if ia < ib else (ib, ia)
        with self._connect() as con:
            row = con.execute(
                "SELECT weight FROM episode_association WHERE lo=? AND hi=?",
                (lo, hi),
            ).fetchone()
        return float(row[0]) if row else 0.0


def merge_epistemic_evidence_dict(base: dict, incoming: dict) -> dict:
    """Union-merge provenance lists used across semantic rows and frames."""

    out = dict(base)
    for key in ("episode_ids", "instruments"):
        if key not in incoming:
            continue
        cur = list(out.get(key) or [])
        seen = set(cur)
        for x in incoming[key]:
            if x not in seen:
                seen.add(x)
                cur.append(x)
        out[key] = cur
    if "journal_id" in incoming and incoming["journal_id"] is not None:
        jcur = list(out.get("episode_ids") or [])
        jid = int(incoming["journal_id"])
        if jid not in jcur:
            jcur.append(jid)
            out["episode_ids"] = jcur
    return out
