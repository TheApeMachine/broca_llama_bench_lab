"""Persistent associative structure between substrate episodes (journal ids).

Edges accumulate when successive cognitive episodes are processed; strengths are
counts (no manual affinity knobs).
"""

from __future__ import annotations

import logging
import math
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
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_episode_assoc_lo ON episode_association(lo)"
            )
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_episode_assoc_hi ON episode_association(hi)"
            )

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
            logger.debug(
                "EpisodeAssociationGraph.bump: lo=%s hi=%s weight=%s", lo, hi, w
            )

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

    def decay_all(
        self, *, gamma: float = 0.99, prune_below: float = 0.01
    ) -> tuple[int, int]:
        """Apply thermodynamic decay to every edge and prune the survivors.

        Returns ``(decayed_edges, pruned_edges)`` so the DMN telemetry can
        attribute work to this phase. ``gamma`` is the multiplicative damping
        factor applied per tick; ``prune_below`` is the weight floor under
        which edges are deleted as having decayed past usefulness.
        """

        g = float(gamma)
        floor = float(prune_below)
        if not (0.0 < g <= 1.0):
            raise ValueError("gamma must be in (0, 1]")
        if not (0.0 <= floor < 1.0) or not math.isfinite(floor):
            raise ValueError(
                f"prune_below must be finite and in [0.0, 1.0), got {prune_below!r}"
            )
        with self._connect() as con:
            decayed_cur = con.execute(
                "UPDATE episode_association SET weight = weight * ?, updated_at = ?",
                (g, time.time()),
            )
            decayed = int(decayed_cur.rowcount or 0)
            pruned_cur = con.execute(
                "DELETE FROM episode_association WHERE weight < ?",
                (floor,),
            )
            pruned = int(pruned_cur.rowcount or 0)
        logger.debug(
            "EpisodeAssociationGraph.decay_all: gamma=%.4f floor=%.4f decayed=%d pruned=%d",
            g,
            floor,
            decayed,
            pruned,
        )
        return decayed, pruned

    def edges(self, *, min_weight: float = 0.0) -> list[tuple[int, int, float]]:
        """All edges above ``min_weight`` (lo, hi, weight). Used for centrality + dream walks."""

        with self._connect() as con:
            rows = con.execute(
                "SELECT lo, hi, weight FROM episode_association WHERE weight >= ? ORDER BY weight DESC",
                (float(min_weight),),
            ).fetchall()
        return [(int(r[0]), int(r[1]), float(r[2])) for r in rows]

    def neighbors(
        self, episode_id: int, *, limit: int = 16, min_weight: float = 0.0
    ) -> list[tuple[int, float]]:
        """Top-weighted neighbors of an episode, used for transitive episode closure."""

        nid = int(episode_id)
        lim = max(1, int(limit))
        with self._connect() as con:
            rows = con.execute(
                """
                SELECT CASE WHEN lo=? THEN hi ELSE lo END AS other, weight
                FROM episode_association
                WHERE (lo=? OR hi=?) AND weight >= ?
                ORDER BY weight DESC LIMIT ?
                """,
                (nid, nid, nid, float(min_weight), lim),
            ).fetchall()
        return [(int(r[0]), float(r[1])) for r in rows]

    def centrality(
        self, *, damping: float = 0.85, iterations: int = 20, min_weight: float = 0.0
    ) -> dict[int, float]:
        """PageRank over the surviving episode association graph.

        Returns the stationary distribution over episode ids; nodes that the
        DMN's random walk visits often are mathematically the most central
        across the user's whole journal, and the worker uses that score to
        boost the confidence of semantic facts that cite those episodes.
        """

        edges = self.edges(min_weight=min_weight)
        if not edges:
            return {}
        nodes: set[int] = set()
        out_weight: dict[int, float] = {}
        adj: dict[int, list[tuple[int, float]]] = {}
        for lo, hi, w in edges:
            nodes.add(lo)
            nodes.add(hi)
            adj.setdefault(lo, []).append((hi, w))
            adj.setdefault(hi, []).append((lo, w))
            out_weight[lo] = out_weight.get(lo, 0.0) + w
            out_weight[hi] = out_weight.get(hi, 0.0) + w
        n = len(nodes)
        if n == 0:
            return {}
        try:
            d = float(damping)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"damping must be numeric, got {damping!r}") from exc
        if not (0.0 <= d <= 1.0) or not math.isfinite(d):
            raise ValueError(
                f"damping must be finite and in [0.0, 1.0], got {damping!r}"
            )
        teleport = (1.0 - d) / n
        rank = {node: 1.0 / n for node in nodes}
        for _ in range(max(1, int(iterations))):
            new_rank = {node: teleport for node in nodes}
            for src, neighbors in adj.items():
                total = out_weight[src]
                share = d * rank[src] / total
                for dst, w in neighbors:
                    new_rank[dst] += share * w
            rank = new_rank
        # normalize to sum 1 in case rounding drifted
        total = sum(rank.values()) or 1.0
        return {node: float(score / total) for node, score in rank.items()}


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
