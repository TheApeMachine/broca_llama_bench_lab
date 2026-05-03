"""SymbolicMemory — SQLite-backed factual store for the cognitive substrate.

The symbolic layer holds three concerns under one schema:

* ``semantic_memory`` — current ``(subject, predicate, object)`` triples with
  confidence and evidence. One row per ``(namespace, subject, predicate)``.
* ``semantic_claims`` — every observation that ever landed, with status
  (``observed`` / ``accepted`` / ``corroborated`` / ``conflict``). The DMN
  reads these to drive belief revision.
* ``memory_reflections`` — the substrate's notes to itself when consolidation
  flips a fact or notices an unresolved conflict.

This is deliberately separate from prompt context. The language module asks
the substrate for a memory result; it does not receive a pasted fact list.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from ..substrate.graph import merge_epistemic_evidence_dict
from .claim_trust import ClaimTrust


logger = logging.getLogger(__name__)


BELIEF_REVISION_LOG_ODDS_THRESHOLD: float = 0.5
BELIEF_REVISION_MIN_CLAIMS: int = 2


class SymbolicMemory:
    """SQLite-backed symbolic/semantic memory for the cognitive substrate."""

    def __init__(self, path: str | Path, *, namespace: str = "main"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.namespace = namespace
        self._sqlite_lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self._init_schema()

    def _ensure_conn(self) -> sqlite3.Connection:
        """Lazy single connection per memory store (WAL, safe across threads via lock)."""

        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.path),
                check_same_thread=False,
                timeout=30.0,
                isolation_level=None,
            )
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=60000")
        return self._conn

    def close(self) -> None:
        with self._sqlite_lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _init_schema(self) -> None:
        with self._sqlite_lock:
            con = self._ensure_conn()
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS semantic_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    namespace TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    evidence_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    UNIQUE(namespace, subject, predicate)
                )
                """
            )
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_semantic_lookup ON semantic_memory(namespace, subject, predicate)"
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS semantic_claims (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    namespace TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    status TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_claim_lookup ON semantic_claims(namespace, subject, predicate, status)"
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_reflections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    namespace TEXT NOT NULL,
                    dedupe_key TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    UNIQUE(namespace, dedupe_key)
                )
                """
            )
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_reflection_lookup ON memory_reflections(namespace, kind, subject, predicate)"
            )

    def upsert(
        self,
        subject: str,
        predicate: str,
        obj: str,
        *,
        confidence: float = 1.0,
        evidence: dict | None = None,
        con: sqlite3.Connection | None = None,
    ) -> None:
        now = time.time()
        row = (
            self.namespace,
            subject.lower(),
            predicate.lower(),
            obj.lower(),
            float(confidence),
            json.dumps(evidence or {}, sort_keys=True),
            now,
            now,
        )
        sql = """
            INSERT INTO semantic_memory(namespace, subject, predicate, object, confidence, evidence_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(namespace, subject, predicate)
            DO UPDATE SET object=excluded.object, confidence=excluded.confidence,
                          evidence_json=excluded.evidence_json, updated_at=excluded.updated_at
        """
        if con is not None:
            con.execute(sql, row)
            return
        with self._sqlite_lock:
            self._ensure_conn().execute(sql, row)

    def record_claim(
        self,
        subject: str,
        predicate: str,
        obj: str,
        *,
        confidence: float = 1.0,
        status: str = "observed",
        evidence: dict | None = None,
    ) -> int:
        now = time.time()
        with self._sqlite_lock:
            con = self._ensure_conn()
            cur = con.execute(
                """
                INSERT INTO semantic_claims(namespace, subject, predicate, object, confidence, status, evidence_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.namespace,
                    subject.lower(),
                    predicate.lower(),
                    obj.lower(),
                    float(confidence),
                    str(status),
                    json.dumps(evidence or {}, sort_keys=True),
                    now,
                ),
            )
            return int(cur.lastrowid)

    def claims(
        self,
        subject: str | None = None,
        predicate: str | None = None,
        *,
        status: str | None = None,
    ) -> list[dict]:
        clauses = ["namespace=?"]
        args: list[Any] = [self.namespace]
        if subject is not None:
            clauses.append("subject=?")
            args.append(subject.lower())
        if predicate is not None:
            clauses.append("predicate=?")
            args.append(predicate.lower())
        if status is not None:
            clauses.append("status=?")
            args.append(status)
        sql = (
            "SELECT id, subject, predicate, object, confidence, status, evidence_json, created_at "
            f"FROM semantic_claims WHERE {' AND '.join(clauses)} ORDER BY id"
        )
        with self._sqlite_lock:
            rows = self._ensure_conn().execute(sql, args).fetchall()
        return [
            {
                "id": int(r[0]),
                "subject": str(r[1]),
                "predicate": str(r[2]),
                "object": str(r[3]),
                "confidence": float(r[4]),
                "status": str(r[5]),
                "evidence": json.loads(r[6]),
                "created_at": float(r[7]),
            }
            for r in rows
        ]

    def record_reflection(
        self,
        kind: str,
        subject: str,
        predicate: str,
        summary: str,
        evidence: dict,
        *,
        dedupe_key: str,
        con: sqlite3.Connection | None = None,
    ) -> int | None:
        now = time.time()
        params = (
            self.namespace,
            dedupe_key,
            kind,
            subject.lower(),
            predicate.lower(),
            summary,
            json.dumps(evidence, sort_keys=True),
            now,
        )
        sql = """
            INSERT OR IGNORE INTO memory_reflections(namespace, dedupe_key, kind, subject, predicate, summary, evidence_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        if con is not None:
            cur = con.execute(sql, params)
            return None if cur.rowcount == 0 else int(cur.lastrowid)
        with self._sqlite_lock:
            cur = self._ensure_conn().execute(sql, params)
            return None if cur.rowcount == 0 else int(cur.lastrowid)

    def reflections(self, *, kind: str | None = None) -> list[dict]:
        clauses = ["namespace=?"]
        args: list[Any] = [self.namespace]
        if kind is not None:
            clauses.append("kind=?")
            args.append(kind)
        sql = (
            "SELECT id, dedupe_key, kind, subject, predicate, summary, evidence_json, created_at "
            f"FROM memory_reflections WHERE {' AND '.join(clauses)} ORDER BY id"
        )
        with self._sqlite_lock:
            rows = self._ensure_conn().execute(sql, args).fetchall()
        return [
            {
                "id": int(r[0]),
                "dedupe_key": str(r[1]),
                "kind": str(r[2]),
                "subject": str(r[3]),
                "predicate": str(r[4]),
                "summary": str(r[5]),
                "evidence": json.loads(r[6]),
                "created_at": float(r[7]),
            }
            for r in rows
        ]

    def consolidate_claims_once(
        self,
        *,
        log_odds_threshold: float = BELIEF_REVISION_LOG_ODDS_THRESHOLD,
        min_claims: int = BELIEF_REVISION_MIN_CLAIMS,
    ) -> list[dict]:
        with self._sqlite_lock:
            claims = self.claims()
            grouped: dict[tuple[str, str], list[dict]] = {}
            for claim in claims:
                grouped.setdefault((claim["subject"], claim["predicate"]), []).append(claim)

            gap_stats = ClaimTrust.population_stats(claims)
            reflections: list[dict] = []
            for (subject, predicate), rows in grouped.items():
                if len({r["object"] for r in rows}) < 2:
                    continue
                support: dict[str, dict[str, Any]] = {}
                for row in rows:
                    entry = support.setdefault(
                        row["object"],
                        {"score": 0.0, "count": 0, "claim_ids": [], "trust_weights": []},
                    )
                    trust = ClaimTrust.weight(row, stats=gap_stats)
                    entry["score"] += float(row["confidence"]) * trust
                    entry["count"] += 1
                    entry["claim_ids"].append(int(row["id"]))
                    entry["trust_weights"].append(float(trust))

                current = self.get(subject, predicate)
                current_obj = current[0] if current is not None else ""
                current_score = float(support.get(current_obj, {}).get("score", 0.0))
                best_obj, best = max(
                    support.items(),
                    key=lambda item: (float(item[1]["score"]), int(item[1]["count"])),
                )
                best_score = float(best["score"])
                best_count = int(best["count"])
                log_odds = math.log(max(best_score, 1e-12)) - math.log(max(current_score, 1e-12))
                evidence = {
                    "support": support,
                    "current_object": current_obj,
                    "candidate_object": best_obj,
                    "log_odds": float(log_odds),
                    "log_odds_threshold": float(log_odds_threshold),
                    "min_claims": int(min_claims),
                    "gap_stats": (
                        {"mu": float(gap_stats[0]), "sigma": float(gap_stats[1])}
                        if gap_stats
                        else None
                    ),
                    "instrument": "background_claim_consolidation",
                }

                if (
                    current_obj
                    and best_obj != current_obj
                    and best_count >= int(min_claims)
                    and log_odds >= float(log_odds_threshold)
                ):
                    claim_ids_digest = hashlib.sha256(
                        json.dumps(
                            sorted(int(i) for i in best["claim_ids"]),
                            separators=(",", ":"),
                        ).encode()
                    ).hexdigest()
                    dedupe = (
                        f"belief_revision:{subject}:{predicate}:"
                        f"{current_obj}->{best_obj}:{claim_ids_digest}"
                    )
                    con = self._ensure_conn()
                    if con.in_transaction:
                        con.rollback()
                    con.execute("BEGIN")
                    try:
                        reflection_id = self.record_reflection(
                            "belief_revision",
                            subject,
                            predicate,
                            f"revised {subject}.{predicate} from {current_obj} to {best_obj}",
                            evidence,
                            dedupe_key=dedupe,
                            con=con,
                        )
                        if reflection_id is not None:
                            self.upsert(
                                subject,
                                predicate,
                                best_obj,
                                confidence=min(
                                    1.0,
                                    best_score
                                    / max(1.0, sum(float(v["score"]) for v in support.values())),
                                ),
                                evidence={**evidence, "reflection_id": reflection_id},
                                con=con,
                            )
                            reflections.append(
                                {"id": reflection_id, "kind": "belief_revision", **evidence}
                            )
                        con.commit()
                    except Exception:
                        con.rollback()
                        raise
                else:
                    dedupe = (
                        f"belief_conflict:{subject}:{predicate}:"
                        f"{','.join(str(r['id']) for r in rows)}"
                    )
                    reflection_id = self.record_reflection(
                        "belief_conflict",
                        subject,
                        predicate,
                        f"unresolved conflict over {subject}.{predicate}",
                        evidence,
                        dedupe_key=dedupe,
                    )
                    if reflection_id is not None:
                        reflections.append(
                            {"id": reflection_id, "kind": "belief_conflict", **evidence}
                        )
            return reflections

    def observe_claim(
        self,
        subject: str,
        predicate: str,
        obj: str,
        *,
        confidence: float = 1.0,
        evidence: dict | None = None,
    ) -> dict:
        subj = subject.lower()
        pred = predicate.lower()
        observed_obj = obj.lower()
        ev = dict(evidence or {})
        current = self.get(subj, pred)
        if current is None:
            claim_id = self.record_claim(
                subj, pred, observed_obj, confidence=confidence, status="accepted", evidence=ev
            )
            self.upsert(
                subj,
                pred,
                observed_obj,
                confidence=confidence,
                evidence={**ev, "claim_id": claim_id, "claim_status": "accepted"},
            )
            return {
                "status": "accepted",
                "claim_id": claim_id,
                "current_object": observed_obj,
                "observed_object": observed_obj,
            }

        current_obj, current_conf, current_ev = current
        if current_obj == observed_obj:
            claim_id = self.record_claim(
                subj,
                pred,
                observed_obj,
                confidence=confidence,
                status="corroborated",
                evidence=ev,
            )
            merged_ev = merge_epistemic_evidence_dict(
                dict(current_ev), {**ev, "claim_id": claim_id, "claim_status": "corroborated"}
            )
            self.upsert(
                subj,
                pred,
                observed_obj,
                confidence=max(float(current_conf), float(confidence)),
                evidence=merged_ev,
            )
            return {
                "status": "corroborated",
                "claim_id": claim_id,
                "current_object": current_obj,
                "observed_object": observed_obj,
            }

        conflict_ev = {
            **ev,
            "conflicts_with": {
                "subject": subj,
                "predicate": pred,
                "object": current_obj,
                "confidence": current_conf,
                "evidence": current_ev,
            },
            "counterfactual": {
                "if_accepted": {"subject": subj, "predicate": pred, "object": observed_obj},
                "would_change_answer_from": current_obj,
                "would_change_answer_to": observed_obj,
            },
        }
        claim_id = self.record_claim(
            subj, pred, observed_obj, confidence=confidence, status="conflict", evidence=conflict_ev
        )
        return {
            "status": "conflict",
            "claim_id": claim_id,
            "current_object": current_obj,
            "current_confidence": current_conf,
            "observed_object": observed_obj,
            "counterfactual": conflict_ev["counterfactual"],
        }

    def get(self, subject: str, predicate: str) -> tuple[str, float, dict] | None:
        with self._sqlite_lock:
            row = self._ensure_conn().execute(
                "SELECT object, confidence, evidence_json FROM semantic_memory "
                "WHERE namespace=? AND subject=? AND predicate=?",
                (self.namespace, subject.lower(), predicate.lower()),
            ).fetchone()
        if row is None:
            return None
        return str(row[0]), float(row[1]), json.loads(row[2])

    def subjects_for_predicate(self, predicate: str) -> list[str]:
        """All subjects with this predicate — drives intrinsic cues without a fixed ENTITY list."""

        pred = predicate.lower()
        with self._sqlite_lock:
            rows = self._ensure_conn().execute(
                "SELECT DISTINCT subject FROM semantic_memory "
                "WHERE namespace=? AND predicate=? ORDER BY subject",
                (self.namespace, pred),
            ).fetchall()
        return [str(r[0]) for r in rows]

    def subjects(self) -> list[str]:
        with self._sqlite_lock:
            rows = self._ensure_conn().execute(
                "SELECT DISTINCT subject FROM semantic_memory WHERE namespace=? ORDER BY subject",
                (self.namespace,),
            ).fetchall()
        return [str(r[0]) for r in rows]

    def records_for_subject(
        self, subject: str
    ) -> list[tuple[str, str, float, dict]]:
        with self._sqlite_lock:
            rows = self._ensure_conn().execute(
                """
                SELECT predicate, object, confidence, evidence_json
                FROM semantic_memory
                WHERE namespace=? AND subject=?
                ORDER BY updated_at DESC, id DESC
                """,
                (self.namespace, subject.lower()),
            ).fetchall()
        return [(str(r[0]), str(r[1]), float(r[2]), json.loads(r[3])) for r in rows]

    def distinct_objects_for_predicate(self, predicate: str) -> frozenset[str]:
        """Known objects already stored for a predicate; used for conflict detection."""

        pred = predicate.lower()
        with self._sqlite_lock:
            rows = self._ensure_conn().execute(
                "SELECT DISTINCT object FROM semantic_memory WHERE namespace=? AND predicate=?",
                (self.namespace, pred),
            ).fetchall()
        return frozenset(str(r[0]).lower() for r in rows)

    def count(self) -> int:
        with self._sqlite_lock:
            row = self._ensure_conn().execute(
                "SELECT COUNT(*) FROM semantic_memory WHERE namespace=?", (self.namespace,)
            ).fetchone()
        return int(row[0])

    def mean_confidence(self) -> float | None:
        with self._sqlite_lock:
            row = self._ensure_conn().execute(
                "SELECT AVG(confidence), COUNT(*) FROM semantic_memory WHERE namespace=?",
                (self.namespace,),
            ).fetchone()
        if row is None or int(row[1]) == 0:
            return None
        return float(row[0])

    def merge_epistemic_evidence(self, subject: str, predicate: str, incoming: dict) -> None:
        got = self.get(subject, predicate)
        if got is None:
            return
        obj, conf, ev_old = got
        merged_ev = merge_epistemic_evidence_dict(dict(ev_old), incoming)
        self.upsert(subject, predicate, obj, confidence=conf, evidence=merged_ev)

    def all_facts(self) -> list[tuple[str, str, str, float, dict]]:
        """Every (subject, predicate, object, confidence, evidence) row in this namespace."""

        with self._sqlite_lock:
            rows = self._ensure_conn().execute(
                "SELECT subject, predicate, object, confidence, evidence_json "
                "FROM semantic_memory WHERE namespace=?",
                (self.namespace,),
            ).fetchall()
        return [
            (str(r[0]), str(r[1]), str(r[2]), float(r[3]), json.loads(r[4]))
            for r in rows
        ]

    def boost_confidence(
        self,
        subject: str,
        predicate: str,
        *,
        factor: float,
        cap: float = 1.0,
        reason: str | None = None,
    ) -> tuple[str, float, float] | None:
        """Multiplicatively raise the confidence of a stored fact (DMN reinforcement).

        Returns ``(object, old_conf, new_conf)`` so callers can log the move,
        or ``None`` if the row doesn't exist.
        """

        got = self.get(subject, predicate)
        if got is None:
            return None
        obj, conf, ev_old = got
        new_conf = float(min(float(cap), max(0.0, conf * float(factor))))
        ev_new = dict(ev_old)
        notes = list(ev_new.get("dmn_consolidation") or [])
        notes.append(
            {
                "ts": time.time(),
                "factor": float(factor),
                "old_confidence": float(conf),
                "new_confidence": float(new_conf),
                "reason": str(reason or "centrality_boost"),
            }
        )
        ev_new["dmn_consolidation"] = notes[-32:]
        self.upsert(subject, predicate, obj, confidence=new_conf, evidence=ev_new)
        return obj, float(conf), float(new_conf)

    def overlapping_subject_pairs(self, *, min_shared: int = 2) -> list[dict]:
        """Pairs of distinct subjects sharing ``>= min_shared`` (predicate, object) tuples.

        Used by the DMN's separation phase to flag entities the substrate
        cannot discriminate from observation alone — high overlap means
        Fristonian ambiguity is high.
        """

        max_bucket = 256
        threshold = max(1, int(min_shared))
        bucket: dict[tuple[str, str], list[str]] = {}
        with self._sqlite_lock:
            rows = self._ensure_conn().execute(
                "SELECT subject, predicate, object FROM semantic_memory WHERE namespace=?",
                (self.namespace,),
            ).fetchall()
        per_subject: dict[str, set[tuple[str, str]]] = {}
        for subj, pred, obj in rows:
            subj = str(subj)
            key = (str(pred), str(obj))
            bucket.setdefault(key, []).append(subj)
            per_subject.setdefault(subj, set()).add(key)
        pair_overlap: dict[tuple[str, str], set[tuple[str, str]]] = {}
        for key, subjects in bucket.items():
            if len(subjects) < 2:
                continue
            unique = sorted(set(subjects))
            if len(unique) > max_bucket:
                logger.info(
                    "SymbolicMemory.overlapping_subject_pairs: capped bucket key=%r n_unique=%d max=%d",
                    key,
                    len(unique),
                    max_bucket,
                )
                unique = unique[:max_bucket]
            for i in range(len(unique)):
                for j in range(i + 1, len(unique)):
                    pair_overlap.setdefault((unique[i], unique[j]), set()).add(key)
        out: list[dict] = []
        for (a, b), shared in pair_overlap.items():
            if len(shared) < threshold:
                continue
            size_a = len(per_subject.get(a, set()))
            size_b = len(per_subject.get(b, set()))
            denom = max(1, min(size_a, size_b))
            ratio = len(shared) / float(denom)
            out.append(
                {
                    "subject_a": a,
                    "subject_b": b,
                    "shared": sorted(shared),
                    "shared_count": len(shared),
                    "size_a": size_a,
                    "size_b": size_b,
                    "overlap_ratio": float(ratio),
                }
            )
        out.sort(
            key=lambda d: (
                -d["overlap_ratio"],
                -d["shared_count"],
                d["subject_a"],
                d["subject_b"],
            )
        )
        return out
