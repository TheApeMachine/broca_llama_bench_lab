"""Persistent affect traces and user/assistant alignment metrics."""

from __future__ import annotations

import json
import math
import sqlite3
import time
from pathlib import Path
from typing import Any

from ..encoders.affect import AffectState


class PersistentAffectTrace:
    """SQLite-backed affect history for both sides of a conversation."""

    def __init__(self, path: str | Path, *, namespace: str = "main") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.namespace = str(namespace)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self.path), timeout=30.0, check_same_thread=False)
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA busy_timeout=60000")
        return con

    def _init_schema(self) -> None:
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS affect_trace (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    namespace TEXT NOT NULL,
                    ts REAL NOT NULL,
                    role TEXT NOT NULL,
                    text_preview TEXT NOT NULL,
                    journal_id INTEGER,
                    response_to_id INTEGER,
                    dominant_emotion TEXT NOT NULL,
                    dominant_score REAL NOT NULL,
                    valence REAL NOT NULL,
                    arousal REAL NOT NULL,
                    entropy REAL NOT NULL,
                    certainty REAL NOT NULL,
                    preference_signal TEXT NOT NULL,
                    preference_strength REAL NOT NULL,
                    distribution_json TEXT NOT NULL,
                    alignment REAL,
                    alignment_json TEXT NOT NULL
                )
                """
            )
            con.execute("CREATE INDEX IF NOT EXISTS idx_affect_trace_ns_role ON affect_trace(namespace, role, id)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_affect_trace_response ON affect_trace(namespace, response_to_id)")

    def record(
        self,
        *,
        role: str,
        text: str,
        affect: AffectState,
        journal_id: int | None = None,
        response_to_id: int | None = None,
        alignment: dict[str, float | str | bool] | None = None,
    ) -> int:
        role_clean = str(role).strip().lower()
        if role_clean not in {"user", "assistant"}:
            raise ValueError(f"affect trace role must be 'user' or 'assistant', got {role!r}")

        alignment_payload = dict(alignment or {})
        alignment_score = alignment_payload.get("alignment")
        if alignment_score is not None:
            alignment_score = float(alignment_score)
            if not math.isfinite(alignment_score):
                raise ValueError("affect alignment must be finite")

        distribution = affect.distribution()
        payload = (
            self.namespace,
            time.time(),
            role_clean,
            str(text)[:512],
            None if journal_id is None else int(journal_id),
            None if response_to_id is None else int(response_to_id),
            str(affect.dominant_emotion),
            float(affect.dominant_score),
            float(affect.valence),
            float(affect.arousal),
            float(affect.entropy),
            float(affect.certainty),
            str(affect.preference_signal),
            float(affect.preference_strength),
            json.dumps(distribution, sort_keys=True),
            alignment_score,
            json.dumps(alignment_payload, sort_keys=True),
        )
        with self._connect() as con:
            cur = con.execute(
                """
                INSERT INTO affect_trace(
                    namespace, ts, role, text_preview, journal_id, response_to_id,
                    dominant_emotion, dominant_score, valence, arousal, entropy, certainty,
                    preference_signal, preference_strength, distribution_json,
                    alignment, alignment_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )
            return int(cur.lastrowid)

    def recent(self, limit: int = 8) -> list[dict[str, Any]]:
        with self._connect() as con:
            rows = con.execute(
                """
                SELECT id, ts, role, text_preview, journal_id, response_to_id,
                       dominant_emotion, dominant_score, valence, arousal,
                       entropy, certainty, preference_signal, preference_strength,
                       distribution_json, alignment, alignment_json
                FROM affect_trace
                WHERE namespace=?
                ORDER BY id DESC
                LIMIT ?
                """,
                (self.namespace, max(1, int(limit))),
            ).fetchall()
        return [self._row_to_dict(row) for row in reversed(rows)]

    def summary(self, *, recent_limit: int = 8) -> dict[str, Any]:
        with self._connect() as con:
            count = int(con.execute("SELECT COUNT(*) FROM affect_trace WHERE namespace=?", (self.namespace,)).fetchone()[0])
            user_count = int(con.execute("SELECT COUNT(*) FROM affect_trace WHERE namespace=? AND role='user'", (self.namespace,)).fetchone()[0])
            assistant_count = int(con.execute("SELECT COUNT(*) FROM affect_trace WHERE namespace=? AND role='assistant'", (self.namespace,)).fetchone()[0])
            paired = con.execute(
                """
                SELECT COUNT(*), AVG(alignment)
                FROM affect_trace
                WHERE namespace=? AND role='assistant' AND alignment IS NOT NULL
                """,
                (self.namespace,),
            ).fetchone()
            last = con.execute(
                """
                SELECT alignment_json
                FROM affect_trace
                WHERE namespace=? AND role='assistant' AND alignment IS NOT NULL
                ORDER BY id DESC
                LIMIT 1
                """,
                (self.namespace,),
            ).fetchone()
        return {
            "count": count,
            "user_count": user_count,
            "assistant_count": assistant_count,
            "paired_count": int(paired[0]),
            "mean_alignment": (None if paired[1] is None else float(paired[1])),
            "last_alignment": (None if last is None else json.loads(last[0])),
            "recent": self.recent(limit=recent_limit),
        }

    @classmethod
    def alignment(cls, user: AffectState, assistant: AffectState) -> dict[str, float | str | bool]:
        user_dist = user.distribution()
        assistant_dist = assistant.distribution()
        distribution_similarity = cls._cosine(user_dist, assistant_dist)
        valence_alignment = max(0.0, 1.0 - abs(float(user.valence) - float(assistant.valence)) / 2.0)
        arousal_alignment = max(0.0, 1.0 - abs(float(user.arousal) - float(assistant.arousal)))
        components = [distribution_similarity, valence_alignment, arousal_alignment]
        alignment = math.prod(components) ** (1.0 / len(components))
        return {
            "alignment": float(max(0.0, min(1.0, alignment))),
            "distribution_similarity": float(distribution_similarity),
            "valence_alignment": float(valence_alignment),
            "arousal_alignment": float(arousal_alignment),
            "valence_delta": float(assistant.valence - user.valence),
            "arousal_delta": float(assistant.arousal - user.arousal),
            "dominant_match": user.dominant_emotion == assistant.dominant_emotion,
            "user_dominant": str(user.dominant_emotion),
            "assistant_dominant": str(assistant.dominant_emotion),
        }

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        labels = sorted(set(a) | set(b))
        if not labels:
            return 0.0
        dot = sum(float(a.get(label, 0.0)) * float(b.get(label, 0.0)) for label in labels)
        na = math.sqrt(sum(float(a.get(label, 0.0)) ** 2 for label in labels))
        nb = math.sqrt(sum(float(b.get(label, 0.0)) ** 2 for label in labels))
        den = na * nb
        if den <= 0.0:
            return 0.0
        return max(0.0, min(1.0, dot / den))

    @staticmethod
    def _row_to_dict(row: sqlite3.Row | tuple[Any, ...]) -> dict[str, Any]:
        return {
            "id": int(row[0]),
            "ts": float(row[1]),
            "role": str(row[2]),
            "text_preview": str(row[3]),
            "journal_id": row[4],
            "response_to_id": row[5],
            "dominant_emotion": str(row[6]),
            "dominant_score": float(row[7]),
            "valence": float(row[8]),
            "arousal": float(row[9]),
            "entropy": float(row[10]),
            "certainty": float(row[11]),
            "preference_signal": str(row[12]),
            "preference_strength": float(row[13]),
            "distribution": json.loads(row[14]),
            "alignment": (None if row[15] is None else float(row[15])),
            "alignment_detail": json.loads(row[16]),
        }
