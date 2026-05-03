"""WorkspaceJournal — SQLite-backed episodic log of substrate turns.

Every utterance the substrate processes produces a :class:`CognitiveFrame`;
the journal records each frame paired with its raw utterance and timestamp.
The DMN replays the journal during REM-style consolidation, the chunking
compiler scans it for repeated motifs, and benchmarks read it to score
cross-turn coherence.

Storage shares the same SQLite file as :class:`SymbolicMemory`. When a
``shared_memory`` is supplied the journal piggybacks on its connection and
lock; when running standalone the journal opens its own short-lived
connections with WAL + busy-retry, since the SQLite file may also be opened
by other components.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path

from ..frame import CognitiveFrame
from .symbolic import SymbolicMemory


logger = logging.getLogger(__name__)


class WorkspaceJournal:
    """Episodic log of workspace frames paired with raw utterances."""

    def __init__(self, path: str | Path, *, shared_memory: SymbolicMemory | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._shared_memory = shared_memory
        if shared_memory is not None and Path(shared_memory.path).resolve() != self.path.resolve():
            raise ValueError(
                "WorkspaceJournal shared_memory must use the same database path as the journal"
            )
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=30.0, check_same_thread=False)
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA busy_timeout=5000")
        return con

    def _init_schema(self) -> None:
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS workspace_journal (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    utterance TEXT NOT NULL,
                    intent TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    evidence_json TEXT NOT NULL
                )
                """
            )

    def append(self, utterance: str, frame: CognitiveFrame, *, ts: float | None = None) -> int:
        now = float(ts) if ts is not None else time.time()
        payload = (
            now,
            utterance,
            frame.intent,
            frame.subject,
            frame.answer,
            float(frame.confidence),
            json.dumps(frame.evidence or {}, sort_keys=True),
        )

        def _insert_on(con: sqlite3.Connection) -> int:
            cur = con.execute(
                """
                INSERT INTO workspace_journal(ts, utterance, intent, subject, answer, confidence, evidence_json)
                VALUES (?,?,?,?,?,?,?)
                """,
                payload,
            )
            return int(cur.lastrowid)

        sm = self._shared_memory
        if sm is not None:
            with sm._sqlite_lock:
                return _insert_on(sm._ensure_conn())

        delay = 0.02
        last_exc: sqlite3.OperationalError | None = None
        for _ in range(30):
            try:
                with self._connect() as con:
                    return _insert_on(con)
            except sqlite3.OperationalError as exc:
                last_exc = exc
                msg = str(exc).lower()
                if "locked" not in msg and "busy" not in msg:
                    raise
                time.sleep(delay)
                delay = min(delay * 1.4, 0.35)
        assert last_exc is not None
        raise last_exc

    def fetch(self, episode_id: int) -> dict | None:
        with self._connect() as con:
            row = con.execute(
                """
                SELECT id, ts, utterance, intent, subject, answer, confidence, evidence_json
                FROM workspace_journal WHERE id=?
                """,
                (int(episode_id),),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def recent(self, limit: int) -> list[dict]:
        lim = max(1, int(limit))
        with self._connect() as con:
            rows = con.execute(
                """
                SELECT id, ts, utterance, intent, subject, answer, confidence, evidence_json
                FROM workspace_journal ORDER BY id DESC LIMIT ?
                """,
                (lim,),
            ).fetchall()
        return [self._row_to_dict(row) for row in reversed(rows)]

    def count(self) -> int:
        with self._connect() as con:
            row = con.execute("SELECT COUNT(*) FROM workspace_journal").fetchone()
        return int(row[0])

    @staticmethod
    def _row_to_dict(row: tuple) -> dict:
        return {
            "id": int(row[0]),
            "ts": float(row[1]),
            "utterance": str(row[2]),
            "intent": str(row[3]),
            "subject": str(row[4]),
            "answer": str(row[5]),
            "confidence": float(row[6]),
            "evidence": json.loads(row[7]),
        }
