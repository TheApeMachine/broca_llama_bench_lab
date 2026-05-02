"""SQLite persistence for multivariate Hawkes state (schema + read/write)."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HawkesLoadedSnapshot:
    """Decoded row from ``hawkes_state`` ready for domain validation."""

    beta: float
    baseline: float
    channels: list[str]
    mu: list[Any]
    alpha: list[list[Any]]
    states_raw: list[Any]


class HawkesRepository:
    """Low-level CRUD for the ``hawkes_state`` table."""

    def __init__(self, path: str | Path, *, namespace: str = "main") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.namespace = namespace

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path)
        con.execute("PRAGMA journal_mode=WAL")
        return con

    def init_schema(self) -> None:
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS hawkes_state (
                    namespace TEXT PRIMARY KEY,
                    beta REAL NOT NULL,
                    baseline REAL NOT NULL,
                    channels_json TEXT NOT NULL,
                    mu_json TEXT NOT NULL,
                    alpha_json TEXT NOT NULL,
                    states_json TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )

    def upsert_process(
        self,
        *,
        beta: float,
        baseline: float,
        channels: list[str],
        mu: list[float],
        alpha: list[list[float]],
        state_dicts: list[dict[str, Any]],
        updated_at: float | None = None,
    ) -> None:
        ts = float(time.time() if updated_at is None else updated_at)
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO hawkes_state(namespace, beta, baseline, channels_json, mu_json, alpha_json, states_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(namespace) DO UPDATE SET
                    beta=excluded.beta,
                    baseline=excluded.baseline,
                    channels_json=excluded.channels_json,
                    mu_json=excluded.mu_json,
                    alpha_json=excluded.alpha_json,
                    states_json=excluded.states_json,
                    updated_at=excluded.updated_at
                """,
                (
                    self.namespace,
                    float(beta),
                    float(baseline),
                    json.dumps(channels),
                    json.dumps(mu),
                    json.dumps(alpha),
                    json.dumps(state_dicts),
                    ts,
                ),
            )

    def fetch(self) -> HawkesLoadedSnapshot | None:
        with self._connect() as con:
            row = con.execute(
                "SELECT beta, baseline, channels_json, mu_json, alpha_json, states_json "
                "FROM hawkes_state WHERE namespace=?",
                (self.namespace,),
            ).fetchone()
        if row is None:
            return None
        channels = list(json.loads(row[2]))
        mu = list(json.loads(row[3]))
        alpha = [list(r) for r in json.loads(row[4])]
        states_raw = json.loads(row[5])
        return HawkesLoadedSnapshot(
            beta=float(row[0]),
            baseline=float(row[1]),
            channels=channels,
            mu=mu,
            alpha=alpha,
            states_raw=states_raw,
        )
