"""SQLite engine construction for file-backed SQLModel stores (WAL, busy timeout)."""

from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine.url import URL
from sqlmodel import SQLModel

logger = logging.getLogger(__name__)


class SqliteEngine:
    """Creates SQLAlchemy engines for on-disk SQLite databases used with SQLModel."""

    CONNECT_TIMEOUT_SECONDS: float = 5.0

    @classmethod
    def create(cls, path: Path | str, *, timeout_seconds: float) -> Engine:
        resolved = Path(path).resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        sqlite_url = URL.create(drivername="sqlite", database=str(resolved))

        engine = create_engine(sqlite_url, connect_args={"timeout": timeout_seconds})

        path_for_log = resolved

        @event.listens_for(engine, "connect")
        def _set_wal(raw_con, _) -> None:  # pragma: no cover — exercised only with real sqlite
            cursor = raw_con.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            row = cursor.fetchone()
            mode_raw = row[0] if row else None
            mode = str(mode_raw).lower() if mode_raw is not None else ""
            if mode != "wal":
                logger.warning(
                    "SQLite (%s): expected journal_mode wal, got %r",
                    path_for_log,
                    mode_raw,
                )

        return engine

    @classmethod
    def create_tables(cls, engine: Engine, *table_models: type[SQLModel]) -> None:
        if not table_models:
            raise ValueError("create_tables requires at least one SQLModel table class")
        SQLModel.metadata.create_all(engine, tables=[m.__table__ for m in table_models])
