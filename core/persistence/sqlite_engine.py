"""Small SQLite connection factory used by file-backed stores."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SqliteEngine:
    """Creates configured stdlib SQLite connections.

    The project stores activation memory with raw SQLite instead of pulling in
    SQLAlchemy/SQLModel.  This compatibility class keeps the old construction
    name while returning a concrete ``sqlite3.Connection``.
    """

    CONNECT_TIMEOUT_SECONDS: float = 5.0

    @classmethod
    def create(cls, path: Path | str, *, timeout_seconds: float) -> sqlite3.Connection:
        resolved = Path(path).resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(resolved), timeout=float(timeout_seconds), check_same_thread=False)
        row = connection.execute("PRAGMA journal_mode=WAL").fetchone()
        mode_raw = row[0] if row else None
        mode = str(mode_raw).lower() if mode_raw is not None else ""

        if mode != "wal":
            logger.warning("SQLite (%s): expected journal_mode wal, got %r", resolved, mode_raw)

        connection.execute("PRAGMA busy_timeout=60000")

        return connection

    @classmethod
    def create_tables(cls, connection: sqlite3.Connection, *table_models: type[Any]) -> None:
        if not table_models:
            raise ValueError("create_tables requires at least one table model class")

        for model in table_models:
            statement = getattr(model, "create_statement", None)
            if not statement:
                raise TypeError(f"{model!r} does not define a create_statement")
            connection.execute(statement)

            for index_statement in getattr(model, "index_statements", ()): 
                connection.execute(index_statement)

        connection.commit()
