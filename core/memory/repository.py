"""Handles schema and inserts for persisted activation-memory rows."""

from pathlib import Path

from sqlmodel import Session

from core.persistence.sqlite_engine import SqliteEngine

from .model import ActivationMemory, ActivationAssociation


class Repository:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._engine = SqliteEngine.create(
            self.path,
            timeout_seconds=SqliteEngine.CONNECT_TIMEOUT_SECONDS,
        )
        SqliteEngine.create_tables(self._engine, ActivationMemory, ActivationAssociation)

    def insert_activation_memory(self, memory: ActivationMemory) -> None:
        with Session(self._engine) as session:
            session.add(memory)
            session.commit()
