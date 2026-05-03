from __future__ import annotations

from sqlalchemy import Column, Index, LargeBinary
from sqlmodel import Field, SQLModel


class ActivationMemory(SQLModel, table=True):
    """SQLite row for :class:`~core.memory.memory.SQLiteActivationMemory`."""

    __tablename__ = "activation_memory"
    __table_args__ = (Index("idx_activation_namespace_kind", "namespace", "kind"),)

    id: int | None = Field(default=None, primary_key=True)
    namespace: str
    kind: str
    dim: int
    key_blob: bytes = Field(sa_column=Column(LargeBinary, nullable=False))
    value_blob: bytes = Field(sa_column=Column(LargeBinary, nullable=False))
    metadata_json: str
    confidence: float
    access_count: int = Field(default=0)
    created_at: float
    updated_at: float


class ActivationAssociation(SQLModel, table=True):
    """Directed edge keyed by unordered pair `(lo, hi)` with ``lo < hi``.

    Mirrors the legacy composite primary key and covering indexes used by raw SQL.
    """

    __tablename__ = "activation_association"
    __table_args__ = (
        Index("idx_activation_assoc_lo", "lo"),
        Index("idx_activation_assoc_hi", "hi"),
    )

    lo: int = Field(primary_key=True)
    hi: int = Field(primary_key=True)
    weight: float
    updated_at: float
