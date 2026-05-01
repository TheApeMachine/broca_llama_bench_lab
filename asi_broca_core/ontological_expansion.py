"""Hebbian orthogonalization: promoting frequent concepts to dedicated axes.

The substrate's default vector space is hashed (``stable_sketch``) — cheap,
collision-prone, but fine for an open vocabulary. As the user keeps talking
about the same concept (their pet, a project, a domain), that concept earns a
*dedicated* unit vector orthogonal to every other promoted concept, so the
substrate's downstream similarity / retrieval / VSA-binding all gain
precision for the things that matter most.

Algorithm:

1. The substrate tracks per-concept access counts; the DMN periodically
   promotes anything that crosses a frequency threshold.
2. Promotion is Gram–Schmidt: take the current sketch, subtract its
   projections onto previously promoted axes, normalize. The result is
   exactly orthogonal to every existing promoted vector.
3. The promoted vector replaces the hash sketch when retrieval returns it,
   so all downstream similarity work uses the dedicated axis automatically.

The store is persistable so the user's vocabulary keeps its sharpened axes
across runs.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class PromotedConcept:
    name: str
    axis: torch.Tensor
    promoted_at: float
    access_count: int
    base_sketch: torch.Tensor


def gram_schmidt_orthogonalize(target: torch.Tensor, basis: Sequence[torch.Tensor]) -> torch.Tensor:
    """Subtract every basis vector's projection from ``target`` and renormalize.

    Numerically stabilized by re-projecting once after the first pass (so a
    near-parallel basis vector that survived the first subtraction is still
    cancelled).
    """

    v = target.detach().to(torch.float32).flatten().clone()
    for b in basis:
        bb = b.detach().to(torch.float32).flatten()
        v = v - (v @ bb) / (bb @ bb).clamp_min(1e-12) * bb
    for b in basis:  # second pass for stability (modified Gram–Schmidt twice)
        bb = b.detach().to(torch.float32).flatten()
        v = v - (v @ bb) / (bb @ bb).clamp_min(1e-12) * bb
    norm = float(v.norm().item())
    if norm < 1e-9:
        # Original target lay in span(basis); produce a fresh orthogonal vector
        # by perturbing along a deterministic direction and re-orthogonalizing.
        rng = torch.Generator()
        rng.manual_seed(int(target.detach().to(torch.float32).abs().sum().item() * 1e6) % (2**31 - 1) or 1)
        perturb = torch.empty_like(v).normal_(0.0, 1.0, generator=rng)
        v = perturb
        for b in basis:
            bb = b.detach().to(torch.float32).flatten()
            v = v - (v @ bb) / (bb @ bb).clamp_min(1e-12) * bb
        norm = float(v.norm().item())
    return v / max(norm, 1e-12)


class OntologicalRegistry:
    """Hebbian-promotion store over the substrate's sketch vector space."""

    def __init__(self, *, dim: int, frequency_threshold: int = 8):
        self.dim = int(dim)
        self.frequency_threshold = int(frequency_threshold)
        self.access_counts: dict[str, int] = {}
        self.promoted: dict[str, PromotedConcept] = {}

    def __len__(self) -> int:
        return len(self.promoted)

    def observe(self, name: str) -> int:
        """Bump the access count for ``name`` and return the new count."""

        c = self.access_counts.get(name, 0) + 1
        self.access_counts[name] = c
        return c

    def is_promoted(self, name: str) -> bool:
        return name in self.promoted

    def basis(self) -> list[torch.Tensor]:
        return [c.axis for c in self.promoted.values()]

    def maybe_promote(self, name: str, base_sketch: torch.Tensor) -> PromotedConcept | None:
        if name in self.promoted:
            return self.promoted[name]
        if self.access_counts.get(name, 0) < self.frequency_threshold:
            return None
        sketch = base_sketch.detach().to(torch.float32).flatten()
        if sketch.numel() != self.dim:
            raise ValueError(f"sketch dim {sketch.numel()} disagrees with registry dim {self.dim}")
        axis = gram_schmidt_orthogonalize(sketch, self.basis())
        promoted = PromotedConcept(
            name=name,
            axis=axis,
            promoted_at=time.time(),
            access_count=self.access_counts[name],
            base_sketch=sketch.detach().clone(),
        )
        self.promoted[name] = promoted
        logger.info(
            "OntologicalRegistry.promote: name=%r access=%d total_promoted=%d residual_norm=%.4f",
            name,
            self.access_counts[name],
            len(self.promoted),
            float(axis.norm().item()),
        )
        return promoted

    def vector_for(self, name: str, base_sketch: torch.Tensor) -> torch.Tensor:
        """Return the dedicated axis if promoted; otherwise the base sketch."""

        promoted = self.promoted.get(name)
        if promoted is None:
            return F.normalize(base_sketch.detach().to(torch.float32).flatten(), dim=0)
        return promoted.axis

    def cosine_to_basis(self, query: torch.Tensor) -> dict[str, float]:
        """Inspectable diagnostic: cosine of ``query`` against every promoted axis."""

        q = F.normalize(query.detach().to(torch.float32).flatten(), dim=0)
        out: dict[str, float] = {}
        for name, c in self.promoted.items():
            out[name] = float((q @ c.axis).item())
        return out


class PersistentOntologicalRegistry:
    """SQLite persistence wrapper for an ``OntologicalRegistry``."""

    def __init__(self, path: str | Path, *, namespace: str = "main"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.namespace = namespace
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path)
        con.execute("PRAGMA journal_mode=WAL")
        return con

    def _init_schema(self) -> None:
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS ontological_registry (
                    namespace TEXT NOT NULL,
                    name TEXT NOT NULL,
                    axis_json TEXT NOT NULL,
                    base_sketch_json TEXT NOT NULL,
                    promoted_at REAL NOT NULL,
                    access_count INTEGER NOT NULL,
                    PRIMARY KEY(namespace, name)
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS ontological_counts (
                    namespace TEXT NOT NULL,
                    name TEXT NOT NULL,
                    count INTEGER NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY(namespace, name)
                )
                """
            )

    def save(self, registry: OntologicalRegistry) -> None:
        now = time.time()
        with self._connect() as con:
            for name, c in registry.promoted.items():
                con.execute(
                    """
                    INSERT INTO ontological_registry(namespace, name, axis_json, base_sketch_json, promoted_at, access_count)
                    VALUES (?,?,?,?,?,?)
                    ON CONFLICT(namespace, name) DO UPDATE SET
                        axis_json=excluded.axis_json,
                        base_sketch_json=excluded.base_sketch_json,
                        access_count=excluded.access_count
                    """,
                    (self.namespace, name, json.dumps(c.axis.tolist()), json.dumps(c.base_sketch.tolist()), float(c.promoted_at), int(c.access_count)),
                )
            for name, count in registry.access_counts.items():
                con.execute(
                    """
                    INSERT INTO ontological_counts(namespace, name, count, updated_at)
                    VALUES (?,?,?,?)
                    ON CONFLICT(namespace, name) DO UPDATE SET count=excluded.count, updated_at=excluded.updated_at
                    """,
                    (self.namespace, name, int(count), now),
                )

    def load(self, *, dim: int, frequency_threshold: int = 8) -> OntologicalRegistry:
        registry = OntologicalRegistry(dim=dim, frequency_threshold=frequency_threshold)
        with self._connect() as con:
            counts = con.execute(
                "SELECT name, count FROM ontological_counts WHERE namespace=?",
                (self.namespace,),
            ).fetchall()
            promoted_rows = con.execute(
                "SELECT name, axis_json, base_sketch_json, promoted_at, access_count FROM ontological_registry WHERE namespace=?",
                (self.namespace,),
            ).fetchall()
        for name, count in counts:
            registry.access_counts[str(name)] = int(count)
        for name, axis_json, base_json, promoted_at, access_count in promoted_rows:
            axis = torch.tensor(json.loads(axis_json), dtype=torch.float32)
            base = torch.tensor(json.loads(base_json), dtype=torch.float32)
            registry.promoted[str(name)] = PromotedConcept(
                name=str(name),
                axis=axis,
                promoted_at=float(promoted_at),
                access_count=int(access_count),
                base_sketch=base,
            )
        return registry
