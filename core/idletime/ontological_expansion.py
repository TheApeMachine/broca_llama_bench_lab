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
across runs. Persisted tensors use a single wire format: standard base64 of
contiguous little-endian fp32 bytes (SQLite column names ``axis_json`` /
``base_sketch_json`` are historical).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F

from .repository import (
    OntologicalRepository,
    PromotedPersistRow,
    decode_fp32_blob,
    encode_fp32_blob,
)

logger = logging.getLogger(__name__)


@dataclass
class PromotedConcept:
    name: str
    axis: torch.Tensor
    promoted_at: float
    access_count: int
    base_sketch: torch.Tensor


def gram_schmidt_orthogonalize(
    target: torch.Tensor, basis: Sequence[torch.Tensor]
) -> torch.Tensor:
    """Subtract every basis vector's projection from ``target`` and renormalize.

    Numerically stabilized by re-projecting once after the first pass (so a
    near-parallel basis vector that survived the first subtraction is still
    cancelled).
    """
    v = target.detach().to(torch.float32).flatten().clone()
    
    # Two passes for stability (modified Gram–Schmidt twice)
    for _ in range(2):
        for b in basis:
            bb = b.detach().to(torch.float32).flatten()
            v = v - (v @ bb) / (bb @ bb).clamp_min(1e-12) * bb

    norm = float(v.norm().item())
    
    if norm < 1e-9:
        # Original target lay in span(basis); produce a fresh orthogonal vector
        # by perturbing along deterministic directions and re-orthogonalizing.
        basis_dim_note = len(basis)
        d_full = int(v.numel())
    
        if len(basis) >= d_full:
            logger.warning(
                "gram_schmidt_orthogonalize: failed to escape span(basis) after perturb retries "
                "(norm=%s, len(basis)=%d)",
                norm,
                basis_dim_note,
            )
    
            raise ValueError(
                f"gram_schmidt_orthogonalize: cannot produce orthogonal direction "
                f"(norm={norm}, len(basis)={basis_dim_note})",
            )
    
        perturbed_ok = False
    
        base_seed = (
            int(target.detach().to(torch.float32).abs().sum().item() * 1e6)
            % (2**31 - 1)
            or 1
        )
    
        for attempt in range(5):
            rng = torch.Generator()
            rng.manual_seed(int((base_seed + attempt * 7919) % (2**31 - 1)) or 1)
            perturb = torch.empty_like(v).normal_(0.0, 1.0, generator=rng)
            v = perturb
    
            for b in basis:
                bb = b.detach().to(torch.float32).flatten()
                v = v - (v @ bb) / (bb @ bb).clamp_min(1e-12) * bb
    
            norm = float(v.norm().item())
    
            if norm >= 1e-9:
                perturbed_ok = True
                break
    
        if not perturbed_ok:
            logger.warning(
                "gram_schmidt_orthogonalize: failed to escape span(basis) after perturb retries "
                "(norm=%s, len(basis)=%d)",
                norm,
                basis_dim_note,
            )
    
            raise ValueError(
                f"gram_schmidt_orthogonalize: cannot produce orthogonal direction "
                f"(norm={norm}, len(basis)={basis_dim_note})",
            )
    
    return v / max(norm, 1e-12)


class OntologicalRegistry:
    """Hebbian-promotion store over the substrate's sketch vector space."""

    def __init__(self, *, dim: int, frequency_threshold: int = 8):
        self.dim = int(dim)
        self.frequency_threshold = int(frequency_threshold)
        self.access_counts: dict[str, int] = {}
        self.promoted: dict[str, PromotedConcept] = {}
        self._lock = threading.RLock()
        self._basis_cache: list[torch.Tensor] | None = None

    def __len__(self) -> int:
        with self._lock:
            return len(self.promoted)

    def observe(self, name: str) -> int:
        """Bump the access count for ``name`` and return the new count."""

        with self._lock:
            c = self.access_counts.get(name, 0) + 1
            self.access_counts[name] = c
            return c

    def is_promoted(self, name: str) -> bool:
        with self._lock:
            return name in self.promoted

    def basis(self) -> list[torch.Tensor]:
        with self._lock:
            if self._basis_cache is None:
                self._basis_cache = [c.axis.clone() for c in self.promoted.values()]
            return [t.clone() for t in self._basis_cache]

    def maybe_promote(
        self, name: str, base_sketch: torch.Tensor
    ) -> PromotedConcept | None:
        with self._lock:
            if name in self.promoted:
                return self.promoted[name]

            if self.access_counts.get(name, 0) < self.frequency_threshold:
                return None
            
            sketch = base_sketch.detach().to(torch.float32).flatten()
            
            if sketch.numel() != self.dim:
                raise ValueError(
                    f"sketch dim {sketch.numel()} disagrees with registry dim {self.dim}"
                )
            
            axis = gram_schmidt_orthogonalize(sketch, self.basis())
            
            promoted = PromotedConcept(
                name=name,
                axis=axis,
                promoted_at=time.time(),
                access_count=self.access_counts[name],
                base_sketch=sketch.detach().clone(),
            )
            
            self.promoted[name] = promoted
            self._basis_cache = None
            
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

        with self._lock:
            promoted = self.promoted.get(name)
            
            if promoted is None:
                return F.normalize(
                    base_sketch.detach().to(torch.float32).flatten(), dim=0
                )
            
            return promoted.axis

    def cosine_to_basis(self, query: torch.Tensor) -> dict[str, float]:
        """Inspectable diagnostic: cosine of ``query`` against every promoted axis."""
        q = F.normalize(query.detach().to(torch.float32).flatten(), dim=0)
        
        with self._lock:
            promoted_items = list(self.promoted.items())
        
        out: dict[str, float] = {}
        
        for name, c in promoted_items:
            out[name] = float((q @ c.axis).item())
        
        return out


def _promoted_axes_max_off_diagonal_dot(promoted: dict[str, PromotedConcept]) -> float:
    axes = [c.axis for c in promoted.values()]
    
    if len(axes) < 2:
        return 0.0
    A = torch.stack([F.normalize(a.flatten(), dim=0) for a in axes])
    G = A @ A.T
    K = G.shape[0]
    
    mask = ~torch.eye(K, dtype=torch.bool, device=G.device)
    
    return float(G.masked_select(mask).abs().max().item())


def _repair_loaded_promoted_axes(
    registry: OntologicalRegistry, *, tol: float = 1e-3
) -> None:
    max_od = _promoted_axes_max_off_diagonal_dot(registry.promoted)
    
    if max_od <= tol:
        return
    
    logger.warning(
        "OntologicalRegistry.load: promoted axes exceed orthogonality tolerance "
        "max_off_diagonal_dot=%.4f (tol=%.2e); applying sequential Gram–Schmidt repair.",
        max_od,
        tol,
    )
    
    names = sorted(
        registry.promoted.keys(), key=lambda n: registry.promoted[n].promoted_at
    )
    
    basis_acc: list[torch.Tensor] = []
    
    for name in names:
        pc = registry.promoted[name]
        new_axis = gram_schmidt_orthogonalize(pc.axis, basis_acc)
        basis_acc.append(new_axis)
    
        registry.promoted[name] = PromotedConcept(
            name=pc.name,
            axis=new_axis,
            promoted_at=pc.promoted_at,
            access_count=pc.access_count,
            base_sketch=pc.base_sketch,
        )
    
    registry._basis_cache = None


class PersistentOntologicalRegistry:
    """SQLite persistence wrapper for an ``OntologicalRegistry``."""

    def __init__(self, path: str | Path, *, namespace: str = "main"):
        self._repo = OntologicalRepository(path, namespace=namespace)
        self._repo.init_schema()

    @property
    def path(self) -> Path:
        return self._repo.path

    @property
    def namespace(self) -> str:
        return self._repo.namespace

    def close(self) -> None:
        self._repo.close()

    def save(self, registry: OntologicalRegistry) -> None:
        promoted = [
            PromotedPersistRow(
                name=name,
                axis_b64=encode_fp32_blob(c.axis),
                base_sketch_b64=encode_fp32_blob(c.base_sketch),
                promoted_at=float(c.promoted_at),
                access_count=int(c.access_count),
            )
            for name, c in registry.promoted.items()
        ]
        self._repo.replace_all(
            promoted=promoted, access_counts=dict(registry.access_counts)
        )

    def load(self, *, dim: int, frequency_threshold: int = 8) -> OntologicalRegistry:
        registry = OntologicalRegistry(dim=dim, frequency_threshold=frequency_threshold)

        for name, count in self._repo.fetch_counts():
            registry.access_counts[str(name)] = int(count)

        for row in self._repo.fetch_promoted():
            axis = decode_fp32_blob(
                row.axis_b64, expected_nelem=dim, field="axis"
            )
            base = decode_fp32_blob(
                row.base_sketch_b64, expected_nelem=dim, field="base_sketch"
            )
            if axis.shape[-1] != dim or base.shape[-1] != dim:
                raise ValueError(
                    f"load: tensor trailing dim mismatch for {row.name!r}: "
                    f"axis.shape={tuple(axis.shape)} base.shape={tuple(base.shape)} expected dim={dim}",
                )
            registry.promoted[row.name] = PromotedConcept(
                name=row.name,
                axis=axis,
                promoted_at=float(row.promoted_at),
                access_count=int(row.access_count),
                base_sketch=base,
            )

        _repair_loaded_promoted_axes(registry)

        return registry
