"""Modern Continuous Hopfield Networks (Ramsauer et al., 2020).

A Modern Continuous Hopfield Network is the rigorous mathematical generalization
of attention: given a set of stored patterns ``X ∈ R^{N × d}`` and a query
``ξ ∈ R^d``, the update rule

    ξ_{t+1} = X^⊤ softmax(β X ξ_t)

converges in *one step* (for large β) to the stored pattern that the query is
closest to under cosine similarity. The fixed-point analysis shows exponential
storage capacity in ``d`` (Theorem 3 of Ramsauer et al.), and the energy
function

    E(ξ) = -lse(β, X ξ) + 0.5 ξ^⊤ ξ + const

is exactly the loss minimized by attention layers in transformers.

This module wraps the update as a graft-friendly retrieval module that drops
into the existing ``KVMemoryGraft`` slot. The temperature β is derived from the
geometry of the store (``β = √d / σ_keys`` per the paper), so callers do not
have to hand-tune retrieval sharpness as the substrate grows.
"""

from __future__ import annotations

import logging
import math
import threading
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def derived_inverse_temperature(keys: torch.Tensor) -> float:
    """β = √d / σ — the paper's recommendation for separability under noise.

    Falls back to ``√d`` (i.e., σ = 1) when the store is too small or too
    uniform to estimate a meaningful spread. Uses ``√512`` when there are no
    keys so the returned scale stays on the usual ``√d`` order of magnitude.
    """

    if keys.numel() == 0:
        default_dim = 512
        return math.sqrt(default_dim)
    d = float(keys.shape[-1])
    flat = keys.reshape(-1, keys.shape[-1])
    if flat.shape[0] < 2:
        return math.sqrt(d)
    sigma = float(flat.std(unbiased=False).clamp_min(1e-6).item())
    return math.sqrt(d) / sigma


def hopfield_update(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    beta: float | None = None,
    iterations: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One-shot (or iterated) Modern Continuous Hopfield retrieval.

    Returns ``(retrieved_value, attention_weights, energy)``.
    Rows of ``keys`` and the trailing dimension of ``query`` agree (affinity is
    ``keys @ query`` flattened to length ``keys.shape[-1]``).
    Rows of ``values`` are softmax-weighted and contracted into the working
    state, which is then reshaped to ``query``'s layout each iteration — so for
    typical vector queries ``values.shape[-1]`` must match ``query.shape[-1]``.
    With β large enough,
    the attention collapses onto a single pattern; with smaller β it returns a
    weighted mixture (which is what the substrate wants when more than one
    memory is genuinely relevant).
    """

    if keys.shape[0] != values.shape[0]:
        raise ValueError(
            f"keys and values disagree on N: {keys.shape[0]} vs {values.shape[0]}"
        )
    if keys.shape[-1] != query.shape[-1]:
        raise ValueError(
            f"keys and query disagree on d: {keys.shape[-1]} vs {query.shape[-1]}"
        )
    if beta is None:
        beta = derived_inverse_temperature(keys)
    b = float(beta)
    q = query
    weights = torch.zeros(keys.shape[0], device=q.device, dtype=q.dtype)
    for _ in range(max(1, int(iterations))):
        scores = b * (keys @ q.flatten().to(keys.dtype))
        weights = F.softmax(scores, dim=-1)
        q = (weights.to(values.dtype) @ values).reshape_as(query)
    # Lyapunov energy E(ξ) = -lse(β X ξ) + 0.5 ‖ξ‖² + 0.5 max‖x‖²
    lse = torch.logsumexp(b * (keys @ q.flatten().to(keys.dtype)), dim=-1) / max(
        b, 1e-12
    )
    max_key_norm_sq = 0.5 * float(keys.norm(dim=-1).pow(2).max().item())
    half_q_norm = 0.5 * float(
        (q.flatten().to(torch.float32) @ q.flatten().to(torch.float32)).item()
    )
    energy = half_q_norm - float(lse.item()) + max_key_norm_sq
    logger.debug(
        "hopfield_update: beta=%.4f iters=%d N=%d d=%d energy=%.4f weight_max=%.4f",
        b,
        int(iterations),
        int(keys.shape[0]),
        int(keys.shape[-1]),
        energy,
        float(weights.max().item()),
    )
    return q, weights, torch.tensor(energy, dtype=torch.float32)


class HopfieldAssociativeMemory:
    """Persistent associative memory with Hopfield-style retrieval.

    Stored as a pair of tensors so the substrate can serialize and reload the
    state across runs. Retrieval uses Modern Hopfield contraction
    (:func:`hopfield_update`), which mixes ``values`` rows in value space and
    reshapes back to ``query``; keep ``keys`` and ``query`` aligned on embedding
    width and ``values`` consistent with ``query`` for the chosen layout.
    Adds rows are appended (older rows aren't forgotten — that's the DMN's
    job); duplicate keys collapse on cosine cleanup at query time without
    distorting the energy basin.
    """

    def __init__(
        self,
        d_model: int,
        *,
        max_items: int = 65_536,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ):
        dm = int(d_model)
        mi = int(max_items)
        if dm <= 0:
            raise ValueError(f"d_model must be a positive integer, got {d_model}")
        if mi <= 0:
            raise ValueError(f"max_items must be a positive integer, got {max_items}")
        self.d_model = dm
        self.max_items = mi
        self.dtype = dtype
        self.device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        self._lock = threading.Lock()
        self._buf_keys = torch.zeros(
            self.max_items, self.d_model, dtype=dtype, device=self.device
        )
        self._buf_values = torch.zeros(
            self.max_items, self.d_model, dtype=dtype, device=self.device
        )
        self._meta_ring: list[dict] = [{} for _ in range(self.max_items)]
        self._write_pos = 0
        self._count = 0
        self.last_debug: dict = {}

    def __len__(self) -> int:
        with self._lock:
            return int(self._count)

    def _active_kv_unlocked(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Chronological keys/values; caller must hold ``_lock``."""

        if self._count == 0:
            z_k = torch.empty(0, self.d_model, dtype=self.dtype, device=self.device)
            z_v = torch.empty(0, self.d_model, dtype=self.dtype, device=self.device)
            return z_k, z_v
        if self._count < self.max_items:
            return self._buf_keys[: self._count], self._buf_values[: self._count]
        wp = self._write_pos
        keys = torch.cat([self._buf_keys[wp:], self._buf_keys[:wp]], dim=0)
        values = torch.cat([self._buf_values[wp:], self._buf_values[:wp]], dim=0)
        return keys, values

    def _active_metadata_unlocked(self) -> list[dict]:
        if self._count == 0:
            return []
        if self._count < self.max_items:
            return [dict(self._meta_ring[i]) for i in range(self._count)]
        wp = self._write_pos
        front = [dict(self._meta_ring[i]) for i in range(wp, self.max_items)]
        front.extend(dict(self._meta_ring[i]) for i in range(wp))
        return front

    @property
    def keys(self) -> torch.Tensor:
        with self._lock:
            k, _ = self._active_kv_unlocked()
            return k.clone() if k.numel() else k

    @property
    def values(self) -> torch.Tensor:
        with self._lock:
            _, v = self._active_kv_unlocked()
            return v.clone() if v.numel() else v

    @property
    def metadata(self) -> list[dict]:
        with self._lock:
            return self._active_metadata_unlocked()

    def remember(
        self, key: torch.Tensor, value: torch.Tensor, *, metadata: Optional[dict] = None
    ) -> None:
        k = key.detach().reshape(-1, self.d_model).to(self.device, self.dtype)
        v = value.detach().reshape(-1, self.d_model).to(self.device, self.dtype)
        if k.shape[0] != v.shape[0]:
            raise ValueError(f"key/value count mismatch: {k.shape[0]} vs {v.shape[0]}")
        b = int(k.shape[0])
        if b > self.max_items:
            k = k[-self.max_items :]
            v = v[-self.max_items :]
            b = int(k.shape[0])
        md = dict(metadata or {})
        with self._lock:
            start = self._write_pos
            end_excl = start + b
            if end_excl <= self.max_items:
                self._buf_keys[start:end_excl] = k
                self._buf_values[start:end_excl] = v
                for ii in range(b):
                    self._meta_ring[start + ii] = dict(md)
            else:
                first = self.max_items - start
                self._buf_keys[start:] = k[:first]
                self._buf_values[start:] = v[:first]
                rest = b - first
                self._buf_keys[:rest] = k[first:]
                self._buf_values[:rest] = v[first:]
                for ii in range(first):
                    self._meta_ring[start + ii] = dict(md)
                for ii in range(rest):
                    self._meta_ring[ii] = dict(md)
            self._write_pos = (start + b) % self.max_items
            self._count = min(self.max_items, self._count + b)
            total_rows = self._count
        logger.debug(
            "HopfieldAssociativeMemory.remember: rows_added=%d total=%d d=%d",
            b,
            total_rows,
            self.d_model,
        )

    def retrieve(
        self, query: torch.Tensor, *, beta: float | None = None, iterations: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with self._lock:
            if self._count == 0:
                zeros = torch.zeros_like(query)
                return zeros, torch.zeros(0, dtype=query.dtype, device=query.device)
            keys, values = self._active_kv_unlocked()
            keys_q = keys.clone()
            values_q = values.clone()
            k_dev = keys_q.device
            k_dtype = keys_q.dtype
            n_pat = int(self._count)

        retrieved, weights, energy = hopfield_update(
            query.to(k_dev, k_dtype),
            keys_q,
            values_q,
            beta=beta,
            iterations=iterations,
        )
        with self._lock:
            self.last_debug = {
                "weight_max": float(weights.max().item()) if weights.numel() else 0.0,
                "energy": float(energy.item()),
                "n": n_pat,
            }
        return retrieved.to(query.dtype), weights
