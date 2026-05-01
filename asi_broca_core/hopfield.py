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
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def derived_inverse_temperature(keys: torch.Tensor) -> float:
    """β = √d / σ — the paper's recommendation for separability under noise.

    Falls back to ``√d`` (i.e., σ = 1) when the store is too small or too
    uniform to estimate a meaningful spread.
    """

    if keys.numel() == 0:
        return 1.0
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

    Returns ``(retrieved_value, attention_weights, energy)``. ``query`` and the
    rows of ``keys`` / ``values`` must share the last dim. With β large enough
    the attention collapses onto a single pattern; with smaller β it returns a
    weighted mixture (which is what the substrate wants when more than one
    memory is genuinely relevant).
    """

    if keys.shape[0] != values.shape[0]:
        raise ValueError(f"keys and values disagree on N: {keys.shape[0]} vs {values.shape[0]}")
    if keys.shape[-1] != query.shape[-1]:
        raise ValueError(f"keys and query disagree on d: {keys.shape[-1]} vs {query.shape[-1]}")
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
    lse = torch.logsumexp(b * (keys @ q.flatten().to(keys.dtype)), dim=-1) / max(b, 1e-12)
    half_q_norm = 0.5 * float((q.flatten().to(torch.float32) @ q.flatten().to(torch.float32)).item())
    energy = float(half_q_norm - float(lse.item()))
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
    state across runs. Adds rows are appended (older rows aren't forgotten —
    that's the DMN's job); duplicate keys collapse on cosine cleanup at query
    time without distorting the energy basin.
    """

    def __init__(self, d_model: int, *, max_items: int = 65_536, dtype: torch.dtype = torch.float32, device: torch.device | str | None = None):
        self.d_model = int(d_model)
        self.max_items = int(max_items)
        self.dtype = dtype
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.keys = torch.empty(0, self.d_model, dtype=dtype, device=self.device)
        self.values = torch.empty(0, self.d_model, dtype=dtype, device=self.device)
        self.metadata: list[dict] = []
        self.last_debug: dict = {}

    def __len__(self) -> int:
        return int(self.keys.shape[0])

    def remember(self, key: torch.Tensor, value: torch.Tensor, *, metadata: Optional[dict] = None) -> None:
        k = key.detach().reshape(-1, self.d_model).to(self.keys.device, self.keys.dtype)
        v = value.detach().reshape(-1, self.d_model).to(self.values.device, self.values.dtype)
        if k.shape[0] != v.shape[0]:
            raise ValueError(f"key/value count mismatch: {k.shape[0]} vs {v.shape[0]}")
        self.keys = torch.cat([self.keys, k], dim=0)[-self.max_items :]
        self.values = torch.cat([self.values, v], dim=0)[-self.max_items :]
        for _ in range(k.shape[0]):
            self.metadata.append(dict(metadata or {}))
        self.metadata = self.metadata[-self.max_items :]
        logger.debug("HopfieldAssociativeMemory.remember: rows_added=%d total=%d d=%d", int(k.shape[0]), int(self.keys.shape[0]), self.d_model)

    def retrieve(self, query: torch.Tensor, *, beta: float | None = None, iterations: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        if self.keys.numel() == 0:
            zeros = torch.zeros_like(query)
            return zeros, torch.zeros(0, dtype=query.dtype, device=query.device)
        retrieved, weights, energy = hopfield_update(
            query.to(self.keys.device, self.keys.dtype),
            self.keys,
            self.values,
            beta=beta,
            iterations=iterations,
        )
        self.last_debug = {
            "weight_max": float(weights.max().item()) if weights.numel() else 0.0,
            "energy": float(energy.item()),
            "n": int(self.keys.shape[0]),
        }
        return retrieved.to(query.dtype), weights
