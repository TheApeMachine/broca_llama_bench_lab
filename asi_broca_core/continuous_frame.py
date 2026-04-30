"""Stable sketch vectors for cognitive frames (ontology-flexible bottleneck).

Instead of fixed one-hot slots per intent/entity/value (which cannot represent new
symbols), we map arbitrary UTF-8 strings to a fixed-dimensional sparse sketch via
deterministic hashing. Numeric faculty scalars are appended unchanged.

Any new entity ("einstein") or intent label occupies overlapping sketch mass like
feature hashing / Bloom-style fingerprints; ``TrainableBrocaGraft`` learns to map
this continuous bag into ``d_model``.
"""

from __future__ import annotations

import hashlib
from typing import Any

import torch

SKETCH_DIM = 128
SKETCH_SEEDS = 8

NUMERIC_FEATURE_FIELDS = (
    "confidence",
    "p_do_positive",
    "p_do_negative",
    "ate",
    "policy_listen",
    "policy_open_left",
    "policy_open_right",
    "delta_ce",
    "bias",
)

NUMERIC_TAIL_LEN = len(NUMERIC_FEATURE_FIELDS)
COGNITIVE_FRAME_DIM = SKETCH_DIM * 3 + NUMERIC_TAIL_LEN


def stable_sketch(text: str, *, dim: int = SKETCH_DIM, n_seeds: int = SKETCH_SEEDS) -> torch.Tensor:
    """Deterministic sparse sketch for ``text`` (stable across processes)."""

    v = torch.zeros(dim, dtype=torch.float32)
    raw = text.strip().lower().encode("utf-8")
    if not raw:
        return v
    for seed in range(int(n_seeds)):
        digest = hashlib.sha256(raw + seed.to_bytes(4, "little")).digest()
        idx = int.from_bytes(digest[:8], "little") % dim
        v[idx] += 1.0 / float(n_seeds)
    return v


def numeric_tail(confidence: float, evidence: dict[str, Any] | None) -> torch.Tensor:
    ev = evidence or {}
    policy = ev.get("policy_posterior", {}) or {}
    return torch.tensor(
        [
            float(confidence),
            float(ev.get("p_do_positive", 0.0)),
            float(ev.get("p_do_negative", 0.0)),
            float(ev.get("ate", 0.0)),
            float(policy.get("listen", 0.0)),
            float(policy.get("open_left", 0.0)),
            float(policy.get("open_right", 0.0)),
            float(ev.get("delta_ce", 0.0)),
            1.0,
        ],
        dtype=torch.float32,
    )


def pack_cognitive_frame(intent: str, subject: str, answer: str, confidence: float, evidence: dict[str, Any] | None) -> torch.Tensor:
    """Concatenate sketch(intent), sketch(subject), sketch(answer), numeric tail."""

    parts = torch.cat(
        [
            stable_sketch(str(intent)),
            stable_sketch(str(subject)),
            stable_sketch(str(answer)),
            numeric_tail(confidence, evidence),
        ],
        dim=0,
    )
    return parts

