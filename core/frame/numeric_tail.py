"""Numeric scalar tail appended to every cognitive frame feature vector.

The tail is a small fixed-order vector of faculty-emitted scalars: the
substrate's confidence, the SCM's intervention readouts, the active-inference
policy posterior's three components, the predictive-coding cross-entropy
delta, and a constant bias term. Order is part of the wire format and lives
in :class:`FrameDimensions.NUMERIC_FEATURE_FIELDS`.
"""

from __future__ import annotations

from typing import Any

import torch

from .dimensions import FrameDimensions


class NumericTail:
    """Encodes a confidence + evidence pair into the fixed numeric tail."""

    @classmethod
    def encode(cls, confidence: float, evidence: dict[str, Any] | None) -> torch.Tensor:
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

    @classmethod
    def length(cls) -> int:
        return FrameDimensions.numeric_tail_len()
