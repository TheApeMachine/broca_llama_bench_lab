"""Token bias preview model."""

from __future__ import annotations

from pydantic import BaseModel


class TokenBias(BaseModel):
    """Human-readable view of one biased token."""

    token_id: int
    token: str
    bias: float
