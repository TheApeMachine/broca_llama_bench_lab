"""Inputs for derived graft strength."""

from __future__ import annotations

from pydantic import BaseModel


class StrengthInputs(BaseModel):
    """Signals composed into one graft strength."""

    intent_actionability: float
    memory_confidence: float
    conformal_set_size: int
    affect_certainty: float
