"""Readiness levels for runtime faculties."""

from __future__ import annotations

from enum import Enum


class Readiness(str, Enum):
    """How much evidence supports a subsystem as currently wired."""

    TOY = "toy"
    PROTOTYPE = "prototype"
    VALIDATED = "validated"
    PRODUCTION = "production"
    EXPERIMENTAL = "experimental"


__all__ = ["Readiness"]
