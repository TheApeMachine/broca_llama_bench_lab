"""Substrate-owned grafts that bridge :class:`CognitiveFrame` into the host.

Construction for host attachment uses :class:`HostGraftsBuilder`; frame tensors
use :class:`FrameGraftProjection` in :mod:`core.grafts.feature`.
"""

from __future__ import annotations

from .builder import HostGraftsBuilder
from .chat_plan import ChatGraftPlan
from .concept_graft import SubstrateConceptGraft
from .feature import FrameGraftProjection
from .lexical_plan import LexicalPlanGraft
from .strength import DerivedStrength, StrengthInputs
from .trainable_feature import TrainableFeatureGraft

__all__ = [
    "DerivedStrength",
    "ChatGraftPlan",
    "FrameGraftProjection",
    "HostGraftsBuilder",
    "LexicalPlanGraft",
    "StrengthInputs",
    "SubstrateConceptGraft",
    "TrainableFeatureGraft",
]
