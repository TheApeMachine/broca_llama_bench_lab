"""Substrate-owned grafts that bridge :class:`CognitiveFrame` into the host.

Construction for host attachment uses :class:`HostGraftsBuilder`; frame tensors
use :class:`FrameGraftProjection` in :mod:`core.grafts.feature`.
"""

from __future__ import annotations

from .builder import HostGraftsBuilder
from .chat_plan import ChatGraftPlan
from .feature import FrameGraftProjection
from .lexical_plan import LexicalPlanGraft
from .logit_bias import SubstrateLogitBiasGraft
from .strength import DerivedStrength, StrengthInputs
from .trainable_feature import TrainableFeatureGraft
from .token_bias import TokenBias

__all__ = [
    "DerivedStrength",
    "ChatGraftPlan",
    "FrameGraftProjection",
    "HostGraftsBuilder",
    "LexicalPlanGraft",
    "StrengthInputs",
    "SubstrateLogitBiasGraft",
    "TokenBias",
    "TrainableFeatureGraft",
]
