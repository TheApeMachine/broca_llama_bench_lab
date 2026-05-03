"""Substrate-owned grafts that bridge :class:`CognitiveFrame` into the host.

Construction for host attachment uses :class:`HostGraftsBuilder`; frame tensors
use :class:`FrameGraftProjection` in :mod:`core.grafts.feature`.
"""

from __future__ import annotations

from .builder import HostGraftsBuilder
from .feature import FrameGraftProjection
from .lexical_plan import LexicalPlanGraft
from .logit_bias import SubstrateLogitBiasGraft
from .trainable_feature import TrainableFeatureGraft

__all__ = [
    "FrameGraftProjection",
    "HostGraftsBuilder",
    "LexicalPlanGraft",
    "SubstrateLogitBiasGraft",
    "TrainableFeatureGraft",
]
