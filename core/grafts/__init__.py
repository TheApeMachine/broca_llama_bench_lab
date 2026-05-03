"""Substrate-owned grafts that bridge :class:`CognitiveFrame` into the host.

Three classes turn a finalized cognitive frame into the per-step ``extra_state``
the frozen LLM consumes:

* :class:`LexicalPlanGraft` — biases the residual stream toward the planned
  next-token direction.
* :class:`TrainableFeatureGraft` — projects continuous frame features into
  the residual stream through a small trainable bridge.
* :class:`SubstrateLogitBiasGraft` — adjusts logits at the last position
  toward subwords the substrate wants the host to emit.

The generic :class:`BaseGraft`, KV-memory grafts, and the dynamic-graft
synthesizer still live in :mod:`core.grafting`; the substrate-owned three
were the ones the substrate monolith defined inline. They are now their own
package, one class per file, the way the rule-1 template wants.
"""

from __future__ import annotations

from .lexical_plan import LexicalPlanGraft
from .logit_bias import SubstrateLogitBiasGraft
from .trainable_feature import TrainableFeatureGraft

__all__ = [
    "LexicalPlanGraft",
    "SubstrateLogitBiasGraft",
    "TrainableFeatureGraft",
]
