"""Closed-form alignment matrices for latent communication between frozen organs.

Every cross-organ projection in the substrate is derived from the organs' own
pretrained weights via ridge regression — no training, no learned adapters.
The construction is the LatentMAS alignment matrix generalised to cross-model
where source and target share pretrained ancestry, plus a closed-form
projection from the substrate's algebraic working memory onto a target
organ's input embedding column space.

References:
- Zou et al., *Latent Collaboration in Multi-Agent Systems* (arXiv:2511.20639):
  the within-organ ``W_a = (W_outᵀ W_out + λI)⁻¹ W_outᵀ W_in`` derivation and
  Wasserstein optimality (Theorem A.1).
"""

from __future__ import annotations

from .alignment_registry import AlignmentRegistry
from .base import BaseAlignment
from .cross_model_alignment import CrossModelAlignment
from .ridge_alignment import RidgeAlignment
from .swm_to_input_projection import SWMToInputProjection

__all__ = [
    "AlignmentRegistry",
    "BaseAlignment",
    "CrossModelAlignment",
    "RidgeAlignment",
    "SWMToInputProjection",
]
