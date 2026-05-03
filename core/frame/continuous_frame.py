"""Removed. This module was the pre-refactor monolith.

Use :mod:`core.frame` directly:

  from core.frame import (
      CognitiveFrame,
      EmbeddingProjector,
      FrameDimensions,
      FramePacker,
      HypervectorProjector,
      NumericTail,
      SubwordProjector,
      TextEncoder,
  )

The free helpers ``stable_sketch``, ``semantic_subword_sketch``,
``pack_cognitive_frame``, ``pack_broca_features``, ``numeric_tail``,
``sparse_project_hypervector``, and ``frozen_subword_projector_from_model``
have been replaced by methods on the classes above. Importing this module
raises so any caller not yet migrated fails loudly.
"""

from __future__ import annotations

raise ImportError(
    "core.frame.continuous_frame has been removed. Import the classes from "
    "core.frame directly (CognitiveFrame, FrameDimensions, FramePacker, "
    "SubwordProjector, EmbeddingProjector, HypervectorProjector, NumericTail)."
)
