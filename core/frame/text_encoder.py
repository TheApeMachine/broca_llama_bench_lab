"""Protocol for text → sketch encoders.

Anything that maps a text string to a fixed-width sketch tensor satisfies
``TextEncoder``. The substrate uses two implementations:

* :class:`core.frame.subword_projector.SubwordProjector` — pure lexical
  (frozen, deterministic, model-free).
* :class:`core.frame.embedding_projector.EmbeddingProjector` — pooled
  frozen-tokenizer embeddings projected through a fixed random matrix.

Both return ``[SKETCH_DIM]`` unit-norm float32 tensors. Callers must not
inspect which implementation produced the vector; they only depend on the
contract.
"""

from __future__ import annotations

from typing import Protocol

import torch


class TextEncoder(Protocol):
    """Maps a UTF-8 string to a fixed-width sketch tensor."""

    def __call__(self, text: str) -> torch.Tensor: ...
