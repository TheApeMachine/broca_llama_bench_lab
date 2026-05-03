"""Frozen-tokenizer embedding projector for sketch-space encoding.

Pools a host model's frozen input embeddings over the encoded tokens of the
input text, then projects through a fixed random matrix into ``SKETCH_DIM``
space. A lexical residual from :class:`SubwordProjector` is added so out-of-
vocabulary text never collapses to zero.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from .dimensions import FrameDimensions
from .subword_projector import SubwordProjector


class EmbeddingProjector:
    """Frozen-embedding-driven :class:`TextEncoder`.

    The projection matrix is sampled once at construction from a deterministic
    generator seeded by ``seed``; subsequent calls are pure functions of the
    embedding weight, the tokenizer, and the input string.

    By default ``embedding_weight`` is held by reference (``.detach()`` only).
    Pass ``clone_embedding=True`` if the caller may mutate the original
    in place.
    """

    def __init__(
        self,
        tokenizer: Any,
        embedding_weight: torch.Tensor,
        *,
        seed: int = 0,
        clone_embedding: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.dim = FrameDimensions.SKETCH_DIM
        w = embedding_weight.detach()
        self.embedding_weight = w.clone() if clone_embedding else w
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed))
        self.projection = torch.empty(
            (self.embedding_weight.shape[1], self.dim), dtype=torch.float32
        )
        self.projection.normal_(mean=0.0, std=1.0 / math.sqrt(float(self.dim)), generator=g)
        self._lexical = SubwordProjector(dim=self.dim)

    def __call__(self, text: str) -> torch.Tensor:
        ids = self._encode(text)
        if not ids:
            return self._lexical.encode(text)
        idx = torch.tensor(ids, dtype=torch.long, device=self.embedding_weight.device)
        pooled = self.embedding_weight.index_select(0, idx).float().mean(dim=0).cpu()
        z = pooled @ self.projection
        z = z + 0.15 * self._lexical.encode(text)
        norm = z.norm()
        if float(norm) > 1e-12:
            z = z / norm
        return z.to(dtype=torch.float32)

    def _encode(self, text: str) -> list[int]:
        try:
            ids = self.tokenizer.encode(str(text), add_special_tokens=False)
        except TypeError:
            ids = self.tokenizer.encode(str(text))
        n_vocab = self.embedding_weight.shape[0]
        return [int(i) for i in ids if 0 <= int(i) < n_vocab]

    @classmethod
    def from_host(
        cls,
        model: Any,
        tokenizer: Any,
        *,
        seed: int = 0,
        clone_embedding: bool = False,
    ) -> "EmbeddingProjector":
        """Construct from a host model's frozen input embeddings.

        Raises ``ValueError`` when the model exposes no usable input embedding;
        the substrate has no fallback path — every host the system supports
        must expose one.
        """

        host = getattr(model, "llm", model)
        get_input_embeddings = getattr(host, "get_input_embeddings", None)
        emb = get_input_embeddings() if callable(get_input_embeddings) else None
        if emb is None:
            inner = getattr(host, "model", None)
            emb = getattr(inner, "embed_tokens", None)
        weight = getattr(emb, "weight", None)
        if weight is None:
            raise ValueError(
                "EmbeddingProjector.from_host: host exposes no input embedding "
                "weight via get_input_embeddings() or model.embed_tokens"
            )
        return cls(tokenizer, weight, seed=seed, clone_embedding=clone_embedding)
