"""TextRelevance — score utterance/route descriptor / frame relevance.

The router needs three flavors of cosine-style scoring:

* :meth:`vector` — encode arbitrary text through the substrate's text
  encoder (or fall back to the lexical sketch when none is supplied).
* :meth:`route` — combine semantic cosine, token overlap, and a
  binary hit-gate into a normalized ``[0, 1]`` relevance score.
* :meth:`frame` — :meth:`route` over a :class:`CognitiveFrame`'s
  descriptor tokens.
"""

from __future__ import annotations

import logging
from typing import Sequence

import torch

from ..frame import CognitiveFrame, SubwordProjector, TextEncoder
from ..host.tokenizer import utterance_words
from .tokens import LexicalTokens


logger = logging.getLogger(__name__)
_SUBWORD = SubwordProjector()


class TextRelevance:
    """Stateless cosine + lexical-overlap scorer for the router."""

    @classmethod
    def vector(cls, text: str, text_encoder: TextEncoder | None) -> torch.Tensor:
        if text_encoder is None:
            return _SUBWORD.encode(text)
        try:
            v = text_encoder(text)
        except (RuntimeError, ValueError):
            logger.error(
                "TextRelevance.vector: text_encoder failed; falling back to subword sketch",
                exc_info=True,
            )
            v = _SUBWORD.encode(text)
        return v.detach().float().cpu().view(-1)

    @classmethod
    def cosine(cls, a: torch.Tensor, b: torch.Tensor) -> float:
        den = (a.norm() * b.norm()).clamp_min(1e-12)
        return float(torch.dot(a.view(-1), b.view(-1)) / den)

    @classmethod
    def route(
        cls,
        utterance: str,
        toks: Sequence[str],
        descriptors: Sequence[str],
        text_encoder: TextEncoder | None,
    ) -> float:
        """Continuous intent relevance plus lexical coverage from a route manifest."""

        descriptor_text = " ".join(descriptors)
        sem = max(
            0.0,
            cls.cosine(
                cls.vector(utterance, text_encoder),
                cls.vector(descriptor_text, text_encoder),
            ),
        )
        words = set(LexicalTokens.words(toks))
        desc_words = set(LexicalTokens.words(utterance_words(descriptor_text)))
        hits = words & desc_words
        overlap = len(hits) / max(1, min(len(words), len(desc_words)))
        hit_gate = 1.0 if hits else 0.0
        # Coefficients (0.50 + 0.35 + 0.25) sum above 1.0; normalize before
        # clamping so the score stays in [0, 1].
        combined = 0.50 * sem + 0.35 * overlap + 0.25 * hit_gate
        return max(0.0, min(1.0, combined / 1.10))

    @classmethod
    def frame(
        cls,
        utterance: str,
        toks: Sequence[str],
        frame: CognitiveFrame,
        text_encoder: TextEncoder | None,
    ) -> float:
        return cls.route(utterance, toks, frame.descriptor_tokens(), text_encoder)
