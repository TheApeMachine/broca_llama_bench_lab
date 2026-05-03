"""Generation — runs the frozen LLM forward and produces tokens.

The host wrapper (:class:`LlamaBrocaHost`) and tokenizer surfaces still
live in :mod:`core.host` for now; this package owns the substrate-side
generation helpers — token batching, decoding, plan-forced generation —
that previously lived inline in the substrate monolith.

Public surface:

* :class:`TokenBatch` — pad-and-mask a list of id sequences for forward.
* :class:`TokenDecoder` — decode generated ids back to text.
* :class:`PlanForcedGenerator` — run the host step-by-step under a fixed
  lexical plan; the substrate's primary chat-side generation path.
"""

from __future__ import annotations

from .chat_decoder import ChatDecoder
from .decoder import PlanForcedGenerator, TokenBatch, TokenDecoder
from .decode_state import DecodeState

__all__ = [
    "ChatDecoder",
    "DecodeState",
    "PlanForcedGenerator",
    "TokenBatch",
    "TokenDecoder",
]
