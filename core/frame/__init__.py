"""Frame — the substrate's shared schema package.

Holds the value types every concern exchanges (CognitiveFrame, ParsedClaim,
ParsedQuery, FacultyCandidate) and the projection machinery that turns text
and hypervectors into the fixed-width continuous vectors the grafts inject
into the frozen LLM.

This package has no verb of its own. It is the data layer below every other
concern; nothing inside it depends on anything in the core package outside
this directory.
"""

from __future__ import annotations

from .cognitive_frame import CognitiveFrame
from .dimensions import FrameDimensions
from .embedding_projector import EmbeddingProjector
from .faculty_candidate import FacultyCandidate
from .frame_packer import FramePacker
from .hypervector_projector import HypervectorProjector
from .numeric_tail import NumericTail
from .parsed_claim import ParsedClaim
from .parsed_query import ParsedQuery
from .subword_projector import SubwordProjector
from .text_encoder import TextEncoder

__all__ = [
    "CognitiveFrame",
    "EmbeddingProjector",
    "FacultyCandidate",
    "FrameDimensions",
    "FramePacker",
    "HypervectorProjector",
    "NumericTail",
    "ParsedClaim",
    "ParsedQuery",
    "SubwordProjector",
    "TextEncoder",
]
