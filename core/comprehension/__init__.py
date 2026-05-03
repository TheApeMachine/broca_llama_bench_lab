"""Comprehension — turns perceptual signals into a CognitiveFrame.

The pipeline:

* :class:`IntentGate` (already in :mod:`core.cognition.intent_gate`) decides
  whether an utterance is even storable.
* :class:`EncoderRelationExtractor` (already in
  :mod:`core.cognition.encoder_relation_extractor`) parses storable
  utterances into a :class:`ParsedClaim`.
* :class:`MemoryQueryParser` resolves questions into a :class:`ParsedQuery`
  against the substrate's known subjects.
* :class:`TextRelevance` scores a candidate frame's descriptor tokens
  against the user's utterance for routing.
* :class:`ClaimPredictionGap` runs the host's lexical surprise gap on a
  contradicting claim so consolidation can attenuate trust.
* :class:`SCMTargetPicker` discovers ``(treatment, outcome)`` from a SCM's
  endogenous labels.
* :class:`CognitiveRouter` ties everything together: every faculty receives
  the utterance and emits a scored :class:`FacultyCandidate`; the highest-
  precision candidate above the relevance floor wins, otherwise the
  workspace receives an ``unknown`` frame with the candidate trace.

Public surface: the classes named in :data:`__all__`.
"""

from __future__ import annotations

from .claim_prediction_gap import ClaimPredictionGap
from .deferred_relation_ingest import DeferredRelationIngest
from .memory_query_parser import MemoryQueryParser
from .router import CognitiveRouter
from .scm_target_picker import SCMTargetPicker
from .text_relevance import TextRelevance
from .tokens import LexicalTokens

__all__ = [
    "ClaimPredictionGap",
    "CognitiveRouter",
    "DeferredRelationIngest",
    "LexicalTokens",
    "MemoryQueryParser",
    "SCMTargetPicker",
    "TextRelevance",
]
