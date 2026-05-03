"""Encoder-backed relation extractor that respects the intent gate.

The substrate's only relation extractor. The previous LLM-driven
``LLMRelationExtractor`` was deleted because it happily turned imperatives
like "Tell me a joke" into the triple ``(me, tell, joke)`` and shoved them
into semantic memory. The pipeline is now:

  1. :class:`IntentGate` decides if the utterance is even storable. If not
     — request, command, greeting, feedback, question — the extractor
     immediately returns ``None``. The router will fall through to other
     faculties (active inference, causal effect) without inventing a fact.
  2. For storable utterances, :class:`ExtractionEncoder` produces zero or more
     ``ExtractedRelation`` triples. The highest-confidence triple becomes a
     :class:`ParsedClaim`; ties are broken by score, then by the order
     GLiNER returned them.

The extractor never falls back on regex or string-splitting. If GLiNER
cannot find a relation in a sentence the user gave us, that is a true
*absence* of a claim and the substrate should not pretend otherwise.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from ..encoders.extraction import ExtractionEncoder, ExtractedRelation
from ..workspace import WorkspacePublisher
from .intent_gate import IntentGate, UtteranceIntent

logger = logging.getLogger(__name__)


class EncoderRelationExtractor:
    """Implements the :class:`RelationExtractor` protocol via GLiNER extraction.

    The class is intentionally small: composition over inheritance, no host
    LLM, no caching beyond what GLiNER does internally. Construction takes a
    pre-built :class:`IntentGate` and :class:`ExtractionEncoder` because both
    are shared with other substrate paths (affect, comprehend()).
    """

    def __init__(self, *, intent_gate: IntentGate, extraction: ExtractionEncoder):
        self._intent_gate = intent_gate
        self._extraction = extraction

    def extract_claim(
        self,
        utterance: str,
        toks: Sequence[str],
        *,
        utterance_intent: UtteranceIntent | None = None,
    ) -> Any:
        from .substrate import ParsedClaim

        text = (utterance or "").strip()
        if not text:
            WorkspacePublisher.emit(
                "cog.relation_extract",
                {"outcome": "empty", "utterance": ""},
            )
            return None
        intent = utterance_intent if utterance_intent is not None else self._intent_gate.classify(text)
        if not intent.allows_storage:
            logger.debug(
                "EncoderRelationExtractor: gated out utterance=%r label=%s conf=%.3f",
                text[:160],
                intent.label,
                intent.confidence,
            )
            WorkspacePublisher.emit(
                "cog.relation_extract",
                {
                    "outcome": "gated_out",
                    "utterance": text[:120],
                    "intent_label": intent.label,
                    "intent_confidence": intent.confidence,
                },
            )
            return None
        relations = self._extraction.extract_relations(text)
        if not relations:
            logger.debug(
                "EncoderRelationExtractor: no relations utterance=%r intent=%s",
                text[:160],
                intent.label,
            )
            WorkspacePublisher.emit(
                "cog.relation_extract",
                {
                    "outcome": "no_relations",
                    "utterance": text[:120],
                    "intent_label": intent.label,
                    "intent_confidence": intent.confidence,
                },
            )
            return None
        best = self._select_best(relations)
        evidence = self._build_evidence(text, intent, relations, best, toks)
        confidence = self._claim_confidence(best, intent)
        WorkspacePublisher.emit(
            "cog.relation_extract",
            {
                "outcome": "extracted",
                "utterance": text[:120],
                "intent_label": intent.label,
                "intent_confidence": intent.confidence,
                "subject": best.subject.lower(),
                "predicate": best.predicate.lower(),
                "object": best.object.lower(),
                "extractor_confidence": float(best.confidence),
                "claim_confidence": confidence,
                "n_alternatives": max(0, len(relations) - 1),
            },
        )
        return ParsedClaim(
            subject=best.subject.lower(),
            predicate=best.predicate.lower(),
            obj=best.object.lower(),
            confidence=confidence,
            evidence=evidence,
        )

    @staticmethod
    def _select_best(relations: list[ExtractedRelation]) -> ExtractedRelation:
        return max(relations, key=lambda r: (float(r.confidence), -len(r.predicate)))

    @staticmethod
    def _claim_confidence(best: ExtractedRelation, intent: UtteranceIntent) -> float:
        # Compose extractor confidence with intent confidence — both must be
        # high for the claim to carry weight downstream.
        gate = max(0.0, min(1.0, float(intent.confidence)))
        ext = max(0.0, min(1.0, float(best.confidence)))
        return float(gate * ext)

    def _build_evidence(
        self,
        utterance: str,
        intent: UtteranceIntent,
        relations: list[ExtractedRelation],
        best: ExtractedRelation,
        toks: Sequence[str],
    ) -> dict[str, Any]:
        return {
            "parser": "encoder_relation_extractor",
            "predicate_surface": best.predicate,
            "source_words": list(toks),
            "utterance": utterance,
            "intent_label": intent.label,
            "intent_confidence": intent.confidence,
            "intent_scores": dict(intent.scores),
            "extractor_confidence": float(best.confidence),
            "subject_label": best.subject_label,
            "object_label": best.object_label,
            "alternative_relations": [
                {
                    "subject": r.subject,
                    "predicate": r.predicate,
                    "object": r.object,
                    "confidence": float(r.confidence),
                }
                for r in relations
                if r is not best
            ],
        }
