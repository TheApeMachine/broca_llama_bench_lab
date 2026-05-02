"""Extraction organ: zero-shot NER, relation extraction, and classification.

Replaces the LLM-based LLMRelationExtractor and regex-based intent detection
with a dedicated 205M parameter encoder model (GLiNER2) that handles:
- Named Entity Recognition with arbitrary dynamic labels
- Relation extraction between entity pairs
- Zero-shot text classification (intent, topic, etc.)
- Template-based structured extraction

Brain analogy: Wernicke's area — language comprehension and semantic parsing.

Model: fastino/gliner2-base-v1 (205M params, ~400MB, CPU-first design)
Fallback: urchade/gliner_medium-v2.1 (209M) if GLiNER2 unavailable
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from .base import BaseOrgan, OrganOutput

logger = logging.getLogger(__name__)

# Default model hierarchy (try in order)
_GLINER2_MODEL = "fastino/gliner2-base-v1"
_GLINER_FALLBACK = "urchade/gliner_medium-v2.1"

# Default entity labels for general-purpose extraction
DEFAULT_ENTITY_LABELS = [
    "person", "organization", "location", "concept",
    "event", "quantity", "time", "relationship",
]


@dataclass
class ExtractedEntity:
    """A named entity extracted from text."""
    text: str
    label: str
    score: float
    start: int = 0
    end: int = 0


@dataclass
class ExtractedRelation:
    """A relation between two entities."""
    subject: str
    predicate: str
    object: str
    confidence: float = 0.0
    subject_label: str = ""
    object_label: str = ""


@dataclass
class ExtractionResult:
    """Complete extraction output from the organ."""
    entities: list[ExtractedEntity] = field(default_factory=list)
    relations: list[ExtractedRelation] = field(default_factory=list)
    classifications: dict[str, list[tuple[str, float]]] = field(default_factory=dict)
    raw_text: str = ""


class ExtractionOrgan(BaseOrgan):
    """Frozen GLiNER2/GLiNER model for structured information extraction.

    Replaces LLM few-shot prompting for relation extraction with a dedicated
    encoder that runs in <10ms per utterance on CPU.

    Usage:
        organ = ExtractionOrgan()
        organ.load()

        # NER
        result = organ.extract_entities("Ada lives in Rome", labels=["person", "location"])

        # Relations (subject, predicate, object triples)
        result = organ.extract_relations("Ada lives in Rome")

        # Classification
        result = organ.classify("where is Ada?", labels=["question", "statement", "request"])
    """

    def __init__(
        self,
        *,
        model_id: str | None = None,
        device: str | None = None,
        entity_threshold: float = 0.3,
        relation_threshold: float = 0.3,
    ):
        # GLiNER output dim is contextual (span-based), not a fixed embedding
        # We report 0 since the organ returns structured data, not vectors
        super().__init__(
            name="extraction",
            model_id=model_id or _GLINER2_MODEL,
            output_dim=0,
            device=device,
        )
        self._entity_threshold = entity_threshold
        self._relation_threshold = relation_threshold
        self._is_gliner2 = False
        self._is_gliner1 = False

    def _load_model(self) -> None:
        """Try GLiNER2 first, fall back to GLiNER v2.1."""
        # Try GLiNER2
        try:
            from gliner import GLiNER
            self._model = GLiNER.from_pretrained(self._model_id)
            if hasattr(self._model, "to"):
                self._model = self._model.to(self.device)
            self._is_gliner2 = "gliner2" in self._model_id.lower()
            self._is_gliner1 = not self._is_gliner2
            logger.info("ExtractionOrgan: loaded %s", self._model_id)
            return
        except ImportError:
            logger.debug("gliner package not available")
        except Exception as exc:
            logger.warning("Failed to load %s: %s", self._model_id, exc)

        # Fallback to GLiNER v2.1
        if self._model_id != _GLINER_FALLBACK:
            try:
                from gliner import GLiNER
                self._model = GLiNER.from_pretrained(_GLINER_FALLBACK)
                if hasattr(self._model, "to"):
                    self._model = self._model.to(self.device)
                self._model_id = _GLINER_FALLBACK
                self._is_gliner1 = True
                logger.info("ExtractionOrgan: loaded fallback %s", _GLINER_FALLBACK)
                return
            except Exception as exc:
                logger.warning("Fallback GLiNER also failed: %s", exc)

        raise ImportError(
            "ExtractionOrgan requires the `gliner` package. "
            "Install with: pip install gliner"
        )

    def extract_entities(
        self,
        text: str,
        *,
        labels: Sequence[str] | None = None,
        threshold: float | None = None,
    ) -> list[ExtractedEntity]:
        """Extract named entities with dynamic labels.

        Args:
            text: Input text to extract from.
            labels: Entity type labels (e.g. ["person", "location", "concept"]).
                Must be lowercase for GLiNER.
            threshold: Minimum confidence score (default: self._entity_threshold).

        Returns:
            List of ExtractedEntity with text spans, labels, and scores.
        """
        self._ensure_loaded()
        start = time.time()

        labels = [l.lower() for l in (labels or DEFAULT_ENTITY_LABELS)]
        thresh = threshold if threshold is not None else self._entity_threshold

        try:
            raw = self._model.predict_entities(text, labels, threshold=thresh)
            entities = [
                ExtractedEntity(
                    text=ent.get("text", ent.get("span", "")),
                    label=ent.get("label", ""),
                    score=float(ent.get("score", 0.0)),
                    start=int(ent.get("start", 0)),
                    end=int(ent.get("end", 0)),
                )
                for ent in raw
            ]
        except Exception as exc:
            logger.warning("Entity extraction failed: %s", exc)
            entities = []

        self._record_call((time.time() - start) * 1000)
        return entities

    def extract_relations(
        self,
        text: str,
        *,
        entity_labels: Sequence[str] | None = None,
        relation_labels: Sequence[str] | None = None,
    ) -> list[ExtractedRelation]:
        """Extract (subject, predicate, object) triples from text.

        First extracts entities, then classifies relations between pairs.
        This replaces the LLM-based LLMRelationExtractor.

        Args:
            text: Input text.
            entity_labels: Labels for NER pass.
            relation_labels: Candidate relation types. If None, uses generic set.

        Returns:
            List of ExtractedRelation triples.
        """
        self._ensure_loaded()
        start = time.time()

        # Step 1: Extract entities
        entities = self.extract_entities(
            text,
            labels=entity_labels or ["person", "organization", "location", "concept", "thing"],
        )

        if len(entities) < 2:
            self._record_call((time.time() - start) * 1000)
            return []

        # Step 2: For each entity pair, infer relation
        # GLiNER2 can do this via schema; GLiNER1 needs the glirel package
        relations: list[ExtractedRelation] = []

        if self._is_gliner2 and hasattr(self._model, "predict_relations"):
            # GLiNER2 native relation extraction
            try:
                rel_labels = list(relation_labels or [
                    "is_in", "is_a", "has", "works_at", "created",
                    "related_to", "part_of", "causes", "located_in",
                ])
                raw_rels = self._model.predict_relations(text, rel_labels)
                for rel in raw_rels:
                    relations.append(ExtractedRelation(
                        subject=rel.get("head", {}).get("text", ""),
                        predicate=rel.get("label", ""),
                        object=rel.get("tail", {}).get("text", ""),
                        confidence=float(rel.get("score", 0.0)),
                        subject_label=rel.get("head", {}).get("label", ""),
                        object_label=rel.get("tail", {}).get("label", ""),
                    ))
            except Exception as exc:
                logger.debug("GLiNER2 relation extraction failed: %s", exc)
        
        # Fallback: synthesize relations from entity co-occurrence + proximity
        if not relations and len(entities) >= 2:
            relations = self._heuristic_relations(text, entities)

        self._record_call((time.time() - start) * 1000)
        return relations

    def classify(
        self,
        text: str,
        *,
        labels: Sequence[str],
        multi_label: bool = True,
        threshold: float = 0.3,
    ) -> list[tuple[str, float]]:
        """Zero-shot text classification with dynamic labels.

        Replaces regex-based intent/sentiment detection.

        Args:
            text: Input text.
            labels: Candidate classification labels.
            multi_label: Allow multiple labels above threshold.
            threshold: Minimum confidence.

        Returns:
            List of (label, score) tuples sorted by score descending.
        """
        self._ensure_loaded()
        start = time.time()

        results: list[tuple[str, float]] = []

        # GLiNER2 has built-in classification
        if self._is_gliner2 and hasattr(self._model, "classify"):
            try:
                raw = self._model.classify(text, list(labels))
                results = [(r["label"], float(r["score"])) for r in raw if float(r.get("score", 0)) >= threshold]
            except Exception as exc:
                logger.debug("GLiNER2 classify failed: %s", exc)

        # Fallback: use entity prediction as a proxy for classification
        # (predict the label as if it were an entity in the text)
        if not results:
            try:
                raw = self._model.predict_entities(text, [l.lower() for l in labels], threshold=threshold)
                seen = set()
                for ent in raw:
                    label = ent.get("label", "")
                    if label not in seen:
                        results.append((label, float(ent.get("score", 0.0))))
                        seen.add(label)
            except Exception as exc:
                logger.debug("Classification fallback also failed: %s", exc)

        results.sort(key=lambda x: x[1], reverse=True)
        if not multi_label and results:
            results = results[:1]

        self._record_call((time.time() - start) * 1000)
        return results

    def process(self, text: str, **kwargs: Any) -> OrganOutput:
        """Unified organ interface: extract entities + relations + classify intent."""
        self._ensure_loaded()
        start = time.time()

        result = ExtractionResult(raw_text=text)
        result.entities = self.extract_entities(text)
        result.relations = self.extract_relations(text)
        result.classifications["intent"] = self.classify(
            text, labels=["question", "statement", "request", "complaint", "praise"]
        )

        elapsed = (time.time() - start) * 1000
        return OrganOutput(
            features=None,  # Extraction organ outputs structured data, not vectors
            metadata={
                "entities": [{"text": e.text, "label": e.label, "score": e.score} for e in result.entities],
                "relations": [{"subject": r.subject, "predicate": r.predicate, "object": r.object, "confidence": r.confidence} for r in result.relations],
                "classifications": result.classifications,
            },
            confidence=max((e.score for e in result.entities), default=0.5),
            latency_ms=elapsed,
            organ_name=self._name,
        )

    def _heuristic_relations(self, text: str, entities: list[ExtractedEntity]) -> list[ExtractedRelation]:
        """Infer relations from entity proximity and basic sentence structure."""
        relations: list[ExtractedRelation] = []
        # Simple: take consecutive entity pairs and assign generic "related_to"
        for i in range(len(entities) - 1):
            e1, e2 = entities[i], entities[i + 1]
            # Extract text between the two entities
            between = text[e1.end:e2.start].strip().lower()
            # Try to find a verb phrase
            predicate = self._extract_predicate(between)
            if predicate:
                relations.append(ExtractedRelation(
                    subject=e1.text.lower(),
                    predicate=predicate,
                    object=e2.text.lower(),
                    confidence=0.5,
                    subject_label=e1.label,
                    object_label=e2.label,
                ))
        return relations

    @staticmethod
    def _extract_predicate(between: str) -> str:
        """Extract a predicate from text between two entities."""
        import re
        between = between.strip(" ,;:")
        # Common verb patterns
        patterns = [
            r"^(is|are|was|were)\s+(located\s+in|based\s+in|a|an|the\s+\w+\s+of)",
            r"^(is|are|was|were)\s+(\w+)",
            r"^(has|have|had)\s+(\w+)",
            r"^(\w+(?:ed|s|es|ing))\b",
        ]
        for pattern in patterns:
            m = re.match(pattern, between)
            if m:
                return m.group(0).strip()[:50]
        # If short enough, use the whole between-text as predicate
        if 1 < len(between.split()) <= 4:
            return between[:50]
        return ""
