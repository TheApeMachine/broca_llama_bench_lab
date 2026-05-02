"""Extraction encoder: zero-shot NER, relation extraction, and identity claims.

Replaces the LLM-based ``LLMRelationExtractor`` with a GLiNER2 encoder for
grounded entities, structured facts, and first-person identity claims. Intent
classification is owned by the GLiClass semantic cascade; this module keeps
``classify_text`` only as a low-level model capability. There is no fallback
chain — if the model cannot be loaded or a method is missing, this module
raises so the flaw is exposed rather than papered over.

Model: ``fastino/gliner2-base-v1`` (205M params, ~400MB, CPU-first design),
loaded via the ``gliner2`` package — *not* ``gliner``. The two libraries
share a name family but expose different APIs; gliner2's surface is
``extract_entities(text, [labels])``, ``classify_text(text, schema)``, and
``extract_json(text, schema)``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..system.event_bus import get_default_bus
from .base import BaseEncoder, EncoderOutput

logger = logging.getLogger(__name__)


EXTRACTION_MODEL_ID = "fastino/gliner2-base-v1"

DEFAULT_ENTITY_LABELS: tuple[str, ...] = (
    "person", "organization", "location", "concept",
    "event", "quantity", "time", "thing",
)

DEFAULT_RELATION_LABELS: tuple[str, ...] = (
    "is_in", "is_a", "has", "works_at", "created",
    "related_to", "part_of", "causes", "located_in",
)

STRUCTURED_FACT_KEY = "fact"
STRUCTURED_FACT_FIELDS: tuple[str, ...] = (
    "subject::str::entity in subject role",
    "predicate::str::verb phrase linking subject and object",
    "object::str::entity in object role",
)
IDENTITY_CLAIM_KEY = "identity"
IDENTITY_CLAIM_FIELDS: tuple[str, ...] = (
    "speaker::str::person speaking",
    "name::str::name or title claimed by speaker",
)

_REQUIRED_GLINER2_METHODS: tuple[str, ...] = (
    "extract_entities",
    "classify_text",
    "extract_json",
)


@dataclass
class ExtractedEntity:
    text: str
    label: str
    score: float
    start: int = 0
    end: int = 0


@dataclass
class ExtractedRelation:
    subject: str
    predicate: str
    object: str
    confidence: float = 0.0
    subject_label: str = ""
    object_label: str = ""


@dataclass
class ExtractionResult:
    entities: list[ExtractedEntity] = field(default_factory=list)
    relations: list[ExtractedRelation] = field(default_factory=list)
    classifications: dict[str, list[tuple[str, float]]] = field(default_factory=dict)
    raw_text: str = ""


class ExtractionEncoder(BaseEncoder):
    """Frozen GLiNER2 model for structured information extraction."""

    def __init__(
        self,
        *,
        model_id: str | None = None,
        device: str | None = None,
        entity_threshold: float = 0.3,
        relation_threshold: float = 0.3,
    ):
        super().__init__(
            name="extraction",
            model_id=model_id or EXTRACTION_MODEL_ID,
            output_dim=0,
            device=device,
        )
        self._entity_threshold = float(entity_threshold)
        self._relation_threshold = float(relation_threshold)

    def _load_model(self) -> None:
        from gliner2 import GLiNER2

        self._model = GLiNER2.from_pretrained(self._model_id)
        if hasattr(self._model, "to"):
            self._model = self._model.to(self.device)
        for required in _REQUIRED_GLINER2_METHODS:
            if not callable(getattr(self._model, required, None)):
                raise RuntimeError(
                    f"ExtractionEncoder requires GLiNER2 with `{required}`; "
                    f"loaded model {self._model_id!r} does not expose it."
                )
        logger.info("ExtractionEncoder: loaded %s", self._model_id)

    def extract_entities(
        self,
        text: str,
        *,
        labels: Sequence[str] | None = None,
        threshold: float | None = None,
    ) -> list[ExtractedEntity]:
        """Run zero-shot NER and return ``ExtractedEntity`` mentions.

        gliner2's ``extract_entities`` returns ``{"entities": {label: [text, ...]}}``
        — a label → mentions map without per-mention scores or offsets. We
        translate that into the legacy list-of-mentions shape; callers that
        depended on per-mention ``score`` see ``1.0`` for every survivor of
        gliner2's internal threshold (``threshold`` here is informational —
        gliner2 applies its own).
        """

        self._ensure_loaded()
        start = time.time()
        _ = threshold
        ent_labels = [str(l).lower() for l in (labels or DEFAULT_ENTITY_LABELS)]
        raw = self._model.extract_entities(text, ent_labels)
        payload = raw.get("entities", raw) if isinstance(raw, dict) else {}
        entities: list[ExtractedEntity] = []
        if isinstance(payload, dict):
            for label, mentions in payload.items():
                if not isinstance(mentions, (list, tuple)):
                    continue
                for mention in mentions:
                    surface = str(mention)
                    span = self._locate_span(text, surface)
                    entities.append(
                        ExtractedEntity(
                            text=surface,
                            label=str(label),
                            score=1.0,
                            start=span[0],
                            end=span[1],
                        )
                    )
        latency = (time.time() - start) * 1000
        self._record_call(latency, method="extract_entities")
        self._publish(
            "encoder.extraction.entities",
            {
                "text": text[:120],
                "n_entities": len(entities),
                "labels": list(ent_labels),
                "entities": [(e.label, e.text) for e in entities[:8]],
                "latency_ms": latency,
            },
        )
        return entities

    @staticmethod
    def _unwrap_gliner_scalar(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, dict) and "text" in value:
            tx = value.get("text")
            return None if tx is None else str(tx)
        return str(value)

    @staticmethod
    def _predicate_gap_from_source(text_full: str, subj_clean: str, obj_clean: str) -> str:
        """Substring strictly between grounded subject/object spans in ``text_full``.

        Raises if the grounding is unusable — no silent Invented predicates.
        """

        tl = text_full.lower()
        sl = subj_clean.strip().lower()
        ol = obj_clean.strip().lower()
        if not sl or not ol:
            raise ValueError(
                "ExtractionEncoder._predicate_gap_from_source: empty subject or object span"
            )
        i = tl.find(sl)
        j = tl.find(ol)
        if i < 0 or j < 0:
            raise ValueError(
                f"ExtractionEncoder._predicate_gap_from_source: substring not located "
                f"subject={subj_clean!r} object={obj_clean!r}"
            )
        if j <= i + len(sl):
            raise ValueError(
                "ExtractionEncoder._predicate_gap_from_source: object does not follow subject span"
            )
        gap_raw = text_full[i + len(sl) : j].strip().strip(",").strip()
        gap_norm = gap_raw.strip().lower()
        if not gap_norm:
            raise ValueError(
                "ExtractionEncoder._predicate_gap_from_source: empty gap between grounded entities"
            )
        return gap_norm

    def extract_relations(
        self,
        text: str,
        *,
        entity_labels: Sequence[str] | None = None,
        relation_labels: Sequence[str] | None = None,
    ) -> list[ExtractedRelation]:
        """Extract subject-predicate-object triples via gliner2's ``extract_json`` structured ``fact``.

        ``fastino/gliner2-base-v1`` emits list-valued structures (see GLiNER2
        README); naming the bundle ``relations`` yielded empty payloads in
        practice, while ``fact`` returns grounded subject/object rows. Predicate
        may be absent in JSON-null form; then it is reconstructed only from the
        source text spans between grounded entities (still fully text-derived).

        ``relation_labels`` remains informational — downstream normalizes predicates.
        """

        self._ensure_loaded()
        start = time.time()
        _ = entity_labels, relation_labels
        identity_relations = self.extract_identity_relations(text)
        if identity_relations:
            latency = (time.time() - start) * 1000
            self._record_call(latency, method="extract_relations")
            self._publish(
                "encoder.extraction.relations",
                {
                    "text": text[:120],
                    "n_relations": len(identity_relations),
                    "relations": [
                        (r.subject, r.predicate, r.object) for r in identity_relations[:5]
                    ],
                    "latency_ms": latency,
                },
            )
            return identity_relations

        schema = {STRUCTURED_FACT_KEY: list(STRUCTURED_FACT_FIELDS)}
        raw = self._model.extract_json(text, schema)
        records: list[dict] = []
        if isinstance(raw, dict):
            primary = raw.get(STRUCTURED_FACT_KEY)
            if isinstance(primary, list):
                records.extend(r for r in primary if isinstance(r, dict))
            elif isinstance(primary, dict) and primary:
                records.append(primary)
            legacy = raw.get("relations")
            if isinstance(legacy, list):
                records.extend(r for r in legacy if isinstance(r, dict))

        relations: list[ExtractedRelation] = []
        for item in records:
            lower = {str(k).lower().replace("-", "").replace("_", ""): v for k, v in item.items()}
            rs = lower.get("subject")
            rp = lower.get("predicate")
            ro = lower.get("object")
            subj_clean = self._unwrap_gliner_scalar(rs)
            obj_clean = self._unwrap_gliner_scalar(ro)
            pred_raw = self._unwrap_gliner_scalar(rp)
            if subj_clean is None or obj_clean is None:
                continue
            subj = str(subj_clean).strip().lower()
            obj = str(obj_clean).strip().lower()
            if not subj or not obj:
                continue
            pred_part = None if pred_raw is None else str(pred_raw).strip().lower()
            if not pred_part or pred_part in {"none", "null"}:
                pred_part = self._predicate_gap_from_source(text, subj, obj)
            if not pred_part:
                continue
            relations.append(
                ExtractedRelation(
                    subject=subj,
                    predicate=pred_part,
                    object=obj,
                    confidence=1.0,
                )
            )

        latency = (time.time() - start) * 1000
        self._record_call(latency, method="extract_relations")
        self._publish(
            "encoder.extraction.relations",
            {
                "text": text[:120],
                "n_relations": len(relations),
                "relations": [
                    (r.subject, r.predicate, r.object) for r in relations[:5]
                ],
                "latency_ms": latency,
            },
        )
        return relations

    def extract_identity_relations(self, text: str) -> list[ExtractedRelation]:
        """Extract first-person identity claims with GLiNER2 structured JSON."""

        self._ensure_loaded()
        start = time.time()
        raw = self._model.extract_json(
            text,
            {IDENTITY_CLAIM_KEY: list(IDENTITY_CLAIM_FIELDS)},
        )
        relations = self._identity_relations_from_raw(raw)
        latency = (time.time() - start) * 1000
        self._record_call(latency, method="extract_identity_relations")
        self._publish(
            "encoder.extraction.identity_relations",
            {
                "text": text[:120],
                "n_relations": len(relations),
                "relations": [
                    (r.subject, r.predicate, r.object) for r in relations[:5]
                ],
                "latency_ms": latency,
            },
        )
        return relations

    def _identity_relations_from_raw(self, raw: Any) -> list[ExtractedRelation]:
        records: list[dict] = []
        if isinstance(raw, dict):
            primary = raw.get(IDENTITY_CLAIM_KEY)
            if isinstance(primary, list):
                malformed = [repr(r) for r in primary if not isinstance(r, dict)]
                if malformed:
                    logger.warning(
                        "ExtractionEncoder.identity: malformed identity records ignored: %s",
                        malformed[:3],
                    )
                records.extend(r for r in primary if isinstance(r, dict))
            elif isinstance(primary, dict) and primary:
                records.append(primary)
            elif primary is not None:
                logger.warning(
                    "ExtractionEncoder.identity: expected %r to be dict or list[dict], got %s",
                    IDENTITY_CLAIM_KEY,
                    type(primary).__name__,
                )
        else:
            logger.warning(
                "ExtractionEncoder.identity: expected raw dict, got %s",
                type(raw).__name__,
            )

        relations: list[ExtractedRelation] = []
        for item in records:
            lower = {str(k).lower().replace("-", "").replace("_", ""): v for k, v in item.items()}
            speaker = self._unwrap_gliner_scalar(lower.get("speaker"))
            name = self._unwrap_gliner_scalar(lower.get("name"))
            if speaker is None or name is None:
                continue
            speaker_clean = speaker.strip()
            name_clean = name.strip()
            if not speaker_clean or not name_clean:
                continue
            if speaker_clean.lower() == name_clean.lower():
                continue
            relations.append(
                ExtractedRelation(
                    subject=speaker_clean,
                    predicate="is",
                    object=name_clean,
                    confidence=1.0,
                    subject_label="speaker",
                    object_label="identity",
                )
            )
        return relations

    def classify(
        self,
        text: str,
        *,
        labels: Sequence[str],
        multi_label: bool = True,
        threshold: float = 0.0,
    ) -> list[tuple[str, float]]:
        """Zero-shot classification via gliner2's ``classify_text``.

        gliner2 returns the *selected* label(s) without per-label scores
        (single-label: a string; multi-label: a list of strings). We translate
        that into the substrate's expected ``[(label, score)]`` shape with
        ``score=1.0`` for selected labels.
        """

        self._ensure_loaded()
        start = time.time()
        label_list = list(labels)
        if not label_list:
            self._record_call((time.time() - start) * 1000)
            return []
        if multi_label:
            schema = {
                "_intent": {
                    "labels": label_list,
                    "multi_label": True,
                    "cls_threshold": float(threshold),
                }
            }
        else:
            schema = {"_intent": label_list}
        raw = self._model.classify_text(text, schema)
        selected = raw.get("_intent") if isinstance(raw, dict) else None
        results: list[tuple[str, float]] = []
        if isinstance(selected, list):
            for label in selected:
                if str(label).strip():
                    results.append((str(label), 1.0))
        elif isinstance(selected, str) and selected.strip():
            results.append((str(selected).strip(), 1.0))
        allowed = frozenset(str(lab) for lab in label_list)
        results = [(lab, scr) for lab, scr in results if lab in allowed]
        # Keep GLiNER2's emitted order — it encodes relevance. Sorting selected
        # labels into the caller's label-list order falsely makes whichever
        # label appears first *in `labels=`* win whenever every score was 1.0.
        latency = (time.time() - start) * 1000
        self._record_call(latency, method="classify")
        self._publish(
            "encoder.extraction.classify",
            {
                "text": text[:120],
                "labels": list(label_list),
                "selected": [label for label, _ in results],
                "multi_label": bool(multi_label),
                "latency_ms": latency,
            },
        )
        return results

    def process(self, text: str, **kwargs: Any) -> EncoderOutput:
        _ = kwargs
        self._ensure_loaded()
        start = time.time()
        result = ExtractionResult(raw_text=text)
        result.entities = self.extract_entities(text)
        result.relations = self.extract_relations(text)
        result.classifications["intent"] = self.classify(
            text,
            labels=("question", "statement", "request", "complaint", "praise"),
        )
        elapsed = (time.time() - start) * 1000
        return EncoderOutput(
            features=None,
            metadata={
                "entities": [
                    {"text": e.text, "label": e.label, "score": e.score}
                    for e in result.entities
                ],
                "relations": [
                    {"subject": r.subject, "predicate": r.predicate, "object": r.object, "confidence": r.confidence}
                    for r in result.relations
                ],
                "classifications": result.classifications,
            },
            confidence=max((e.score for e in result.entities), default=0.5),
            latency_ms=elapsed,
            encoder_name=self._name,
        )

    @staticmethod
    def _locate_span(text: str, mention: str) -> tuple[int, int]:
        if not mention:
            return (0, 0)
        idx = text.lower().find(mention.lower())
        if idx < 0:
            return (0, 0)
        return (idx, idx + len(mention))

    @staticmethod
    def _publish(topic: str, payload: dict) -> None:
        get_default_bus().publish(topic, payload)
