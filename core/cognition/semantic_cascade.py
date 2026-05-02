"""Hierarchical semantic analysis feeding the cognition gates."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..encoders.classification import SemanticClassificationEncoder
from ..encoders.extraction import ExtractedEntity, ExtractedRelation, ExtractionEncoder


class SemanticCascade:
    """Run parallel semantic axes, then collapse them into substrate intent."""

    AXES: dict[str, tuple[str, ...]] = {
        "speech_act": ("claim", "question", "request", "command", "greeting", "feedback"),
        "polarity": ("affirmation", "negation", "correction"),
        "content_role": ("self_description", "world_fact", "task_instruction", "social_signal"),
        "storage": ("storable", "non_storable"),
    }
    SPAN_LABELS: tuple[str, ...] = (
        "claim",
        "question",
        "request",
        "command",
        "greeting",
        "feedback",
        "negation",
        "correction",
    )
    PROMPT = (
        "Classify this utterance for a cognitive substrate. Separate speech act, "
        "polarity, content role, and whether the utterance contains durable semantic content."
    )
    EXAMPLES: tuple[dict[str, Any], ...] = (
        {
            "text": "Ada lives in Rome.",
            "labels": [
                "speech_act.claim",
                "polarity.affirmation",
                "content_role.world_fact",
                "storage.storable",
            ],
        },
        {
            "text": "Where is Ada?",
            "labels": [
                "speech_act.question",
                "polarity.affirmation",
                "content_role.world_fact",
                "storage.non_storable",
            ],
        },
        {
            "text": "Tell me a joke.",
            "labels": [
                "speech_act.request",
                "polarity.affirmation",
                "content_role.task_instruction",
                "storage.non_storable",
            ],
        },
        {
            "text": "No, I did not say that.",
            "labels": [
                "speech_act.feedback",
                "polarity.correction",
                "content_role.social_signal",
                "storage.non_storable",
            ],
        },
        {
            "text": "I am the Magnificent.",
            "labels": [
                "speech_act.claim",
                "polarity.affirmation",
                "content_role.self_description",
                "storage.storable",
            ],
        },
    )
    INTENT_FROM_SPEECH_ACT: dict[str, str] = {
        "claim": "statement",
        "question": "question",
        "request": "request",
        "command": "command",
        "greeting": "greeting",
        "feedback": "feedback",
    }

    def __init__(
        self,
        *,
        classifier: SemanticClassificationEncoder,
        extraction: ExtractionEncoder,
    ):
        self.classifier = classifier
        self.extraction = extraction
        self._labels = {axis: list(labels) for axis, labels in self.AXES.items()}

    def intent_scores(self, text: str) -> dict[str, Any]:
        if not text.strip():
            raise ValueError("SemanticCascade.intent_scores requires non-empty text")
        branches = self._run_branches(text)
        extraction = branches["extraction"]
        identity_relations = extraction["identity_relations"]
        fact_relations = extraction["fact_relations"]
        intent_spans = extraction["intent_spans"]
        axes = branches["axes"]
        speech_scores = self._require_axis(axes, "speech_act")
        semantic_scores = {
            canonical: float(speech_scores[source])
            for source, canonical in self.INTENT_FROM_SPEECH_ACT.items()
            if source in speech_scores
        }
        if set(semantic_scores) != set(self.INTENT_FROM_SPEECH_ACT.values()):
            raise RuntimeError(
                f"SemanticCascade.intent_scores: incomplete intent scores {semantic_scores!r}"
            )
        span_scores = self._span_intent_scores(text, intent_spans)
        polarity_scores = self._span_polarity_scores(text, intent_spans)
        scores = dict(semantic_scores)
        for intent_label, span_score in span_scores.items():
            scores[intent_label] = max(scores[intent_label], span_score)

        if identity_relations:
            identity_confidence = max(float(rel.confidence) for rel in identity_relations)
            scores["statement"] = max(scores["statement"], identity_confidence)
            label = "statement"
            confidence = scores[label]
        elif span_scores:
            label = max(
                span_scores,
                key=lambda item: (span_scores[item], semantic_scores[item]),
            )
            confidence = max(span_scores[label], semantic_scores[label])
        elif fact_relations:
            fact_confidence = max(float(rel.confidence) for rel in fact_relations)
            scores["statement"] = max(scores["statement"], fact_confidence)
            label = "statement"
            confidence = scores[label]
        elif polarity_scores:
            polarity_confidence = max(polarity_scores.values())
            scores["feedback"] = max(scores["feedback"], polarity_confidence)
            label = "feedback"
            confidence = scores[label]
        else:
            label, confidence = max(scores.items(), key=lambda item: item[1])

        allows_storage = self._allows_storage(label, axes, identity_relations, fact_relations)
        return {
            "label": label,
            "confidence": float(confidence),
            "scores": scores,
            "allows_storage": allows_storage,
            "evidence": {
                "semantic_axes": axes,
                "semantic_allows_storage": allows_storage,
                "intent_spans": [
                    {
                        "text": span.text,
                        "label": span.label,
                        "score": float(span.score),
                        "start": int(span.start),
                        "end": int(span.end),
                    }
                    for span in intent_spans
                ],
                "identity_relations": [
                    {
                        "subject": rel.subject,
                        "predicate": rel.predicate,
                        "object": rel.object,
                        "confidence": float(rel.confidence),
                    }
                    for rel in identity_relations
                ],
                "fact_relations": [
                    {
                        "subject": rel.subject,
                        "predicate": rel.predicate,
                        "object": rel.object,
                        "confidence": float(rel.confidence),
                    }
                    for rel in fact_relations
                ],
            },
        }

    def _run_branches(self, text: str) -> dict[str, Any]:
        branches = {
            "extraction": lambda: self._extract_semantic_evidence(text),
            "axes": lambda: self.classifier.classify_axes(
                text,
                self._labels,
                prompt=self.PROMPT,
                examples=self.EXAMPLES,
            ),
        }
        with ThreadPoolExecutor(max_workers=len(branches)) as executor:
            futures = {name: executor.submit(branch) for name, branch in branches.items()}
            return {name: future.result() for name, future in futures.items()}

    def _extract_semantic_evidence(self, text: str) -> dict[str, Any]:
        relations = self.extraction.extract_relations(text)
        intent_spans = self.extraction.extract_entities(text, labels=self.SPAN_LABELS)
        identity_relations = [
            rel
            for rel in relations
            if rel.subject_label == "speaker" and rel.object_label == "identity"
        ]
        fact_relations = [rel for rel in relations if rel not in identity_relations]
        return {
            "identity_relations": identity_relations,
            "fact_relations": fact_relations,
            "intent_spans": intent_spans,
        }

    def _span_intent_scores(
        self,
        text: str,
        spans: list[ExtractedEntity],
    ) -> dict[str, float]:
        denom = float(len(text.strip()))
        scores: dict[str, float] = {}
        for span in spans:
            source_label = span.label.strip().lower()
            canonical = self.INTENT_FROM_SPEECH_ACT.get(source_label)
            if canonical is None:
                continue
            coverage = self._span_coverage(span, denom)
            scores[canonical] = max(scores.get(canonical, 0.0), coverage)
        return scores

    def _span_polarity_scores(
        self,
        text: str,
        spans: list[ExtractedEntity],
    ) -> dict[str, float]:
        denom = float(len(text.strip()))
        out: dict[str, float] = {}
        for span in spans:
            label = span.label.strip().lower()
            if label not in {"negation", "correction"}:
                continue
            out[label] = max(out.get(label, 0.0), self._span_coverage(span, denom))
        return out

    @staticmethod
    def _span_coverage(span: ExtractedEntity, denom: float) -> float:
        span_len = span.end - span.start
        if span_len <= 0:
            span_len = len(span.text.strip())
        return min(float(span_len) / denom, 1.0)

    def _allows_storage(
        self,
        label: str,
        axes: dict[str, dict[str, float]],
        identity_relations: list[ExtractedRelation],
        fact_relations: list[ExtractedRelation],
    ) -> bool:
        if label != "statement":
            return False
        if identity_relations or fact_relations:
            return True
        storage_scores = self._require_axis(axes, "storage")
        for required in ("storable", "non_storable"):
            if required not in storage_scores:
                raise RuntimeError(f"SemanticCascade requires storage label {required!r}")
        return storage_scores["storable"] >= storage_scores["non_storable"]

    @staticmethod
    def _require_axis(axes: dict[str, dict[str, float]], axis: str) -> dict[str, float]:
        values = axes.get(axis)
        if not isinstance(values, dict) or not values:
            raise RuntimeError(f"SemanticCascade requires axis {axis!r}, got {axes!r}")
        return values
