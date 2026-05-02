from core.cognition.semantic_cascade import SemanticCascade
from core.encoders.extraction import ExtractedEntity, ExtractedRelation


class StubSemanticClassificationEncoder:
    def __init__(self, axes):
        self.axes = axes
        self.calls = []

    def classify_axes(self, text, labels, *, prompt, examples):
        self.calls.append(
            {
                "text": text,
                "labels": labels,
                "prompt": prompt,
                "examples": examples,
            }
        )
        return self.axes


class StubExtractionEncoder:
    def __init__(self, *, relations=None, spans=None):
        self.relations = list(relations or [])
        self.spans = list(spans or [])
        self.relation_calls = []
        self.entity_calls = []

    def extract_relations(self, text):
        self.relation_calls.append(text)
        return list(self.relations)

    def extract_entities(self, text, *, labels):
        self.entity_calls.append((text, tuple(labels)))
        return list(self.spans)


def _axes(*, storable=1.0, non_storable=0.0, **speech_scores):
    return {
        "speech_act": {
            "claim": speech_scores.get("claim", 0.0),
            "question": speech_scores.get("question", 0.0),
            "request": speech_scores.get("request", 0.0),
            "command": speech_scores.get("command", 0.0),
            "greeting": speech_scores.get("greeting", 0.0),
            "feedback": speech_scores.get("feedback", 0.0),
        },
        "polarity": {"affirmation": 1.0, "negation": 0.0, "correction": 0.0},
        "content_role": {"self_description": 0.0, "world_fact": 1.0, "task_instruction": 0.0, "social_signal": 0.0},
        "storage": {"storable": storable, "non_storable": non_storable},
    }


def test_cascade_maps_request_axis_to_request_intent():
    classifier = StubSemanticClassificationEncoder(_axes(request=0.9, claim=0.1))
    extraction = StubExtractionEncoder()
    cascade = SemanticCascade(classifier=classifier, extraction=extraction)

    result = cascade.intent_scores("Tell me a joke")

    assert result["label"] == "request"
    assert result["scores"]["request"] == 0.9
    assert result["allows_storage"] is False
    assert classifier.calls[0]["labels"] == {axis: list(labels) for axis, labels in SemanticCascade.AXES.items()}
    assert extraction.relation_calls == ["Tell me a joke"]
    assert extraction.entity_calls == [
        ("Tell me a joke", SemanticCascade.SPEECH_SPAN_LABELS),
        ("Tell me a joke", SemanticCascade.SOCIAL_SPAN_LABELS),
    ]


def test_identity_relation_overrides_greeting_axis_as_statement():
    classifier = StubSemanticClassificationEncoder(_axes(greeting=1.0, claim=0.2))
    extraction = StubExtractionEncoder(
        relations=[
            ExtractedRelation(
                subject="I",
                predicate="is",
                object="the Magnificent",
                confidence=1.0,
                subject_label="speaker",
                object_label="identity",
            )
        ]
    )
    cascade = SemanticCascade(classifier=classifier, extraction=extraction)

    result = cascade.intent_scores("I am the Magnificent")

    assert result["label"] == "statement"
    assert result["confidence"] == 1.0
    assert result["allows_storage"] is True
    assert result["evidence"]["identity_relations"][0]["object"] == "the Magnificent"


def test_storage_axis_can_block_non_durable_claims():
    classifier = StubSemanticClassificationEncoder(_axes(claim=0.9, non_storable=0.8, storable=0.2))
    extraction = StubExtractionEncoder()
    cascade = SemanticCascade(classifier=classifier, extraction=extraction)

    result = cascade.intent_scores("That is cool")

    assert result["label"] == "statement"
    assert result["allows_storage"] is False
    assert result["evidence"]["semantic_allows_storage"] is False


def test_span_evidence_overrides_bad_semantic_top_label():
    classifier = StubSemanticClassificationEncoder(_axes(greeting=0.95, request=0.4))
    extraction = StubExtractionEncoder(
        spans=[
            ExtractedEntity(
                text="Tell me a joke",
                label="request",
                score=1.0,
                start=0,
                end=14,
            )
        ]
    )
    cascade = SemanticCascade(classifier=classifier, extraction=extraction)

    result = cascade.intent_scores("Tell me a joke.")

    assert result["label"] == "request"
    assert result["evidence"]["intent_spans"][0]["label"] == "request"


def test_fact_relation_overrides_bad_semantic_top_label():
    classifier = StubSemanticClassificationEncoder(_axes(greeting=0.95, claim=0.1))
    extraction = StubExtractionEncoder(
        relations=[
            ExtractedRelation(
                subject="ada",
                predicate="lives in",
                object="rome",
                confidence=1.0,
            )
        ]
    )
    cascade = SemanticCascade(classifier=classifier, extraction=extraction)

    result = cascade.intent_scores("Ada lives in Rome.")

    assert result["label"] == "statement"
    assert result["allows_storage"] is True
    assert result["evidence"]["fact_relations"][0]["subject"] == "ada"


def test_request_span_wins_same_coverage_question_span():
    classifier = StubSemanticClassificationEncoder(_axes(question=0.7, request=0.4))
    extraction = StubExtractionEncoder(
        spans=[
            ExtractedEntity(
                text="Tell me a joke",
                label="question",
                score=1.0,
                start=0,
                end=14,
            ),
            ExtractedEntity(
                text="Tell me a joke",
                label="request",
                score=1.0,
                start=0,
                end=14,
            ),
        ]
    )
    cascade = SemanticCascade(classifier=classifier, extraction=extraction)

    result = cascade.intent_scores("Tell me a joke.")

    assert result["label"] == "request"
