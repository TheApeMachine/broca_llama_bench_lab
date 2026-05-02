from core.cognition.semantic_cascade import SemanticCascade


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
    cascade = SemanticCascade(classifier=classifier)

    result = cascade.intent_scores("Tell me a joke")

    assert result["label"] == "request"
    assert result["scores"]["request"] == 0.9
    assert result["allows_storage"] is False
    assert classifier.calls[0]["labels"] == {axis: list(labels) for axis, labels in SemanticCascade.AXES.items()}
    assert "identity_relations" not in result["evidence"]
    assert "fact_relations" not in result["evidence"]
    assert "intent_spans" not in result["evidence"]


def test_statement_axis_allows_durable_storage():
    classifier = StubSemanticClassificationEncoder(_axes(claim=1.0, greeting=0.2))
    cascade = SemanticCascade(classifier=classifier)

    result = cascade.intent_scores("I am the Magnificent")

    assert result["label"] == "statement"
    assert result["confidence"] == 1.0
    assert result["allows_storage"] is True


def test_storage_axis_can_block_non_durable_claims():
    classifier = StubSemanticClassificationEncoder(_axes(claim=0.9, non_storable=0.8, storable=0.2))
    cascade = SemanticCascade(classifier=classifier)

    result = cascade.intent_scores("That is cool")

    assert result["label"] == "statement"
    assert result["allows_storage"] is False
    assert result["evidence"]["semantic_allows_storage"] is False


def test_request_axis_selects_request_without_extraction_evidence():
    classifier = StubSemanticClassificationEncoder(_axes(greeting=0.4, request=0.95))
    cascade = SemanticCascade(classifier=classifier)

    result = cascade.intent_scores("Tell me a joke.")

    assert result["label"] == "request"
    assert result["allows_storage"] is False


def test_claim_axis_selects_statement_without_relation_evidence():
    classifier = StubSemanticClassificationEncoder(_axes(greeting=0.1, claim=0.95))
    cascade = SemanticCascade(classifier=classifier)

    result = cascade.intent_scores("Ada lives in Rome.")

    assert result["label"] == "statement"
    assert result["allows_storage"] is True


def test_highest_speech_axis_wins():
    classifier = StubSemanticClassificationEncoder(_axes(question=0.7, request=0.8))
    cascade = SemanticCascade(classifier=classifier)

    result = cascade.intent_scores("Tell me a joke.")

    assert result["label"] == "request"
