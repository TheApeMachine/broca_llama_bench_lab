import pytest

from core.encoders.classification import SemanticClassificationEncoder


def test_semantic_classification_encoder_normalizes_hierarchical_axes():
    encoder = SemanticClassificationEncoder()
    labels = {"speech_act": ["claim", "request"], "polarity": ["affirmation", "negation"]}
    raw = {
        "speech_act": {"claim": 0.8, "request": 0.2},
        "polarity": {"affirmation": 0.7, "negation": 0.1},
    }

    out = encoder._normalize_axes(raw, labels)

    assert out == raw


def test_semantic_classification_encoder_rejects_missing_axis():
    encoder = SemanticClassificationEncoder()
    labels = {"speech_act": ["claim", "request"]}

    with pytest.raises(RuntimeError, match="missing hierarchical axis"):
        encoder._normalize_axes({}, labels)


def test_semantic_classification_encoder_rejects_missing_label():
    encoder = SemanticClassificationEncoder()
    labels = {"speech_act": ["claim", "request"]}

    with pytest.raises(RuntimeError, match="missing label"):
        encoder._normalize_axes({"speech_act": {"claim": 0.8}}, labels)
