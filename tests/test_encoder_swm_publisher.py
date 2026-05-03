from __future__ import annotations

import pytest
import torch

from core.substrate.prediction_error import PredictionErrorVector
from core.swm import EncoderSWMPublisher, SubstrateWorkingMemory, SWMSource
from core.symbolic import VSACodebook, cosine, hypervector


@pytest.fixture
def fresh_swm_publisher() -> tuple[SubstrateWorkingMemory, EncoderSWMPublisher, VSACodebook]:
    swm = SubstrateWorkingMemory()
    book = VSACodebook(dim=swm.dim, base_seed=0)
    errors = PredictionErrorVector()
    pub = EncoderSWMPublisher(swm=swm, codebook=book, prediction_errors=errors, seed=0)
    return swm, pub, book


def test_publish_hidden_writes_pooled_normalized_slot(fresh_swm_publisher):
    swm, pub, _ = fresh_swm_publisher
    hidden = torch.randn(1, 7, 768)
    pub.publish_hidden(source=SWMSource.GLINER2, hidden=hidden, confidence=0.9)

    slot = swm.read("gliner2.hidden")
    assert slot.source is SWMSource.GLINER2
    assert slot.vector.shape == (swm.dim,)
    norm = slot.vector.norm().item()
    assert abs(norm - 1.0) < 1e-4, f"hidden slot must be unit-norm, got {norm:.6f}"


def test_publish_hidden_records_error(fresh_swm_publisher):
    _, pub, _ = fresh_swm_publisher
    pub.publish_hidden(source=SWMSource.GLINER2, hidden=torch.randn(1, 3, 768), confidence=0.7)
    assert abs(pub.prediction_errors.get(SWMSource.GLINER2).error - 0.3) < 1e-6


def test_publish_hidden_rejects_out_of_range_confidence(fresh_swm_publisher):
    _, pub, _ = fresh_swm_publisher
    with pytest.raises(ValueError, match="confidence"):
        pub.publish_hidden(source=SWMSource.GLINER2, hidden=torch.randn(1, 3, 768), confidence=1.5)


def test_publish_hidden_dim_change_raises(fresh_swm_publisher):
    _, pub, _ = fresh_swm_publisher
    pub.publish_hidden(source=SWMSource.GLINER2, hidden=torch.randn(1, 5, 768), confidence=1.0)
    with pytest.raises(ValueError, match="hidden dim must be stable"):
        pub.publish_hidden(source=SWMSource.GLINER2, hidden=torch.randn(1, 5, 384), confidence=1.0)


def test_publish_relations_round_trips_via_codebook(fresh_swm_publisher):
    swm, pub, book = fresh_swm_publisher
    pub.publish_relations(
        source=SWMSource.GLINER2,
        triples=[("ada", "lives_in", "london")],
    )
    slot = swm.read("gliner2.relations")
    name, _ = book.decode_role(
        slot.vector,
        "ROLE_OBJECT",
        candidates=["paris", "rome", "london", "berlin"],
    )
    assert name == "london"


def test_publish_classifications_recovers_label(fresh_swm_publisher):
    swm, pub, book = fresh_swm_publisher
    pub.publish_classifications(source=SWMSource.GLICLASS, labels=["question"])
    slot = swm.read("gliclass.classifications")
    cos = cosine(slot.vector, book.atom("question"))
    assert cos > 0.9, f"classification slot should align with its single label, cos={cos:.4f}"


def test_publish_entities_writes_a_slot(fresh_swm_publisher):
    swm, pub, _ = fresh_swm_publisher
    pub.publish_entities(
        source=SWMSource.GLINER2,
        entities=[("person", "ada"), ("location", "london")],
    )
    assert swm.has("gliner2.entities")
    slot = swm.read("gliner2.entities")
    assert slot.vector.shape == (swm.dim,)


def test_empty_payloads_are_silently_skipped(fresh_swm_publisher):
    swm, pub, _ = fresh_swm_publisher
    pub.publish_relations(source=SWMSource.GLINER2, triples=[])
    pub.publish_classifications(source=SWMSource.GLICLASS, labels=[])
    pub.publish_entities(source=SWMSource.GLINER2, entities=[])
    assert len(swm) == 0


def test_distinct_organs_get_distinct_jl_projections(fresh_swm_publisher):
    swm, pub, _ = fresh_swm_publisher
    pub.publish_hidden(source=SWMSource.GLINER2, hidden=torch.randn(1, 3, 64), confidence=1.0)
    pub.publish_hidden(source=SWMSource.GLICLASS, hidden=torch.randn(1, 3, 64), confidence=1.0)
    a = swm.read("gliner2.hidden").vector
    b = swm.read("gliclass.hidden").vector
    # Different JL seeds per organ -> different projections of unrelated inputs
    # should not happen to align by accident.
    cos = cosine(a, b)
    assert abs(cos) < 0.5, f"unrelated organ slots should be near-orthogonal, got cos={cos:.4f}"
