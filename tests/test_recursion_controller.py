from __future__ import annotations

import types

import pytest
import torch
import torch.nn as nn

from core.calibration.recursion_halt import RecursionHalt
from core.grafting.alignment import SWMToInputProjection
from core.grafts.swm_residual_graft import SWMResidualGraft
from core.host import LatentDecoder, LlamaBrocaHost
from core.substrate.recursion_controller import (
    LLAMA_THOUGHT_SLOT_FMT,
    RECURSIVE_THOUGHT_SLOT_FMT,
    RecursionController,
)
from core.substrate.prediction_error import PredictionErrorVector
from core.swm import EncoderSWMPublisher, SubstrateWorkingMemory, SWMSource
from core.symbolic import VSACodebook


D_MODEL = 4
VOCAB = 8


class _FakeLayer(nn.Module):
    def forward(self, x, *args, **kwargs):
        return (x + 0.01,)


class _FakeInnerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, D_MODEL)
        self.layers = nn.ModuleList([_FakeLayer(), _FakeLayer()])

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        return_dict=True,
        use_cache=False,
        past_key_values=None,
        **_kwargs,
    ):
        if inputs_embeds is None and input_ids is None:
            raise ValueError("must provide input_ids or inputs_embeds")
        x = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)[0]
        new_past = (past_key_values or 0) + 1
        return types.SimpleNamespace(last_hidden_state=x, past_key_values=new_past)


class _FakeLlamaLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(
            hidden_size=D_MODEL,
            max_position_embeddings=128,
            num_hidden_layers=2,
            model_type="llama",
        )
        self.model = _FakeInnerModel()
        self.lm_head = nn.Linear(D_MODEL, VOCAB, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight

    def get_input_embeddings(self):
        return self.model.embed_tokens


@pytest.fixture
def assembled_controller() -> tuple[RecursionController, SubstrateWorkingMemory, EncoderSWMPublisher, LlamaBrocaHost]:
    swm = SubstrateWorkingMemory()
    book = VSACodebook(dim=swm.dim, base_seed=0)
    errors = PredictionErrorVector()
    publisher = EncoderSWMPublisher(swm=swm, codebook=book, prediction_errors=errors, seed=0)

    host = LlamaBrocaHost(_FakeLlamaLM())

    proj = SWMToInputProjection(
        name="swm_to_host",
        d_swm=swm.dim,
        w_in_target=host.llm.model.embed_tokens.weight.detach(),
        seed=1,
    )
    graft = SWMResidualGraft(swm=swm, projection=proj)
    host.add_graft("final_hidden", graft)

    decoder = LatentDecoder(host=host, m_latent_steps=2)
    halt = RecursionHalt(swm=swm, max_rounds=2)
    controller = RecursionController(
        swm=swm,
        publisher=publisher,
        latent_decoder=decoder,
        residual_graft=graft,
        halt=halt,
    )
    return controller, swm, publisher, host


def test_run_requires_organ_slots_to_be_populated(assembled_controller):
    controller, _swm, _, _ = assembled_controller
    with pytest.raises(RuntimeError, match="organ slots"):
        controller.run(input_ids=torch.tensor([[1, 2, 3]]))


def test_run_produces_one_thought_and_one_llama_slot_per_round(assembled_controller):
    controller, swm, publisher, _ = assembled_controller
    publisher.publish_hidden(source=SWMSource.GLINER2, hidden=torch.randn(1, 4, 64), confidence=1.0)
    publisher.publish_hidden(source=SWMSource.GLICLASS, hidden=torch.randn(1, 4, 64), confidence=1.0)

    trace = controller.run(input_ids=torch.tensor([[1, 2, 3]]))

    assert trace.rounds == 2
    assert trace.thought_slots == [
        RECURSIVE_THOUGHT_SLOT_FMT.format(round=0),
        RECURSIVE_THOUGHT_SLOT_FMT.format(round=1),
    ]
    assert trace.llama_slots == [
        LLAMA_THOUGHT_SLOT_FMT.format(round=0),
        LLAMA_THOUGHT_SLOT_FMT.format(round=1),
    ]
    assert trace.final_thought_slot == RECURSIVE_THOUGHT_SLOT_FMT.format(round=1)
    assert trace.final_llama_slot == LLAMA_THOUGHT_SLOT_FMT.format(round=1)
    for slot_name in trace.thought_slots + trace.llama_slots:
        assert swm.has(slot_name)


def test_run_halts_on_max_rounds_with_correct_reason(assembled_controller):
    controller, _swm, publisher, _ = assembled_controller
    publisher.publish_hidden(source=SWMSource.GLINER2, hidden=torch.randn(1, 4, 64), confidence=1.0)
    publisher.publish_hidden(source=SWMSource.GLICLASS, hidden=torch.randn(1, 4, 64), confidence=1.0)

    trace = controller.run(input_ids=torch.tensor([[1, 2, 3]]))
    assert trace.halts[-1].halt is True
    assert trace.halts[-1].reason == "max_rounds_reached"


def test_run_rejects_2d_input():
    swm = SubstrateWorkingMemory()
    book = VSACodebook(dim=swm.dim, base_seed=0)
    errors = PredictionErrorVector()
    publisher = EncoderSWMPublisher(swm=swm, codebook=book, prediction_errors=errors, seed=0)
    host = LlamaBrocaHost(_FakeLlamaLM())
    proj = SWMToInputProjection(
        name="swm_to_host",
        d_swm=swm.dim,
        w_in_target=host.llm.model.embed_tokens.weight.detach(),
        seed=0,
    )
    graft = SWMResidualGraft(swm=swm, projection=proj)
    decoder = LatentDecoder(host=host, m_latent_steps=1)
    halt = RecursionHalt(swm=swm, max_rounds=1)
    controller = RecursionController(
        swm=swm,
        publisher=publisher,
        latent_decoder=decoder,
        residual_graft=graft,
        halt=halt,
    )

    publisher.publish_hidden(source=SWMSource.GLINER2, hidden=torch.randn(1, 4, 64), confidence=1.0)
    with pytest.raises(ValueError, match="batch, seq"):
        controller.run(input_ids=torch.tensor([1, 2, 3]))
