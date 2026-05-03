from __future__ import annotations

import types

import pytest
import torch
import torch.nn as nn

from core.calibration.recursion_halt import RecursionHalt
from core.grafting.alignment import SWMToInputProjection
from core.grafts.swm_residual_graft import SWMResidualGraft
from core.host import LatentDecoder, LlamaBrocaHost
from core.perception.imagination import (
    DEFAULT_K_TRAJECTORIES,
    DEFAULT_T_HORIZON,
    ImaginedPlan,
    ImaginedTrajectory,
    VJEPAImaginer,
    jl_unfold_predictor,
)
from core.substrate.prediction_error import PredictionErrorVector
from core.substrate.recursion_controller import RecursionController
from core.swm import EncoderSWMPublisher, SubstrateWorkingMemory, SWMSource
from core.symbolic import VSACodebook


D_MODEL = 4
VOCAB = 8
D_VJEPA = 32


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
def assembled_imaginer() -> tuple[VJEPAImaginer, RecursionController, EncoderSWMPublisher, PredictionErrorVector]:
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
    decoder = LatentDecoder(host=host, m_latent_steps=1)
    halt = RecursionHalt(swm=swm, max_rounds=1)
    controller = RecursionController(
        swm=swm,
        publisher=publisher,
        latent_decoder=decoder,
        residual_graft=graft,
        halt=halt,
    )
    imaginer = VJEPAImaginer(
        swm=swm,
        publisher=publisher,
        controller=controller,
        prediction_errors=errors,
        k_trajectories=2,
        t_horizon=2,
        seed=7,
    )
    return imaginer, controller, publisher, errors


def test_default_k_and_t():
    assert DEFAULT_K_TRAJECTORIES == 3
    assert DEFAULT_T_HORIZON == 4


def test_jl_unfold_predictor_is_deterministic():
    pred = jl_unfold_predictor(seed=42)
    f = torch.zeros(D_VJEPA)
    a = pred(f, 0, 0)
    b = pred(f, 0, 0)
    assert torch.allclose(a, b)


def test_jl_unfold_predictor_diverges_across_trajectories():
    pred = jl_unfold_predictor(seed=42)
    f = torch.zeros(D_VJEPA)
    a = pred(f, 0, 0)
    b = pred(f, 1, 0)
    assert not torch.allclose(a, b)


def test_imagine_produces_k_trajectories(assembled_imaginer):
    imaginer, _, publisher, _ = assembled_imaginer
    publisher.publish_hidden(source=SWMSource.GLINER2, hidden=torch.randn(1, 4, 64), confidence=1.0)
    publisher.publish_hidden(source=SWMSource.GLICLASS, hidden=torch.randn(1, 4, 64), confidence=1.0)

    plan = imaginer.imagine(
        current_features=torch.randn(D_VJEPA),
        prompt_input_ids=torch.tensor([[1, 2, 3]]),
    )

    assert isinstance(plan, ImaginedPlan)
    assert len(plan.candidates) == 2
    for traj in plan.candidates:
        assert isinstance(traj, ImaginedTrajectory)
        assert len(traj.features) == 2
        assert traj.joint_free_energy >= 0.0


def test_imagine_chooses_min_efe_trajectory(assembled_imaginer):
    imaginer, _, publisher, _ = assembled_imaginer
    publisher.publish_hidden(source=SWMSource.GLINER2, hidden=torch.randn(1, 4, 64), confidence=1.0)
    publisher.publish_hidden(source=SWMSource.GLICLASS, hidden=torch.randn(1, 4, 64), confidence=1.0)

    plan = imaginer.imagine(
        current_features=torch.randn(D_VJEPA),
        prompt_input_ids=torch.tensor([[1, 2, 3]]),
    )

    min_efe = min(c.joint_free_energy for c in plan.candidates)
    assert plan.chosen.joint_free_energy == min_efe


def test_imagine_rejects_non_1d_or_2d_features(assembled_imaginer):
    imaginer, _, publisher, _ = assembled_imaginer
    publisher.publish_hidden(source=SWMSource.GLINER2, hidden=torch.randn(1, 4, 64), confidence=1.0)

    with pytest.raises(ValueError, match="1-D or 2-D"):
        imaginer.imagine(
            current_features=torch.randn(2, 3, D_VJEPA),
            prompt_input_ids=torch.tensor([[1, 2, 3]]),
        )


def test_imaginer_rejects_non_positive_k_or_t():
    swm = SubstrateWorkingMemory()
    book = VSACodebook(dim=swm.dim)
    errors = PredictionErrorVector()
    publisher = EncoderSWMPublisher(swm=swm, codebook=book, prediction_errors=errors)
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

    with pytest.raises(ValueError):
        VJEPAImaginer(
            swm=swm,
            publisher=publisher,
            controller=controller,
            prediction_errors=errors,
            k_trajectories=0,
            t_horizon=1,
        )

    with pytest.raises(ValueError):
        VJEPAImaginer(
            swm=swm,
            publisher=publisher,
            controller=controller,
            prediction_errors=errors,
            k_trajectories=1,
            t_horizon=0,
        )


def test_custom_predictor_is_used(assembled_imaginer):
    imaginer, _, publisher, _ = assembled_imaginer
    publisher.publish_hidden(source=SWMSource.GLINER2, hidden=torch.randn(1, 4, 64), confidence=1.0)

    sentinel_called: list[tuple[int, int]] = []

    def _predictor(current: torch.Tensor, k: int, t: int) -> torch.Tensor:
        sentinel_called.append((k, t))
        return torch.zeros_like(current)

    swm = imaginer._swm
    book = VSACodebook(dim=swm.dim)
    errors = imaginer._errors

    custom = VJEPAImaginer(
        swm=swm,
        publisher=imaginer._publisher,
        controller=imaginer._controller,
        prediction_errors=errors,
        k_trajectories=2,
        t_horizon=3,
        predictor=_predictor,
    )

    custom.imagine(
        current_features=torch.randn(D_VJEPA),
        prompt_input_ids=torch.tensor([[1, 2, 3]]),
    )

    assert len(sentinel_called) == 2 * 3
    assert {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)} == set(sentinel_called)
