from __future__ import annotations

import types

import pytest
import torch
import torch.nn as nn

from core.host import DEFAULT_M_LATENT_STEPS, LatentDecoder, LlamaBrocaHost


D_MODEL = 4
VOCAB = 8


class _FakeLayer(nn.Module):
    def forward(self, x, *args, **kwargs):
        return (x + 0.01,)


class _FakeInnerModel(nn.Module):
    """Tiny stand-in for transformers.LlamaModel.

    Accepts both ``input_ids`` and ``inputs_embeds`` so the host's
    ``latent_forward`` path is exercised end-to-end without downloading a
    real Llama checkpoint.
    """

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

        if inputs_embeds is not None and input_ids is not None:
            raise ValueError("provide exactly one of input_ids / inputs_embeds")

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
        # Tied embeddings: lm_head.weight shares the embed_tokens.weight.
        self.lm_head.weight = self.model.embed_tokens.weight

    def get_input_embeddings(self):
        return self.model.embed_tokens


def _build_host_and_decoder(*, m: int = 3) -> tuple[LlamaBrocaHost, LatentDecoder]:
    host = LlamaBrocaHost(_FakeLlamaLM())
    decoder = LatentDecoder(host=host, m_latent_steps=m)
    return host, decoder


def test_decoder_alignment_is_identity_for_tied_embeddings():
    _, decoder = _build_host_and_decoder()
    eye = torch.eye(D_MODEL, dtype=torch.float32)
    diff = (decoder.alignment.matrix - eye).abs().max().item()
    assert diff < 1e-3, f"tied-embedding Wₐ should be identity, max abs deviation {diff:.6f}"


def test_default_m_latent_steps_is_40():
    assert DEFAULT_M_LATENT_STEPS == 40


def test_latent_forward_returns_hidden_and_past_kv():
    host, _ = _build_host_and_decoder()
    embeds = torch.randn(1, 3, D_MODEL)
    hidden, past_kv = host.latent_forward(inputs_embeds=embeds)
    assert hidden.shape == (1, 3, D_MODEL)
    assert past_kv == 1


def test_latent_forward_rejects_wrong_d_model():
    host, _ = _build_host_and_decoder()
    with pytest.raises(ValueError, match="d_model"):
        host.latent_forward(inputs_embeds=torch.randn(1, 3, D_MODEL + 1))


def test_latent_forward_rejects_2d_input():
    host, _ = _build_host_and_decoder()
    with pytest.raises(ValueError):
        host.latent_forward(inputs_embeds=torch.randn(3, D_MODEL))


def test_think_runs_m_latent_steps_and_returns_last_hidden():
    host, decoder = _build_host_and_decoder(m=5)
    input_ids = torch.tensor([[1, 2, 3]])
    last_hidden, past_kv = decoder.think(input_ids=input_ids)

    assert last_hidden.shape == (1, 1, D_MODEL)
    # 1 prompt forward + m=5 latent forwards = 6 model calls -> past_kv counter = 6
    assert past_kv == 6


def test_think_extends_attention_mask_each_step():
    """Each latent step appends one position; the underlying model sees a
    sequence that grows by one per step. This test verifies the call count."""

    _, decoder = _build_host_and_decoder(m=3)
    input_ids = torch.tensor([[1, 2, 3, 4]])
    _, past_kv = decoder.think(input_ids=input_ids)
    # 1 prompt + 3 latent = 4 model calls
    assert past_kv == 4


def test_think_rejects_host_without_latent_forward():
    """Construction is permissive (only embedding access is needed); .think()
    is where the host contract is enforced."""

    import types as _types

    class _MinimalHost:
        def __init__(self, lm):
            self.llm = lm
            self.lm_head = lm.lm_head

    decoder = LatentDecoder(host=_MinimalHost(_FakeLlamaLM()), m_latent_steps=1)
    with pytest.raises(TypeError, match="latent_forward"):
        decoder.think(input_ids=torch.tensor([[1, 2]]))


def test_decoder_rejects_non_positive_m():
    host, _ = _build_host_and_decoder()
    with pytest.raises(ValueError):
        LatentDecoder(host=host, m_latent_steps=0)


def test_layer_grafts_fire_during_latent_rollout():
    """Residual-stream grafts must apply during latent_forward exactly as in token forward."""

    class AddGraft(nn.Module):
        def __init__(self, delta):
            super().__init__()
            self.delta = float(delta)

        def forward(self, x, state):
            return x + self.delta

    host = LlamaBrocaHost(_FakeLlamaLM())
    slot = LlamaBrocaHost.layer_post_slot(0)
    host.add_graft(slot, AddGraft(7.0))

    plain_embeds = host.llm.model.embed_tokens(torch.tensor([[1, 2, 3]]))
    hidden_with_graft, _ = host.latent_forward(inputs_embeds=plain_embeds)

    host.clear_slot_grafts(slot)
    hidden_no_graft, _ = host.latent_forward(inputs_embeds=plain_embeds)

    diff = (hidden_with_graft - hidden_no_graft).mean().item()
    # Layer 0 graft adds 7.0; layer 1 then adds its +0.01 either way.
    assert abs(diff - 7.0) < 1e-3, f"expected +7.0 graft delta to propagate, got {diff:.6f}"
