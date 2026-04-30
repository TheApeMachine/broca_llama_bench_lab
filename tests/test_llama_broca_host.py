
from __future__ import annotations

import types

import torch
import torch.nn as nn

from asi_broca_core.llama_broca_host import LlamaBrocaHost


class AddGraft(nn.Module):
    def forward(self, x, state):
        return x + 10.0


class FakeLayer(nn.Module):
    def forward(self, x, *args, **kwargs):
        return (x + 1.0,)


class FakeInnerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(8, 4)
        self.layers = nn.ModuleList([FakeLayer(), FakeLayer()])

    def forward(self, input_ids, attention_mask=None, return_dict=True):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)[0]
        return types.SimpleNamespace(last_hidden_state=x)


class FakeLlamaLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=4, max_position_embeddings=16, num_hidden_layers=2, model_type="llama")
        self.model = FakeInnerModel()
        self.lm_head = nn.Linear(4, 8, bias=False)


def test_llama_broca_host_layer_post_hook_changes_residual_stream():
    lm = FakeLlamaLM()
    host = LlamaBrocaHost(lm)
    slot = LlamaBrocaHost.layer_post_slot(0)
    host.add_graft(slot, AddGraft())

    idx = torch.tensor([[1, 2, 3]])
    logits, cache = host(idx, return_cache=True)

    assert logits.shape == (1, 3, 8)
    assert "layer.0.post.pre" in cache
    assert "layer.0.post.post" in cache
    diff = cache["layer.0.post.post"] - cache["layer.0.post.pre"]
    assert torch.allclose(diff, torch.full_like(diff, 10.0))


def test_llama_clear_slot_grafts_removes_layer_hook():
    lm = FakeLlamaLM()
    host = LlamaBrocaHost(lm)
    slot = LlamaBrocaHost.layer_post_slot(0)
    host.add_graft(slot, AddGraft())
    assert 0 in host._hook_handles
    host.clear_slot_grafts(slot)
    assert 0 not in host._hook_handles
