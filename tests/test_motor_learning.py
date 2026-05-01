"""Tests for the motor-learning trainer.

Builds a minimal host + tokenizer pair so the gradient bookkeeping can be
verified without loading a real Llama checkpoint. The host implements the
exact surface ``GraftMotorTrainer`` calls into: ``parameters()``, ``train()``,
``eval()``, and ``__call__(input_ids, mask, extra_state=...) -> logits``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import pytest

from asi_broca_core.motor_learning import (
    GraftMotorTrainer,
    MotorLearningConfig,
    freeze_all_but,
)


class _StubHFTokenizer:
    def __init__(self, vocab: int = 50):
        self.vocab = vocab

    def apply_chat_template(
        self, messages, add_generation_prompt: bool = True, return_tensors: str = "pt"
    ):
        # Encode each message naively as the index of its content[0] mod vocab.
        ids = []
        for m in messages:
            for ch in m.get("content", ""):
                ids.append(ord(ch) % self.vocab)
        if not ids:
            ids = [1]
        return torch.tensor(ids, dtype=torch.long).view(1, -1)


class _Tokenizer:
    def __init__(self, hf):
        self.inner = hf


class _StubHost(nn.Module):
    """Tiny transformer-shaped host with a single residual + a Broca-style add."""

    def __init__(self, vocab: int = 50, d: int = 8):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.proj = nn.Linear(d, vocab, bias=False)
        self.graft_module = nn.Linear(d, d, bias=False)  # the only trainable thing
        self.cfg = type("cfg", (), {"d_model": d})()

    def forward(self, input_ids, attention_mask, extra_state=None):
        x = self.embed(input_ids)
        # Apply the "graft" only at the last position to mirror the real path.
        last_idx = input_ids.shape[1] - 1
        x_last = x[:, last_idx]
        delta = self.graft_module(x_last)
        x = x.clone()
        x[:, last_idx] = x_last + delta
        return self.proj(x)


def test_freeze_all_but_keeps_only_listed_params_trainable():
    host = _StubHost()
    grafts = [host.graft_module]
    kept = freeze_all_but([p for g in grafts for p in g.parameters()], host)
    assert kept == {id(p) for p in host.graft_module.parameters()}
    for name, p in host.named_parameters():
        if id(p) in kept:
            assert p.requires_grad
        else:
            assert not p.requires_grad


def test_motor_trainer_only_updates_graft_params():
    host = _StubHost()
    tokenizer = _Tokenizer(_StubHFTokenizer(vocab=50))
    trainer = GraftMotorTrainer(
        host,
        tokenizer,
        [host.graft_module],
        config=MotorLearningConfig(min_replay_for_step=1, max_replay_per_tick=4),
    )
    embed_before = host.embed.weight.detach().clone()
    proj_before = host.proj.weight.detach().clone()
    graft_before = host.graft_module.weight.detach().clone()

    replay = [
        {
            "messages": [{"role": "user", "content": "hello world"}],
            "speech_plan_tokens": torch.tensor([3, 7, 11], dtype=torch.long).clone(),
        }
        for _ in range(4)
    ]

    result = trainer.step(replay)
    assert not result.get("skipped", False), result
    assert result["items"] == 4

    # Frozen params must not move.
    assert torch.allclose(host.embed.weight, embed_before)
    assert torch.allclose(host.proj.weight, proj_before)
    # Graft must move.
    assert not torch.allclose(host.graft_module.weight, graft_before)


def test_motor_trainer_skips_when_replay_below_threshold():
    host = _StubHost()
    tokenizer = _Tokenizer(_StubHFTokenizer())
    trainer = GraftMotorTrainer(
        host,
        tokenizer,
        [host.graft_module],
        config=MotorLearningConfig(min_replay_for_step=4),
    )
    replay = [
        {
            "messages": [{"role": "user", "content": "hi"}],
            "speech_plan_tokens": torch.tensor([1]),
        }
    ]
    out = trainer.step(replay)
    assert out["skipped"] is True


def test_motor_trainer_loss_is_finite_and_steps_increment():
    host = _StubHost()
    tokenizer = _Tokenizer(_StubHFTokenizer())
    trainer = GraftMotorTrainer(
        host,
        tokenizer,
        [host.graft_module],
        config=MotorLearningConfig(min_replay_for_step=2, max_replay_per_tick=4),
    )
    replay = [
        {
            "messages": [{"role": "user", "content": "hello"}],
            "speech_plan_tokens": torch.tensor([2, 4], dtype=torch.long),
        },
        {
            "messages": [{"role": "user", "content": "world"}],
            "speech_plan_tokens": torch.tensor([3, 5], dtype=torch.long),
        },
    ]
    out1 = trainer.step(replay)
    out2 = trainer.step(replay)
    assert out1["steps"] == 1
    assert out2["steps"] == 2
    assert torch.isfinite(torch.tensor(out1["loss"]))
    assert torch.isfinite(torch.tensor(out2["loss"]))
