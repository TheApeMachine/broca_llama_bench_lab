"""Tests for the dynamic graft synthesizer.

A miniature stub host stands in for ``LlamaBrocaHost`` so the synthesizer
can be exercised without loading any LLM weights. The stub implements
exactly the surface :func:`capture_activation_mode` calls into:

  *  ``parameters()`` (for device detection)
  *  ``eval()``
  *  ``grafts_enabled(bool)`` context manager
  *  ``__call__(ids, attention_mask, return_cache=True) -> (logits, cache_dict)``
  *  ``lm_head`` (an ``nn.Linear`` whose ``weight`` rows are the value-mode
     "next-token" directions)
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import pytest
import torch
import torch.nn as nn

from core.grafting.dynamic_grafts import (
    ACTIVATION_MODE_KIND,
    CapturedActivationMode,
    DynamicGraftSynthesizer,
    capture_activation_mode,
)
from core.grafting.grafts import KVMemoryGraft
from core.memory import SQLiteActivationMemory


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


@dataclass
class _Batch:
    ids: torch.Tensor
    attention_mask: torch.Tensor


class _StubTokenizer:
    """Deterministic batch_encode: hashes characters into the configured vocab."""

    def __init__(self, vocab: int = 64):
        self.vocab = int(vocab)
        self.token_to_id: dict[str, int] = {chr(i + 97): i % vocab for i in range(26)}

    def batch_encode(self, prompts, *, device=None):
        rows: list[list[int]] = []
        max_len = 0
        for p in prompts:
            ids = [(ord(c) % self.vocab) + 1 for c in str(p)]  # avoid 0 (pad)
            if not ids:
                ids = [1]
            rows.append(ids)
            max_len = max(max_len, len(ids))
        ids_t = torch.zeros((len(rows), max_len), dtype=torch.long)
        mask_t = torch.zeros((len(rows), max_len), dtype=torch.long)
        for i, row in enumerate(rows):
            ids_t[i, : len(row)] = torch.tensor(row, dtype=torch.long)
            mask_t[i, : len(row)] = 1
        if device is not None:
            ids_t = ids_t.to(device)
            mask_t = mask_t.to(device)
        return _Batch(ids=ids_t, attention_mask=mask_t)

    def encode(self, text: str):
        return [(ord(c) % self.vocab) + 1 for c in text]


class _StubHost(nn.Module):
    """A fixed transformer-shaped stub: embeds tokens, runs through one linear,
    captures the resulting hidden state as the activation cache, and uses an
    explicit ``lm_head`` so next_token value directions are well-defined."""

    def __init__(self, *, vocab: int = 64, d_model: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab + 1, d_model)
        self.transform = nn.Linear(d_model, d_model)
        self.lm_head = nn.Linear(d_model, vocab + 1, bias=False)
        self.cfg = type("cfg", (), {"d_model": d_model})()
        self._grafts_enabled = True

    @contextlib.contextmanager
    def grafts_enabled(self, enabled: bool):
        old = self._grafts_enabled
        self._grafts_enabled = bool(enabled)
        try:
            yield
        finally:
            self._grafts_enabled = old

    def forward(self, ids: torch.Tensor, attention_mask, *, return_cache: bool = False):
        x = self.embed(ids)
        h = torch.tanh(self.transform(x))
        cache = None
        if return_cache:
            cache = {"final_hidden.pre": h.detach().clone()}
        logits = self.lm_head(h)
        if return_cache:
            return logits, cache
        return logits


# ---------------------------------------------------------------------------
# capture_activation_mode
# ---------------------------------------------------------------------------


def _make_pair():
    torch.manual_seed(0)
    host = _StubHost()
    tok = _StubTokenizer()
    return host, tok


def test_capture_activation_mode_produces_unit_norm_key_and_value():
    host, tok = _make_pair()
    captured = capture_activation_mode(
        host, tok,
        name="poetic",
        prompt="reply only in poetry",
        slot="final_hidden",
        query_mode="sequence_mean",
        value_mode="mean_activation",
    )
    assert captured.name == "poetic"
    assert captured.slot == "final_hidden"
    assert captured.key.shape == (host.cfg.d_model,)
    assert captured.value.shape == (host.cfg.d_model,)
    # Both key and value must be unit vectors (numerical stability ε allowed).
    assert abs(captured.key.norm().item() - 1.0) < 1e-5
    assert abs(captured.value.norm().item() - 1.0) < 1e-5


def test_capture_activation_mode_with_last_token_query_uses_last_position():
    host, tok = _make_pair()
    captured = capture_activation_mode(
        host, tok,
        name="terse",
        prompt="ab",
        slot="final_hidden",
        query_mode="last_token",
        value_mode="mean_activation",
    )
    # Sanity: the captured key should differ from the sequence-mean key.
    captured_mean = capture_activation_mode(
        host, tok,
        name="terse_mean",
        prompt="ab",
        slot="final_hidden",
        query_mode="sequence_mean",
        value_mode="mean_activation",
    )
    assert not torch.allclose(captured.key, captured_mean.key)


def test_capture_activation_mode_next_token_value_aligns_with_lm_head():
    host, tok = _make_pair()
    target = "b"
    captured = capture_activation_mode(
        host, tok,
        name="say_b",
        prompt="prefix",
        slot="final_hidden",
        query_mode="sequence_mean",
        value_mode="next_token",
        target_token=target,
    )
    target_id = tok.token_to_id[target]
    # Note: token_to_id stores raw vocab index; batch_encode shifts by +1 to avoid pad=0.
    # capture_activation_mode resolves via tokenizer.token_to_id (no shift), so it
    # indexes the same row from lm_head.weight that the test will compare.
    expected = torch.nn.functional.normalize(
        host.lm_head.weight[target_id].detach().reshape(1, -1), dim=-1
    ).reshape(-1)
    # cosine similarity should be 1.0 (same direction).
    cos = torch.nn.functional.cosine_similarity(
        captured.value.reshape(1, -1), expected.reshape(1, -1)
    ).item()
    assert cos > 0.999


def test_capture_activation_mode_next_token_requires_target():
    host, tok = _make_pair()
    with pytest.raises(ValueError):
        capture_activation_mode(
            host, tok,
            name="bad",
            prompt="anything",
            value_mode="next_token",
            target_token=None,
        )


def test_capture_activation_mode_rejects_invalid_value_mode():
    host, tok = _make_pair()
    with pytest.raises(ValueError):
        capture_activation_mode(
            host, tok, name="bad", prompt="x",
            value_mode="not_a_real_mode",  # type: ignore[arg-type]
        )


def test_capture_activation_mode_rejects_invalid_query_mode():
    host, tok = _make_pair()
    with pytest.raises(ValueError):
        capture_activation_mode(
            host, tok, name="bad", prompt="x",
            query_mode="not_a_real_mode",
        )


# ---------------------------------------------------------------------------
# DynamicGraftSynthesizer
# ---------------------------------------------------------------------------


def test_synthesizer_persists_modes_across_processes(tmp_path):
    db = tmp_path / "activations.sqlite"
    store = SQLiteActivationMemory(db, default_namespace="dgst")
    synth = DynamicGraftSynthesizer(store, namespace="dgst")
    host, tok = _make_pair()

    captured = synth.synthesize(
        host, tok, name="poetic", prompt="reply only in poetry",
    )
    assert captured.record_id is not None
    assert synth.count() == 1

    # New synthesizer against the same store should still see the persisted mode.
    synth2 = DynamicGraftSynthesizer(SQLiteActivationMemory(db, default_namespace="dgst"), namespace="dgst")
    modes = synth2.list_modes()
    assert len(modes) == 1
    assert modes[0].name == "poetic"
    # Persisted key is unit-norm with the same dimensionality the host emitted.
    assert modes[0].key.shape == (host.cfg.d_model,)


def test_synthesizer_loads_modes_into_kv_graft(tmp_path):
    db = tmp_path / "activations.sqlite"
    store = SQLiteActivationMemory(db, default_namespace="dgst")
    synth = DynamicGraftSynthesizer(store, namespace="dgst")
    host, tok = _make_pair()

    synth.synthesize(host, tok, name="poetic", prompt="poetry mode")
    synth.synthesize(host, tok, name="terse", prompt="terse mode")

    graft = KVMemoryGraft(d_model=host.cfg.d_model, max_items=8)
    loaded = synth.load_modes(graft, clear_first=True)
    assert loaded == 2
    assert int(graft.keys.shape[0]) == 2
    # Filter loading: only the named mode should land in the graft.
    graft2 = KVMemoryGraft(d_model=host.cfg.d_model, max_items=8)
    loaded2 = synth.load_modes(graft2, names=["terse"])
    assert loaded2 == 1
    assert int(graft2.keys.shape[0]) == 1
    assert graft2.metadata[0].get("name") == "terse"


def test_synthesizer_remove_mode_clears_persisted_records(tmp_path):
    db = tmp_path / "activations.sqlite"
    store = SQLiteActivationMemory(db, default_namespace="dgst")
    synth = DynamicGraftSynthesizer(store, namespace="dgst")
    host, tok = _make_pair()

    synth.synthesize(host, tok, name="poetic", prompt="poetry")
    synth.synthesize(host, tok, name="poetic", prompt="poetry v2")  # duplicate name
    synth.synthesize(host, tok, name="other", prompt="other")
    assert synth.count() == 3
    n_removed = synth.remove_mode("poetic")
    assert n_removed == 2
    assert synth.count() == 1
    # The "other" mode survived.
    remaining = synth.list_modes()
    assert len(remaining) == 1
    assert remaining[0].name == "other"


def test_synthesizer_namespace_isolation(tmp_path):
    db = tmp_path / "activations.sqlite"
    store_a = SQLiteActivationMemory(db, default_namespace="ns_a")
    store_b = SQLiteActivationMemory(db, default_namespace="ns_b")
    synth_a = DynamicGraftSynthesizer(store_a, namespace="ns_a")
    synth_b = DynamicGraftSynthesizer(store_b, namespace="ns_b")
    host, tok = _make_pair()
    synth_a.synthesize(host, tok, name="m1", prompt="p1")
    assert synth_a.count() == 1
    assert synth_b.count() == 0


def test_loaded_mode_actually_modifies_kv_graft_retrieval(tmp_path):
    """End-to-end: loading a captured mode into a KVMemoryGraft and querying with the
    captured key should produce a non-zero residual delta."""

    db = tmp_path / "activations.sqlite"
    store = SQLiteActivationMemory(db, default_namespace="dgst")
    synth = DynamicGraftSynthesizer(store, namespace="dgst")
    host, tok = _make_pair()
    captured = synth.synthesize(host, tok, name="poetic", prompt="reply in poetry")

    graft = KVMemoryGraft(d_model=host.cfg.d_model, max_items=4, query_mode="sequence_mean")
    synth.load_modes(graft)

    # Construct a minimal residual stream where the sequence mean equals the captured key direction.
    bsz, seq_len = 1, 3
    x = captured.key.detach().clone().to(dtype=torch.float32).view(1, 1, -1).expand(bsz, seq_len, -1).contiguous()
    state = {
        "attention_mask": torch.ones((bsz, seq_len), dtype=torch.bool),
        "token_ids": torch.zeros((bsz, seq_len), dtype=torch.long),
        "last_indices": torch.tensor([seq_len - 1], dtype=torch.long),
    }
    out = graft(x, state)
    # The graft should have injected a non-trivial delta at the last position.
    delta_last = (out - x)[0, -1]
    assert delta_last.norm().item() > 1e-3


def test_capture_activation_mode_writes_metadata_tag():
    host, tok = _make_pair()
    captured = capture_activation_mode(host, tok, name="poetic", prompt="x")
    assert captured.metadata.get("tag") == ACTIVATION_MODE_KIND
    assert captured.metadata.get("name") == "poetic"
    assert captured.metadata.get("slot") == "final_hidden"


def test_synthesizer_cpu_round_trip_preserves_vectors():
    host, tok = _make_pair()
    captured = capture_activation_mode(host, tok, name="poetic", prompt="x")
    cpu = captured.cpu()
    assert torch.allclose(captured.key, cpu.key)
    assert torch.allclose(captured.value, cpu.value)
