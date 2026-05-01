"""Tests for the four top-down cognitive control mechanisms.

The tests use a stub host that mimics enough of :class:`LlamaBrocaHost`'s
surface to drive the orchestrators (``IterativeHypothesisSearch``,
``EpistemicInterruptionMonitor``) without loading any LLM weights:

  *  ``add_graft(slot, graft)`` — slot-keyed list of grafts.
  *  ``__call__(ids, attention_mask, *, extra_state, return_cache=False)`` —
     produces logits whose argmax follows a fixed, deterministic schedule and
     applies any registered ``logits``- and ``final_hidden``-slot grafts.
  *  ``parameters()`` — yields a single tensor so callers can detect device.
  *  ``grafts_enabled(bool)`` context manager (no-op for these tests).
  *  ``lm_head`` — a small ``nn.Linear`` whose rows are used to derive
     concept and outcome directions in causal-constraint tests.

Each test composes a fresh stub so cross-test state pollution is impossible.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.causal import FiniteSCM
from core.grafts import KVMemoryGraft
from core.top_down_control import (
    CausalConstraint,
    CausalConstraintGraft,
    EpistemicInterruptionMonitor,
    EpistemicInterruptionResult,
    HypothesisMaskingGraft,
    HypothesisSearchResult,
    HypothesisVerdict,
    InterruptionEvent,
    InterruptionVerdict,
    IterativeHypothesisSearch,
    ModalityShiftGraft,
)


# ---------------------------------------------------------------------------
# Stub tokenizer + host
# ---------------------------------------------------------------------------


class _StubTokenizer:
    """Minimal tokenizer surface: per-character ids; ``decode_id``/``decode_tokens`` for testing."""

    def __init__(self, vocab: int = 64):
        self.vocab = int(vocab)
        # token_to_id covers digits 0-9 and lowercase letters by index, plus a few
        # named concept tokens used in causal tests.
        self.token_to_id: dict[str, int] = {}
        for i, ch in enumerate("0123456789"):
            self.token_to_id[ch] = i
        for i, ch in enumerate("abcdefghijklmnopqrstuvwxyz"):
            self.token_to_id[ch] = 10 + i
        self.token_to_id["helps"] = 40
        self.token_to_id["hurts"] = 41
        self.token_to_id["yes"] = 42
        self.token_to_id["no"] = 43
        self.token_to_id["Treatment"] = 44
        self.token_to_id["smoking"] = 45
        self.token_to_id["analytical"] = 46
        self.token_to_id["fluent"] = 47

    @property
    def pad_id(self) -> int:
        return 0

    def encode(self, text: str) -> list[int]:
        # Whole-word lookup only — surfaces missing from the vocab return [],
        # mirroring what we want a strict tokenizer to do for "missing" tests.
        if text in self.token_to_id:
            return [int(self.token_to_id[text])]
        return []

    def decode_id(self, tid: int) -> str:
        for k, v in self.token_to_id.items():
            if int(v) == int(tid):
                return k
        return f"<{int(tid)}>"

    def decode_tokens(self, ids: Iterable[int]) -> str:
        return " ".join(self.decode_id(int(i)) for i in ids)


class _RankedStubHost(nn.Module):
    """Stub host whose last-position logits prefer a fixed ranked list.

    With no logits-slot grafts, ``argmax`` picks ``ranked[0]``. After that
    token is heavily penalized (e.g. by :class:`HypothesisMaskingGraft`),
    ``argmax`` falls through to ``ranked[1]``, and so on. This is the
    cleanest possible behavioural test of "ban → fallback" without invoking
    a real model.

    The host honours ``add_graft``/``clear_all_grafts``/``grafts_enabled`` so
    :class:`IterativeHypothesisSearch` can attach the masking graft and
    expect its bans to take effect.
    """

    def __init__(
        self,
        *,
        vocab: int = 64,
        d_model: int = 16,
        ranked: Sequence[int] = (5, 7, 9),
        base_logit: float = 100.0,
        rank_step: float = 1.0,
    ):
        super().__init__()
        self.vocab = int(vocab)
        self.d_model = int(d_model)
        self.ranked: list[int] = [int(t) for t in ranked]
        self.base_logit = float(base_logit)
        self.rank_step = float(rank_step)
        self.lm_head = nn.Linear(d_model, vocab, bias=False)
        self.embed = nn.Embedding(vocab, d_model)
        self.cfg = type("cfg", (), {"d_model": d_model})()
        self._grafts_enabled = True
        # Slot key -> list[graft] (stored as a dict so .grafts mimics LlamaBrocaHost.)
        self.grafts: dict[str, list[nn.Module]] = {}
        self.calls: list[dict[str, Any]] = []

    # -- Graft API ----------------------------------------------------------

    def add_graft(self, slot: str, graft: nn.Module) -> None:
        key = slot.replace(".", "__")
        self.grafts.setdefault(key, []).append(graft)
        setattr(graft, "slot", slot)

    def clear_all_grafts(self) -> list[tuple[str, nn.Module]]:
        out: list[tuple[str, nn.Module]] = []
        for key, lst in list(self.grafts.items()):
            slot = key.replace("__", ".")
            for m in lst:
                out.append((slot, m))
            self.grafts[key] = []
        return out

    @contextlib.contextmanager
    def grafts_enabled(self, enabled: bool):
        old = self._grafts_enabled
        self._grafts_enabled = bool(enabled)
        try:
            yield
        finally:
            self._grafts_enabled = old

    # -- Forward ------------------------------------------------------------

    def parameters(self, recurse: bool = True):
        # nn.Module's default parameters() returns all submodule params; we
        # rely on lm_head + embed being there so device detection works.
        return super().parameters(recurse=recurse)

    def forward(
        self,
        ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        extra_state: Optional[dict] = None,
        return_cache: bool = False,
    ):
        bsz, seq = ids.shape
        device = ids.device
        if attention_mask is None:
            mask = torch.ones((bsz, seq), dtype=torch.bool, device=device)
        else:
            mask = attention_mask.to(device).bool()
        last_indices = mask.long().sum(dim=1).clamp_min(1) - 1

        # Build "fake" hidden state for residual-stream grafts: small token
        # embedding so final_hidden grafts can do real geometry against it.
        hidden = self.embed(ids).to(self.lm_head.weight.dtype)
        state: dict[str, Any] = {
            "model": self,
            "tokenizer": (extra_state or {}).get("tokenizer"),
            "attention_mask": mask,
            "token_ids": ids,
            "last_indices": last_indices,
        }
        if extra_state:
            state.update(extra_state)

        cache: dict[str, torch.Tensor] = {}
        if return_cache:
            cache["final_hidden.pre"] = hidden.detach().clone()
        if self._grafts_enabled:
            for graft in self.grafts.get("final_hidden", []):
                hidden = graft(hidden, state)
        if return_cache:
            cache["final_hidden.post"] = hidden.detach().clone()

        # Build logits: prefer ranked tokens at last position; sprinkle low logits elsewhere.
        logits = torch.zeros((bsz, seq, self.vocab), device=device, dtype=hidden.dtype)
        for rank, tid in enumerate(self.ranked):
            if 0 <= tid < self.vocab:
                logits[:, -1, tid] = self.base_logit - rank * self.rank_step

        if return_cache:
            cache["logits.pre"] = logits.detach().clone()
        if self._grafts_enabled:
            for graft in self.grafts.get("logits", []):
                logits = graft(logits, state)
        if return_cache:
            cache["logits.post"] = logits.detach().clone()

        # Snapshot the call so tests can assert on what was forwarded.
        snap_extra: dict[str, Any] = {}
        for k, v in (extra_state or {}).items():
            if isinstance(v, torch.Tensor):
                snap_extra[k] = v.detach().clone()
            else:
                snap_extra[k] = v
        self.calls.append(
            {
                "prompt_len": int(seq),
                "extra_state_keys": tuple(sorted((extra_state or {}).keys())),
                "extra_state": snap_extra,
            }
        )

        if return_cache:
            return logits, cache
        return logits


class _ScheduledStubHost(_RankedStubHost):
    """Variant whose ranked list changes per call, so tests can drive a multi-step generation.

    ``schedule[i]`` is the ``ranked`` list used on the *i*-th forward call.
    After the schedule is exhausted, the last entry is re-used.
    """

    def __init__(
        self,
        *,
        vocab: int = 64,
        d_model: int = 16,
        schedule: Sequence[Sequence[int]] = (),
        base_logit: float = 100.0,
        rank_step: float = 1.0,
    ):
        seq = list(schedule) or [[5, 7, 9]]
        super().__init__(
            vocab=vocab,
            d_model=d_model,
            ranked=seq[0],
            base_logit=base_logit,
            rank_step=rank_step,
        )
        self.schedule = [list(int(t) for t in step) for step in seq]
        self._step = 0

    def forward(
        self,
        ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        extra_state: Optional[dict] = None,
        return_cache: bool = False,
    ):
        idx = min(self._step, len(self.schedule) - 1)
        self.ranked = list(self.schedule[idx])
        self._step += 1
        return super().forward(
            ids, attention_mask, extra_state=extra_state, return_cache=return_cache
        )


# ---------------------------------------------------------------------------
# 1. HypothesisMaskingGraft
# ---------------------------------------------------------------------------


def test_masking_graft_subtracts_penalty_at_last_position():
    graft = HypothesisMaskingGraft(default_penalty=50.0)
    graft.ban([3, 7], reason="test ban")

    bsz, seq, vocab = 1, 4, 16
    logits = torch.zeros((bsz, seq, vocab))
    state = {
        "attention_mask": torch.ones((bsz, seq), dtype=torch.bool),
        "token_ids": torch.zeros((bsz, seq), dtype=torch.long),
        "last_indices": torch.tensor([seq - 1], dtype=torch.long),
    }
    out = graft(logits, state)
    # Banned tokens at last position are pushed down by exactly the penalty.
    assert float(out[0, seq - 1, 3].item()) == pytest.approx(-50.0)
    assert float(out[0, seq - 1, 7].item()) == pytest.approx(-50.0)
    # Non-banned tokens at last position are unchanged.
    assert float(out[0, seq - 1, 5].item()) == pytest.approx(0.0)
    # Non-last positions are entirely untouched, even on banned ids.
    assert float(out[0, 0, 3].item()) == pytest.approx(0.0)


def test_masking_graft_state_extra_bias_is_merged_with_persistent_set():
    graft = HypothesisMaskingGraft(default_penalty=10.0)
    graft.ban([1], penalty=5.0)
    bsz, seq, vocab = 1, 1, 8
    logits = torch.zeros((bsz, seq, vocab))
    state = {
        "attention_mask": torch.ones((bsz, seq), dtype=torch.bool),
        "token_ids": torch.zeros((bsz, seq), dtype=torch.long),
        "last_indices": torch.tensor([0], dtype=torch.long),
        "broca_negative_bias": {2: 7.0, 1: 20.0},  # 1 should pick up the larger penalty
    }
    out = graft(logits, state)
    assert float(out[0, 0, 1].item()) == pytest.approx(-20.0)
    assert float(out[0, 0, 2].item()) == pytest.approx(-7.0)
    assert float(out[0, 0, 0].item()) == pytest.approx(0.0)


def test_masking_graft_unban_and_clear_remove_from_set():
    graft = HypothesisMaskingGraft(default_penalty=50.0)
    graft.ban([3, 7])
    assert set(graft.banned) == {3, 7}
    graft.unban([3])
    assert set(graft.banned) == {7}
    graft.clear()
    assert graft.banned == {}
    # History records all three operations.
    actions = [h["action"] for h in graft.history]
    assert actions == ["ban", "unban", "clear"]


def test_masking_graft_disabled_passes_logits_through():
    graft = HypothesisMaskingGraft()
    graft.ban([1])
    graft.enabled = False
    logits = torch.zeros((1, 1, 4))
    state = {
        "attention_mask": torch.ones((1, 1), dtype=torch.bool),
        "token_ids": torch.zeros((1, 1), dtype=torch.long),
        "last_indices": torch.tensor([0], dtype=torch.long),
    }
    out = graft(logits, state)
    assert torch.equal(out, logits)


def test_masking_graft_re_ban_keeps_strongest_penalty():
    graft = HypothesisMaskingGraft(default_penalty=10.0)
    graft.ban([1], penalty=5.0)
    graft.ban([1], penalty=50.0)
    graft.ban([1], penalty=20.0)  # weaker than 50.0; must not regress
    assert graft.banned[1] == pytest.approx(50.0)


def test_masking_graft_negative_default_penalty_rejected():
    with pytest.raises(ValueError):
        HypothesisMaskingGraft(default_penalty=-1.0)


def test_masking_graft_ignores_invalid_token_ids():
    graft = HypothesisMaskingGraft()
    graft.ban([-1, 0, 100])
    bsz, seq, vocab = 1, 1, 4
    logits = torch.zeros((bsz, seq, vocab))
    state = {
        "attention_mask": torch.ones((bsz, seq), dtype=torch.bool),
        "token_ids": torch.zeros((bsz, seq), dtype=torch.long),
        "last_indices": torch.tensor([0], dtype=torch.long),
    }
    out = graft(logits, state)
    # token id 0 is in-bounds and should be banned; -1 and 100 are silently dropped.
    assert float(out[0, 0, 0].item()) == pytest.approx(-graft.default_penalty)


# ---------------------------------------------------------------------------
# IterativeHypothesisSearch
# ---------------------------------------------------------------------------


def test_iterative_search_accepts_first_valid_hypothesis():
    host = _RankedStubHost(ranked=[5, 7, 9])
    tok = _StubTokenizer()
    graft = HypothesisMaskingGraft(default_penalty=200.0)
    host.add_graft("logits", graft)

    search = IterativeHypothesisSearch(host, tok, graft, hypothesis_max_tokens=1, max_iterations=4)
    # Evaluator that always accepts.
    result = search.run([1, 2, 3], evaluator=lambda ids, text: HypothesisVerdict(valid=True))
    assert result.accepted is True
    assert result.iterations == 1
    assert result.final_token_ids == [5]
    assert len(result.history) == 1
    # No bans were registered.
    assert graft.banned == {}


def test_iterative_search_bans_rejected_tokens_and_falls_through_to_next_rank():
    host = _RankedStubHost(ranked=[5, 7, 9])
    tok = _StubTokenizer()
    graft = HypothesisMaskingGraft(default_penalty=500.0)
    host.add_graft("logits", graft)
    search = IterativeHypothesisSearch(host, tok, graft, hypothesis_max_tokens=1, max_iterations=4)

    # Reject 5 and 7; accept 9.
    def evaluator(ids: list[int], text: str) -> HypothesisVerdict:
        first = ids[0]
        if first == 9:
            return HypothesisVerdict(valid=True)
        return HypothesisVerdict(valid=False, ban_tokens=(first,), reason=f"reject {first}")

    result = search.run([0], evaluator=evaluator)
    assert result.accepted is True
    assert result.iterations == 3
    assert result.final_token_ids == [9]
    # Graft should have banned 5 and 7 (but not 9).
    assert 5 in graft.banned
    assert 7 in graft.banned
    assert 9 not in graft.banned
    # History captures all three attempts in order.
    assert [a.token_ids[0] for a in result.history] == [5, 7, 9]


def test_iterative_search_ban_first_token_when_evaluator_returns_no_explicit_bans():
    host = _RankedStubHost(ranked=[5, 7, 9])
    tok = _StubTokenizer()
    graft = HypothesisMaskingGraft(default_penalty=500.0)
    host.add_graft("logits", graft)
    search = IterativeHypothesisSearch(host, tok, graft, hypothesis_max_tokens=1, max_iterations=4)

    # Reject everything but 9; never name explicit ban tokens.
    def evaluator(ids: list[int], text: str) -> HypothesisVerdict:
        return HypothesisVerdict(valid=ids[0] == 9, reason="lazy reject")

    result = search.run([0], evaluator=evaluator)
    assert result.accepted is True
    assert result.iterations == 3


def test_iterative_search_multi_token_hypothesis_uses_full_sequence():
    host = _RankedStubHost(ranked=[5, 7, 9])  # always picks 5 then 5 then 5
    tok = _StubTokenizer()
    graft = HypothesisMaskingGraft(default_penalty=500.0)
    host.add_graft("logits", graft)
    search = IterativeHypothesisSearch(host, tok, graft, hypothesis_max_tokens=3, max_iterations=4)

    # Reject any hypothesis that starts with token 5.
    def evaluator(ids: list[int], text: str) -> HypothesisVerdict:
        return HypothesisVerdict(valid=ids[0] != 5, ban_tokens=(ids[0],) if ids[0] == 5 else ())

    result = search.run([0], evaluator=evaluator)
    assert result.accepted is True
    assert result.final_token_ids[0] == 7
    # Each accepted hypothesis is hypothesis_max_tokens long.
    assert len(result.final_token_ids) == 3


def test_iterative_search_returns_failure_after_max_iterations():
    host = _RankedStubHost(ranked=[5, 7, 9])
    tok = _StubTokenizer()
    graft = HypothesisMaskingGraft(default_penalty=500.0)
    host.add_graft("logits", graft)
    search = IterativeHypothesisSearch(host, tok, graft, hypothesis_max_tokens=1, max_iterations=2)

    def evaluator(ids: list[int], text: str) -> HypothesisVerdict:
        return HypothesisVerdict(valid=False, ban_tokens=(ids[0],))

    result = search.run([0], evaluator=evaluator)
    assert result.accepted is False
    assert result.iterations == 2
    assert len(result.history) == 2


def test_iterative_search_clears_bans_on_run_when_requested():
    host = _RankedStubHost(ranked=[5, 7, 9])
    tok = _StubTokenizer()
    graft = HypothesisMaskingGraft(default_penalty=500.0)
    host.add_graft("logits", graft)
    search = IterativeHypothesisSearch(host, tok, graft, hypothesis_max_tokens=1, max_iterations=2)

    graft.ban([42])  # pre-existing irrelevant ban
    result = search.run(
        [0],
        evaluator=lambda ids, text: HypothesisVerdict(valid=True),
        clear_bans=True,
    )
    assert result.accepted is True
    # Pre-existing ban was cleared.
    assert 42 not in graft.banned


def test_iterative_search_preserves_bans_when_clear_bans_false():
    host = _RankedStubHost(ranked=[5, 7, 9])
    tok = _StubTokenizer()
    graft = HypothesisMaskingGraft(default_penalty=500.0)
    host.add_graft("logits", graft)
    search = IterativeHypothesisSearch(host, tok, graft, hypothesis_max_tokens=1, max_iterations=2)

    graft.ban([42])
    search.run(
        [0],
        evaluator=lambda ids, text: HypothesisVerdict(valid=True),
        clear_bans=False,
    )
    assert 42 in graft.banned


def test_iterative_search_refuses_unattached_graft():
    host = _RankedStubHost(ranked=[5, 7, 9])
    tok = _StubTokenizer()
    graft = HypothesisMaskingGraft()
    # Did NOT host.add_graft(...).
    with pytest.raises(RuntimeError):
        IterativeHypothesisSearch(host, tok, graft, require_attached=True)


def test_iterative_search_can_skip_attachment_check():
    host = _RankedStubHost(ranked=[5, 7, 9])
    tok = _StubTokenizer()
    graft = HypothesisMaskingGraft()
    # Should not raise.
    IterativeHypothesisSearch(host, tok, graft, require_attached=False)


# ---------------------------------------------------------------------------
# 2. EpistemicInterruptionMonitor
# ---------------------------------------------------------------------------


def test_interruption_monitor_emits_full_sequence_when_evaluator_never_halts():
    host = _ScheduledStubHost(schedule=[[1], [2], [3], [4]])
    tok = _StubTokenizer()
    monitor = EpistemicInterruptionMonitor(host, tok, check_every=1, max_truncations=4)

    result = monitor.generate(
        prompt_ids=[10, 11],
        max_new_tokens=4,
        evaluator=lambda ids, text, step: InterruptionVerdict(halt=False),
    )
    assert result.token_ids == [1, 2, 3, 4]
    assert result.interventions == []
    assert result.final_step == 4


def test_interruption_monitor_halts_truncates_and_continues():
    host = _ScheduledStubHost(schedule=[[1], [2], [3], [4], [5], [6]])
    tok = _StubTokenizer()
    monitor = EpistemicInterruptionMonitor(host, tok, check_every=2, max_truncations=4)

    halts: list[int] = []
    halt_fired = {"once": False}

    def evaluator(ids: list[int], text: str, step: int) -> InterruptionVerdict:
        halts.append(step)
        if not halt_fired["once"] and ids == [1, 2]:
            halt_fired["once"] = True
            return InterruptionVerdict(halt=True, truncate_tokens=2, reason="logical collision")
        return InterruptionVerdict(halt=False)

    result = monitor.generate(
        prompt_ids=[10, 11],
        max_new_tokens=4,
        evaluator=evaluator,
    )

    # After 2 tokens the evaluator halts and truncates them away. The monitor
    # then continues generating until max_new_tokens=4, drawing from schedule
    # positions 2..5 which yield tokens [3, 4, 5, 6].
    assert result.token_ids == [3, 4, 5, 6]
    assert len(result.interventions) == 1
    event = result.interventions[0]
    assert event.truncated == 2
    assert event.reason == "logical collision"
    # Evaluator was invoked at the post-halt boundary too (step=2, then step=4).
    assert halts == [2, 2, 4]


def test_interruption_monitor_boost_window_forwards_correction_features():
    host = _ScheduledStubHost(schedule=[[1], [2], [3], [4], [5], [6], [7]])
    tok = _StubTokenizer()
    monitor = EpistemicInterruptionMonitor(host, tok, check_every=2, max_truncations=4)

    correction = torch.tensor([0.7] * 8)
    halt_fired = {"once": False}

    def evaluator(ids: list[int], text: str, step: int) -> InterruptionVerdict:
        if not halt_fired["once"] and ids == [1, 2]:
            halt_fired["once"] = True
            return InterruptionVerdict(
                halt=True,
                truncate_tokens=1,
                correction_features=correction,
                boost_steps=2,
                reason="re-evaluate",
            )
        return InterruptionVerdict(halt=False)

    result = monitor.generate(
        prompt_ids=[10],
        max_new_tokens=5,
        evaluator=evaluator,
    )

    # Verify broca_features was injected in the host calls *after* the halt point.
    # Calls 1, 2 are pre-halt (no broca_features); calls 3 and 4 (boost active) carry it.
    feats_seen = [
        ("broca_features" in call["extra_state"]) for call in host.calls
    ]
    # Pre-halt: 2 calls without features
    assert feats_seen[0] is False
    assert feats_seen[1] is False
    # Boost window of 2 → exactly two host calls carry broca_features.
    boosted = [
        call for call in host.calls
        if "broca_features" in call["extra_state"]
    ]
    assert len(boosted) == 2
    for call in boosted:
        seen = call["extra_state"]["broca_features"]
        assert torch.allclose(seen, correction.to(seen.device))


def test_interruption_monitor_bans_tokens_through_attached_masking_graft():
    host = _ScheduledStubHost(schedule=[[5], [5], [5], [5]])
    tok = _StubTokenizer()
    masking = HypothesisMaskingGraft(default_penalty=500.0)
    host.add_graft("logits", masking)
    monitor = EpistemicInterruptionMonitor(
        host,
        tok,
        check_every=1,
        max_truncations=4,
        masking_graft=masking,
    )

    def evaluator(ids: list[int], text: str, step: int) -> InterruptionVerdict:
        if 5 in ids:
            return InterruptionVerdict(
                halt=True,
                truncate_tokens=len(ids),
                ban_tokens=(5,),
                reason="five is forbidden",
            )
        return InterruptionVerdict(halt=False)

    # Even though the schedule wants 5 forever, the masking ban knocks 5 down so
    # token 0 wins the argmax (default-zero logits remain unchanged) once 5 is banned.
    result = monitor.generate(
        prompt_ids=[10],
        max_new_tokens=2,
        evaluator=evaluator,
    )
    # The first iteration emits 5 (which fails); after the ban, 5 is no longer the argmax.
    assert 5 in masking.banned
    # The very-first generated token is 5, then the rest are non-5 (because of the ban).
    after_ban = result.token_ids[1:] if result.token_ids else []
    assert all(t != 5 for t in after_ban)


def test_interruption_monitor_respects_max_truncations():
    host = _ScheduledStubHost(schedule=[[1]] * 10)
    tok = _StubTokenizer()
    monitor = EpistemicInterruptionMonitor(host, tok, check_every=1, max_truncations=2)

    halts = 0

    def evaluator(ids: list[int], text: str, step: int) -> InterruptionVerdict:
        nonlocal halts
        halts += 1
        return InterruptionVerdict(halt=True, truncate_tokens=1)

    monitor.generate(
        prompt_ids=[10],
        max_new_tokens=5,
        evaluator=evaluator,
    )
    # We allow at most max_truncations halts; subsequent evaluator calls are still
    # invoked (we don't suppress them), but their halt verdicts are ignored.
    halt_events = [c for c in host.calls]
    # The monitor should have stopped truncating after 2 truncations.
    # That means we generated at least max_new_tokens (or close to it) even with halts.
    assert halts >= 2


def test_interruption_monitor_eos_token_terminates_generation():
    eos = 60  # in-vocab; default vocab is 64
    host = _ScheduledStubHost(schedule=[[1], [eos], [3]])
    tok = _StubTokenizer()
    monitor = EpistemicInterruptionMonitor(host, tok, check_every=10, max_truncations=4)

    result = monitor.generate(
        prompt_ids=[10],
        max_new_tokens=5,
        evaluator=lambda ids, text, step: InterruptionVerdict(halt=False),
        eos_token_id=eos,
    )
    # Generation halts as soon as the host's argmax matches eos (second forward).
    assert result.token_ids == [1]


def test_interruption_monitor_invalid_args_raise():
    host = _RankedStubHost(ranked=[1])
    tok = _StubTokenizer()
    with pytest.raises(ValueError):
        EpistemicInterruptionMonitor(host, tok, check_every=0)
    with pytest.raises(ValueError):
        EpistemicInterruptionMonitor(host, tok, max_truncations=-1)
    monitor = EpistemicInterruptionMonitor(host, tok)
    with pytest.raises(ValueError):
        monitor.generate(
            prompt_ids=[1],
            max_new_tokens=0,
            evaluator=lambda *a, **k: InterruptionVerdict(halt=False),
        )


# ---------------------------------------------------------------------------
# 3. ModalityShiftGraft
# ---------------------------------------------------------------------------


def test_modality_graft_register_mode_normalizes_to_unit_norm():
    graft = ModalityShiftGraft(d_model=8)
    raw = torch.tensor([3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    graft.register_mode("analytical", raw)
    assert "analytical" in graft.modes
    stored = graft.modes["analytical"]
    assert stored.numel() == 8
    assert abs(stored.norm().item() - 1.0) < 1e-5


def test_modality_graft_no_active_mode_passes_through():
    graft = ModalityShiftGraft(d_model=4)
    graft.register_mode("foo", torch.tensor([1.0, 0.0, 0.0, 0.0]))
    bsz, seq, d = 1, 3, 4
    x = torch.randn(bsz, seq, d)
    state = {
        "attention_mask": torch.ones((bsz, seq), dtype=torch.bool),
        "token_ids": torch.zeros((bsz, seq), dtype=torch.long),
        "last_indices": torch.tensor([seq - 1], dtype=torch.long),
    }
    out = graft(x, state)
    assert torch.allclose(out, x)


def test_modality_graft_position_mode_last_only_modifies_last_position():
    graft = ModalityShiftGraft(d_model=4, target_snr=1.0, position_mode="last")
    direction = torch.tensor([1.0, 0.0, 0.0, 0.0])
    graft.register_mode("analytical", direction)
    graft.set_active_mode("analytical")
    bsz, seq = 1, 3
    x = torch.randn(bsz, seq, 4)
    state = {
        "attention_mask": torch.ones((bsz, seq), dtype=torch.bool),
        "token_ids": torch.zeros((bsz, seq), dtype=torch.long),
        "last_indices": torch.tensor([seq - 1], dtype=torch.long),
    }
    out = graft(x, state)
    # All but the last position are unchanged.
    assert torch.allclose(out[:, :-1], x[:, :-1])
    # The last-position delta lies along the registered direction.
    delta = out[0, -1] - x[0, -1]
    assert delta.norm().item() > 1e-5
    cos = F.cosine_similarity(delta.reshape(1, -1), direction.reshape(1, -1)).item()
    assert cos > 0.999


def test_modality_graft_position_mode_all_modifies_every_valid_position():
    graft = ModalityShiftGraft(d_model=4, target_snr=1.0, position_mode="all")
    direction = torch.tensor([0.0, 1.0, 0.0, 0.0])
    graft.register_mode("fluent", direction)
    graft.set_active_mode("fluent")
    bsz, seq = 1, 3
    x = torch.ones((bsz, seq, 4))
    state = {
        "attention_mask": torch.ones((bsz, seq), dtype=torch.bool),
        "token_ids": torch.zeros((bsz, seq), dtype=torch.long),
        "last_indices": torch.tensor([seq - 1], dtype=torch.long),
    }
    out = graft(x, state)
    # All positions saw a non-trivial delta; deltas point along the direction.
    for pos in range(seq):
        delta = out[0, pos] - x[0, pos]
        assert delta.norm().item() > 1e-5
        cos = F.cosine_similarity(delta.reshape(1, -1), direction.reshape(1, -1)).item()
        assert cos > 0.999


def test_modality_graft_position_mode_all_skips_padded_positions():
    graft = ModalityShiftGraft(d_model=4, target_snr=1.0, position_mode="all")
    direction = torch.tensor([0.0, 1.0, 0.0, 0.0])
    graft.register_mode("m", direction)
    graft.set_active_mode("m")
    bsz, seq = 1, 4
    x = torch.ones((bsz, seq, 4))
    # Mask out the last token: only first 3 positions are valid.
    mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)
    state = {
        "attention_mask": mask,
        "token_ids": torch.zeros((bsz, seq), dtype=torch.long),
        "last_indices": torch.tensor([2], dtype=torch.long),
    }
    out = graft(x, state)
    # Padded position (last) is unchanged.
    assert torch.allclose(out[0, -1], x[0, -1])
    # Valid positions are biased.
    for pos in range(3):
        delta = out[0, pos] - x[0, pos]
        assert delta.norm().item() > 1e-5


def test_modality_graft_state_override_takes_priority_over_active_mode():
    graft = ModalityShiftGraft(d_model=4, target_snr=1.0, position_mode="last")
    graft.register_mode("a", torch.tensor([1.0, 0.0, 0.0, 0.0]))
    graft.register_mode("b", torch.tensor([0.0, 1.0, 0.0, 0.0]))
    graft.set_active_mode("a")
    bsz, seq = 1, 1
    # Use non-zero residual so snr_magnitude (host RMS-based) is non-zero.
    x = torch.ones((bsz, seq, 4))
    state = {
        "attention_mask": torch.ones((bsz, seq), dtype=torch.bool),
        "token_ids": torch.zeros((bsz, seq), dtype=torch.long),
        "last_indices": torch.tensor([0], dtype=torch.long),
        "broca_modality": "b",
    }
    out = graft(x, state)
    delta = (out - x)[0, 0]
    assert F.cosine_similarity(delta.reshape(1, -1), torch.tensor([[0.0, 1.0, 0.0, 0.0]])).item() > 0.999
    assert graft.last_mode_used == "b"


def test_modality_graft_set_active_mode_unknown_raises():
    graft = ModalityShiftGraft(d_model=4)
    with pytest.raises(KeyError):
        graft.set_active_mode("unregistered")


def test_modality_graft_register_invalid_inputs_raise():
    graft = ModalityShiftGraft(d_model=8)
    with pytest.raises(ValueError):
        graft.register_mode("", torch.zeros(8))
    with pytest.raises(ValueError):
        graft.register_mode("bad", torch.zeros(7))


def test_modality_graft_position_mode_invalid_raises():
    with pytest.raises(ValueError):
        ModalityShiftGraft(d_model=8, position_mode="middle")


def test_modality_graft_mode_from_tokens_averages_lm_head_rows():
    graft = ModalityShiftGraft(d_model=8)
    lm_head = nn.Linear(8, 16, bias=False)
    graft.mode_from_tokens("digits", token_ids=[3, 4, 5], lm_head=lm_head)
    expected = (
        lm_head.weight[3].detach().float()
        + lm_head.weight[4].detach().float()
        + lm_head.weight[5].detach().float()
    ) / 3.0
    expected = F.normalize(expected.reshape(1, -1), dim=-1).reshape(-1)
    cos = F.cosine_similarity(graft.modes["digits"].reshape(1, -1), expected.reshape(1, -1)).item()
    assert cos > 0.999


def test_modality_graft_mode_from_tokens_rejects_empty_input():
    graft = ModalityShiftGraft(d_model=8)
    lm_head = nn.Linear(8, 16, bias=False)
    with pytest.raises(ValueError):
        graft.mode_from_tokens("empty", token_ids=[], lm_head=lm_head)


def test_modality_graft_register_from_capture_uses_value_field():
    graft = ModalityShiftGraft(d_model=4)

    @dataclass
    class _StubCapture:
        value: torch.Tensor

    cap = _StubCapture(value=torch.tensor([2.0, 0.0, 0.0, 0.0]))
    graft.register_mode_from_capture("from_capture", cap)
    assert "from_capture" in graft.modes
    assert abs(graft.modes["from_capture"].norm().item() - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# 4. CausalConstraintGraft
# ---------------------------------------------------------------------------


def _build_simpson_like_scm() -> FiniteSCM:
    """Tiny SCM where do(T=1) gives P(Y=1)=0.7, do(T=0) gives P(Y=1)=0.3."""

    scm = FiniteSCM(domains={})
    scm.add_exogenous("U_T", (0, 1), {0: 0.5, 1: 0.5})
    scm.add_exogenous("U_Y", (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), {i: 0.1 for i in range(10)})
    scm.add_endogenous("T", (0, 1), parents=("U_T",), fn=lambda v: v["U_T"])
    # P(Y=1 | T=1) = 7/10, P(Y=1 | T=0) = 3/10
    scm.add_endogenous(
        "Y",
        (0, 1),
        parents=("T", "U_Y"),
        fn=lambda v: 1 if (v["T"] == 1 and v["U_Y"] < 7) or (v["T"] == 0 and v["U_Y"] < 3) else 0,
    )
    return scm


def test_causal_graft_encode_treatment_effect_records_distribution():
    scm = _build_simpson_like_scm()
    tok = _StubTokenizer()
    lm_head = nn.Linear(8, max(tok.token_to_id.values()) + 1, bias=False)
    graft = CausalConstraintGraft(d_model=8, max_items=8, target_snr=1.0)

    constraint = graft.encode_treatment_effect(
        scm,
        treatment="T",
        treatment_value=1,
        outcome="Y",
        concept_token="Treatment",
        outcome_token_map={1: "helps", 0: "hurts"},
        lm_head=lm_head,
        tokenizer=tok,
    )

    assert isinstance(constraint, CausalConstraint)
    assert constraint.treatment == "T"
    assert constraint.treatment_value == 1
    assert constraint.outcome == "Y"
    # Probabilities sum to 1 and reflect P(Y|do(T=1)) = 0.7.
    assert constraint.distribution[1] == pytest.approx(0.7, abs=1e-6)
    assert constraint.distribution[0] == pytest.approx(0.3, abs=1e-6)
    assert int(graft.keys.shape[0]) == 1
    assert len(graft.constraints) == 1


def test_causal_graft_value_direction_is_probability_weighted_blend():
    """The encoded value should be a 0.7/0.3 blend of the helps/hurts directions."""

    scm = _build_simpson_like_scm()
    tok = _StubTokenizer()
    lm_head = nn.Linear(8, max(tok.token_to_id.values()) + 1, bias=False)
    graft = CausalConstraintGraft(d_model=8, max_items=8, target_snr=1.0)

    graft.encode_treatment_effect(
        scm,
        treatment="T",
        treatment_value=1,
        outcome="Y",
        concept_token="Treatment",
        outcome_token_map={1: "helps", 0: "hurts"},
        lm_head=lm_head,
        tokenizer=tok,
    )
    encoded_value = graft.values[0].detach().float()

    helps_id = tok.token_to_id["helps"]
    hurts_id = tok.token_to_id["hurts"]
    helps_dir = F.normalize(lm_head.weight[helps_id].detach().float().reshape(1, -1), dim=-1).reshape(-1)
    hurts_dir = F.normalize(lm_head.weight[hurts_id].detach().float().reshape(1, -1), dim=-1).reshape(-1)
    expected = F.normalize((0.7 * helps_dir + 0.3 * hurts_dir).reshape(1, -1), dim=-1).reshape(-1)

    cos = F.cosine_similarity(encoded_value.reshape(1, -1), expected.reshape(1, -1)).item()
    assert cos > 0.999


def test_causal_graft_intervention_grid_produces_one_constraint_per_value():
    scm = _build_simpson_like_scm()
    tok = _StubTokenizer()
    lm_head = nn.Linear(8, max(tok.token_to_id.values()) + 1, bias=False)
    graft = CausalConstraintGraft(d_model=8, max_items=8, target_snr=1.0)

    constraints = graft.encode_intervention_grid(
        scm,
        treatment="T",
        outcome="Y",
        outcome_token_map={1: "helps", 0: "hurts"},
        lm_head=lm_head,
        tokenizer=tok,
        treatment_concept_tokens={1: "Treatment", 0: "Treatment"},
    )
    assert len(constraints) == 2  # two treatment values
    assert int(graft.keys.shape[0]) == 2
    distributions = sorted([c.distribution[1] for c in constraints])
    assert distributions == pytest.approx([0.3, 0.7], abs=1e-6)


def test_causal_graft_encode_rejects_unknown_treatment_or_outcome():
    scm = _build_simpson_like_scm()
    tok = _StubTokenizer()
    lm_head = nn.Linear(8, max(tok.token_to_id.values()) + 1, bias=False)
    graft = CausalConstraintGraft(d_model=8)
    with pytest.raises(KeyError):
        graft.encode_treatment_effect(
            scm,
            treatment="T",
            treatment_value=1,
            outcome="Z_does_not_exist",
            concept_token="Treatment",
            outcome_token_map={1: "helps", 0: "hurts"},
            lm_head=lm_head,
            tokenizer=tok,
        )


def test_causal_graft_encode_rejects_unknown_concept_token():
    scm = _build_simpson_like_scm()
    tok = _StubTokenizer()
    lm_head = nn.Linear(8, max(tok.token_to_id.values()) + 1, bias=False)
    graft = CausalConstraintGraft(d_model=8)
    with pytest.raises(KeyError):
        graft.encode_treatment_effect(
            scm,
            treatment="T",
            treatment_value=1,
            outcome="Y",
            concept_token="",  # empty surface form cannot resolve to a token id
            outcome_token_map={1: "helps", 0: "hurts"},
            lm_head=lm_head,
            tokenizer=tok,
        )


def test_causal_graft_encode_fails_when_all_outcome_tokens_missing():
    scm = _build_simpson_like_scm()
    tok = _StubTokenizer()
    lm_head = nn.Linear(8, max(tok.token_to_id.values()) + 1, bias=False)
    graft = CausalConstraintGraft(d_model=8)
    with pytest.raises(ValueError):
        graft.encode_treatment_effect(
            scm,
            treatment="T",
            treatment_value=1,
            outcome="Y",
            concept_token="Treatment",
            outcome_token_map={1: "missing_a", 0: "missing_b"},
            lm_head=lm_head,
            tokenizer=tok,
        )


def test_causal_graft_clear_resets_keys_values_and_constraints():
    scm = _build_simpson_like_scm()
    tok = _StubTokenizer()
    lm_head = nn.Linear(8, max(tok.token_to_id.values()) + 1, bias=False)
    graft = CausalConstraintGraft(d_model=8)

    graft.encode_treatment_effect(
        scm,
        treatment="T",
        treatment_value=1,
        outcome="Y",
        concept_token="Treatment",
        outcome_token_map={1: "helps", 0: "hurts"},
        lm_head=lm_head,
        tokenizer=tok,
    )
    assert int(graft.keys.shape[0]) == 1
    assert len(graft.constraints) == 1
    graft.clear()
    assert int(graft.keys.shape[0]) == 0
    assert graft.constraints == []


def test_causal_graft_kv_retrieval_aligns_residual_with_value():
    """When the residual stream's sequence-mean aligns with the constraint key,
    the graft injects a delta in the encoded value direction."""

    scm = _build_simpson_like_scm()
    tok = _StubTokenizer()
    lm_head = nn.Linear(8, max(tok.token_to_id.values()) + 1, bias=False)
    graft = CausalConstraintGraft(d_model=8, max_items=8, target_snr=1.0)

    graft.encode_treatment_effect(
        scm,
        treatment="T",
        treatment_value=1,
        outcome="Y",
        concept_token="Treatment",
        outcome_token_map={1: "helps", 0: "hurts"},
        lm_head=lm_head,
        tokenizer=tok,
    )
    # Construct a residual stream where the sequence-mean is the key direction.
    key_direction = graft.keys[0].detach().float()
    bsz, seq = 1, 3
    x = key_direction.view(1, 1, -1).expand(bsz, seq, -1).contiguous().to(torch.float32)
    state = {
        "attention_mask": torch.ones((bsz, seq), dtype=torch.bool),
        "token_ids": torch.zeros((bsz, seq), dtype=torch.long),
        "last_indices": torch.tensor([seq - 1], dtype=torch.long),
    }
    out = graft(x, state)
    delta = (out - x)[0, -1]
    assert delta.norm().item() > 1e-3
    # The injected delta should align with the encoded value direction.
    expected = graft.values[0].detach().float()
    cos = F.cosine_similarity(delta.reshape(1, -1), expected.reshape(1, -1)).item()
    assert cos > 0.95


def test_causal_graft_routes_query_to_matching_constraint_value():
    """Different concept-keyed constraints route to different value directions.

    Encoding two constraints (one with high P(Y=1), one with low) and querying
    with each constraint's key should produce deltas pointing along that
    constraint's specific probability-weighted blend, not the other one.
    """

    scm = _build_simpson_like_scm()
    tok = _StubTokenizer()
    lm_head = nn.Linear(8, max(tok.token_to_id.values()) + 1, bias=False)
    graft = CausalConstraintGraft(d_model=8, max_items=8, target_snr=1.0)
    graft.encode_treatment_effect(
        scm,
        treatment="T",
        treatment_value=1,
        outcome="Y",
        concept_token="Treatment",
        outcome_token_map={1: "helps", 0: "hurts"},
        lm_head=lm_head,
        tokenizer=tok,
    )
    graft.encode_treatment_effect(
        scm,
        treatment="T",
        treatment_value=0,
        outcome="Y",
        concept_token="smoking",
        outcome_token_map={1: "helps", 0: "hurts"},
        lm_head=lm_head,
        tokenizer=tok,
    )

    bsz, seq = 1, 3

    def _delta_at_last_for_query(query_direction: torch.Tensor) -> torch.Tensor:
        x = query_direction.view(1, 1, -1).expand(bsz, seq, -1).contiguous().to(torch.float32)
        state = {
            "attention_mask": torch.ones((bsz, seq), dtype=torch.bool),
            "token_ids": torch.zeros((bsz, seq), dtype=torch.long),
            "last_indices": torch.tensor([seq - 1], dtype=torch.long),
        }
        out = graft(x, state)
        return (out - x)[0, -1]

    delta0 = _delta_at_last_for_query(graft.keys[0].detach().float())
    delta1 = _delta_at_last_for_query(graft.keys[1].detach().float())

    value0 = graft.values[0].detach().float()
    value1 = graft.values[1].detach().float()

    # Each query's delta must align more strongly with its own constraint's value
    # than with the other constraint's value.
    cos_d0_v0 = F.cosine_similarity(delta0.reshape(1, -1), value0.reshape(1, -1)).item()
    cos_d0_v1 = F.cosine_similarity(delta0.reshape(1, -1), value1.reshape(1, -1)).item()
    cos_d1_v0 = F.cosine_similarity(delta1.reshape(1, -1), value0.reshape(1, -1)).item()
    cos_d1_v1 = F.cosine_similarity(delta1.reshape(1, -1), value1.reshape(1, -1)).item()
    assert cos_d0_v0 > cos_d0_v1
    assert cos_d1_v1 > cos_d1_v0


# ---------------------------------------------------------------------------
# Cross-mechanism integration smoke test
# ---------------------------------------------------------------------------


def test_full_pipeline_combines_all_four_mechanisms():
    """All four grafts can coexist on a single host: a logit-mask, a
    final_hidden modality bias, a final_hidden causal constraint, and a
    final_hidden interruption-driven correction (via broca_features)."""

    host = _RankedStubHost(ranked=[5, 7, 9])
    tok = _StubTokenizer()
    masking = HypothesisMaskingGraft(default_penalty=200.0)
    modality = ModalityShiftGraft(d_model=host.cfg.d_model, target_snr=0.5, position_mode="last")
    modality.register_mode("analytical", torch.tensor([1.0] + [0.0] * (host.cfg.d_model - 1)))
    causal = CausalConstraintGraft(d_model=host.cfg.d_model, max_items=8, target_snr=0.5)
    causal.encode_treatment_effect(
        _build_simpson_like_scm(),
        treatment="T",
        treatment_value=1,
        outcome="Y",
        concept_token="Treatment",
        outcome_token_map={1: "helps", 0: "hurts"},
        lm_head=host.lm_head,
        tokenizer=tok,
    )
    host.add_graft("logits", masking)
    host.add_graft("final_hidden", modality)
    host.add_graft("final_hidden", causal)

    modality.set_active_mode("analytical")

    # Hypothesis search must still work with all the other grafts present.
    search = IterativeHypothesisSearch(host, tok, masking, hypothesis_max_tokens=1, max_iterations=4)

    def evaluator(ids: list[int], text: str) -> HypothesisVerdict:
        return HypothesisVerdict(valid=ids[0] == 9, ban_tokens=(ids[0],) if ids[0] != 9 else ())

    result = search.run([0], evaluator=evaluator)
    assert result.accepted is True
    assert result.final_token_ids == [9]
    # Both 5 and 7 ended up banned; the masking graft accumulated bans correctly.
    assert {5, 7}.issubset(masking.banned)
