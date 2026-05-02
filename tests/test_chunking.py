"""Tests for the DMN Chunking Compiler.

These tests exercise motif detection, registry persistence, and the running
mean-vector update without loading any LLM. A tiny in-memory ``mind`` stub
provides the journal surface (``recent``) and a ``text_encoder`` slot.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import torch

from core.idletime.chunking import (
    ChunkingDetectionConfig,
    CompiledMacro,
    DMNChunkingCompiler,
    MacroChunkRegistry,
    macro_frame_features,
    _macro_name_for_pattern,
)
from core.frame.continuous_frame import BROCA_FEATURE_DIM


class _StubJournal:
    def __init__(self, rows: list[dict]):
        self._rows = list(rows)

    def recent(self, *, limit: int) -> list[dict]:
        return list(self._rows[-int(limit):])


class _StubMind:
    def __init__(self, rows: list[dict]):
        self.journal = _StubJournal(rows)
        self.text_encoder = None


class _StubHawkes:
    baseline = 0.05

    def intensity(self, channel: str) -> float:
        return 10.0 if channel == "stress_episode" else 0.06


class _StubMindSalience(_StubMind):
    def __init__(self, rows: list[dict]):
        super().__init__(rows)
        self.hawkes = _StubHawkes()


def _row(
    jid: int,
    intent: str,
    *,
    subject: str = "subject",
    answer: str = "answer",
    confidence: float = 0.7,
    evidence: dict | None = None,
) -> dict:
    ev = {"journal_id": jid}
    if evidence:
        ev = {**evidence, **ev}
    ev["journal_id"] = jid
    return {
        "id": jid,
        "ts": time.time(),
        "utterance": f"u_{jid}",
        "intent": intent,
        "subject": subject,
        "answer": answer,
        "confidence": confidence,
        "evidence": ev,
    }


def test_compiler_salience_path_compiles_without_frequency_repetition(tmp_path):
    rows = [
        _row(1, "stress_episode", evidence={"lexical_surprise_gap": 1.5}),
        _row(2, "shutdown_loop"),
    ]
    db = tmp_path / "broca.sqlite"
    reg = MacroChunkRegistry(db, namespace="dmn")
    mind = _StubMindSalience(rows)
    compiler = DMNChunkingCompiler(
        mind,
        registry=reg,
        config=ChunkingDetectionConfig(
            window_size=32,
            min_motif_length=2,
            max_motif_length=2,
            min_repetitions=99,
            salience_oneshot_threshold=8.0,
            max_macros_per_tick=4,
        ),
    )
    result = compiler.run_once()
    assert result["compiled"] == 1
    macro = reg.get("macro_stress_episode__shutdown_loop")
    assert macro is not None
    assert macro.pattern == ("stress_episode", "shutdown_loop")
    assert result.get("reflections") is not None
    assert len(result["reflections"]) > 0
    assert result["reflections"][0].get("compile_via") == "salience"


# --- find_repeated_motifs --------------------------------------------------


def test_find_repeated_motifs_detects_simple_pattern():
    intents = [
        "memory_lookup", "causal_effect",
        "memory_lookup", "causal_effect",
        "memory_lookup", "causal_effect",
        "active_action",
    ]
    motifs = DMNChunkingCompiler.find_repeated_motifs(
        intents,
        min_motif_length=2,
        max_motif_length=2,
        min_repetitions=3,
    )
    assert any(pat == ("memory_lookup", "causal_effect") and len(starts) == 3 for pat, starts in motifs)


def test_find_repeated_motifs_uses_non_overlapping_count():
    # Greedy left-to-right non-overlap: AAAA contains 2 of "AA", not 3.
    intents = ["a", "a", "a", "a"]
    motifs = DMNChunkingCompiler.find_repeated_motifs(
        intents,
        min_motif_length=2,
        max_motif_length=2,
        min_repetitions=2,
    )
    assert motifs, "expected at least one motif"
    pat, starts = motifs[0]
    assert pat == ("a", "a")
    assert len(starts) == 2  # non-overlapping


def test_find_repeated_motifs_skips_all_unknown_windows():
    intents = ["unknown", "unknown", "unknown", "unknown"]
    motifs = DMNChunkingCompiler.find_repeated_motifs(
        intents,
        min_motif_length=2,
        max_motif_length=2,
        min_repetitions=2,
    )
    assert motifs == [], "all-unknown windows should not produce motifs"


def test_find_repeated_motifs_short_input_yields_nothing():
    motifs = DMNChunkingCompiler.find_repeated_motifs(
        ["a"],
        min_motif_length=2,
        max_motif_length=4,
        min_repetitions=2,
    )
    assert motifs == []


def test_find_repeated_motifs_prefers_long_patterns_first():
    intents = ["a", "b", "c", "a", "b", "c", "a", "b", "c"]
    motifs = DMNChunkingCompiler.find_repeated_motifs(
        intents,
        min_motif_length=2,
        max_motif_length=3,
        min_repetitions=2,
    )
    # Longest motifs reported first (descending L), so the first hit must be length 3.
    assert motifs and len(motifs[0][0]) == 3
    assert motifs[0][0] == ("a", "b", "c")


# --- macro registry --------------------------------------------------------


def test_registry_upsert_and_get_round_trip(tmp_path):
    db = tmp_path / "macros.sqlite"
    reg = MacroChunkRegistry(db, namespace="test")
    feat = torch.arange(8, dtype=torch.float32)
    macro = CompiledMacro(
        name="macro_a__b",
        pattern=("a", "b"),
        observation_count=3,
        avg_confidence=0.42,
        feature_vector=feat.clone(),
        last_seen_at=time.time(),
        member_episodes=[1, 2, 3, 4, 5, 6],
    )
    rid = reg.upsert(macro)
    assert isinstance(rid, int) and rid > 0
    fetched = reg.get("macro_a__b")
    assert fetched is not None
    assert fetched.pattern == ("a", "b")
    assert fetched.observation_count == 3
    assert torch.allclose(fetched.feature_vector, feat)
    assert fetched.member_episodes == [1, 2, 3, 4, 5, 6]


def test_registry_upsert_running_mean_blends_observations(tmp_path):
    """Re-upserting the same macro should blend feature vectors via a count-weighted mean."""

    db = tmp_path / "macros.sqlite"
    reg = MacroChunkRegistry(db, namespace="test")
    pattern = ("a", "b")
    name = _macro_name_for_pattern(pattern)
    feat1 = torch.zeros(8, dtype=torch.float32)
    feat2 = torch.ones(8, dtype=torch.float32)
    reg.upsert(
        CompiledMacro(
            name=name,
            pattern=pattern,
            observation_count=2,
            avg_confidence=0.4,
            feature_vector=feat1,
            last_seen_at=time.time(),
            member_episodes=[1, 2],
        )
    )
    reg.upsert(
        CompiledMacro(
            name=name,
            pattern=pattern,
            observation_count=2,
            avg_confidence=0.8,
            feature_vector=feat2,
            last_seen_at=time.time(),
            member_episodes=[3, 4],
        )
    )
    merged = reg.get(name)
    assert merged is not None
    # Equal weights → mean of zeros and ones is 0.5 across the board.
    assert torch.allclose(merged.feature_vector, torch.full((8,), 0.5))
    # Total observations accumulate.
    assert merged.observation_count == 4
    # Confidence is the count-weighted blend.
    assert abs(merged.avg_confidence - 0.6) < 1e-6
    # Member episode set unions cleanly.
    assert sorted(merged.member_episodes) == [1, 2, 3, 4]


def test_registry_remove_and_count(tmp_path):
    db = tmp_path / "macros.sqlite"
    reg = MacroChunkRegistry(db, namespace="test")
    macro = CompiledMacro(
        name="m",
        pattern=("x", "y"),
        observation_count=3,
        avg_confidence=0.5,
        feature_vector=torch.zeros(4),
        last_seen_at=time.time(),
        member_episodes=[],
    )
    reg.upsert(macro)
    assert reg.count() == 1
    assert reg.remove("m") is True
    assert reg.count() == 0
    assert reg.remove("does_not_exist") is False


def test_registry_namespace_isolation(tmp_path):
    db = tmp_path / "macros.sqlite"
    reg_a = MacroChunkRegistry(db, namespace="ns_a")
    reg_b = MacroChunkRegistry(db, namespace="ns_b")
    reg_a.upsert(
        CompiledMacro(
            name="m", pattern=("x", "y"),
            observation_count=2, avg_confidence=0.5,
            feature_vector=torch.zeros(2),
            last_seen_at=time.time(), member_episodes=[],
        )
    )
    assert reg_a.count() == 1
    assert reg_b.count() == 0


# --- prefix matching -------------------------------------------------------


def test_macro_matches_prefix_picks_most_observed(tmp_path):
    db = tmp_path / "macros.sqlite"
    reg = MacroChunkRegistry(db, namespace="t")
    reg.upsert(
        CompiledMacro(
            name=_macro_name_for_pattern(("a", "b", "c")),
            pattern=("a", "b", "c"),
            observation_count=2,
            avg_confidence=0.4,
            feature_vector=torch.zeros(4),
            last_seen_at=time.time(),
            member_episodes=[],
        )
    )
    reg.upsert(
        CompiledMacro(
            name=_macro_name_for_pattern(("a", "b", "d")),
            pattern=("a", "b", "d"),
            observation_count=10,  # more observations
            avg_confidence=0.8,
            feature_vector=torch.zeros(4),
            last_seen_at=time.time(),
            member_episodes=[],
        )
    )
    match = reg.find_macro_matching_prefix(["a", "b"])
    assert match is not None
    assert match.pattern == ("a", "b", "d")
    assert match.predicted_next_intent() == "d"


def test_macro_no_prefix_match_returns_none(tmp_path):
    db = tmp_path / "macros.sqlite"
    reg = MacroChunkRegistry(db, namespace="t")
    reg.upsert(
        CompiledMacro(
            name="foo",
            pattern=("x", "y"),
            observation_count=2,
            avg_confidence=0.4,
            feature_vector=torch.zeros(4),
            last_seen_at=time.time(),
            member_episodes=[],
        )
    )
    assert reg.find_macro_matching_prefix(["a", "b"]) is None


def test_macro_singleton_pattern_never_matches(tmp_path):
    db = tmp_path / "macros.sqlite"
    reg = MacroChunkRegistry(db, namespace="t")
    reg.upsert(
        CompiledMacro(
            name="solo",
            pattern=("a",),
            observation_count=10,
            avg_confidence=0.9,
            feature_vector=torch.zeros(4),
            last_seen_at=time.time(),
            member_episodes=[],
        )
    )
    # A length-1 pattern has no prefix to match against.
    assert reg.find_macro_matching_prefix(["a"]) is None


# --- end-to-end DMN compiler tick ------------------------------------------


def test_compiler_run_once_compiles_repeated_motif(tmp_path):
    rows = []
    for i in range(6):
        if i % 2 == 0:
            rows.append(_row(i, "memory_lookup", confidence=0.6))
        else:
            rows.append(_row(i, "causal_effect", confidence=0.8))
    rows.append(_row(99, "active_action", confidence=0.5))

    db = tmp_path / "broca.sqlite"
    reg = MacroChunkRegistry(db, namespace="dmn")
    mind = _StubMind(rows)
    compiler = DMNChunkingCompiler(
        mind,
        registry=reg,
        config=ChunkingDetectionConfig(
            window_size=32, min_motif_length=2, max_motif_length=2, min_repetitions=3, max_macros_per_tick=4
        ),
    )

    result = compiler.run_once()
    assert result["compiled"] >= 1
    macros = reg.all_macros()
    names = [m.name for m in macros]
    assert any(m.pattern == ("memory_lookup", "causal_effect") for m in macros), names
    macro = next(m for m in macros if m.pattern == ("memory_lookup", "causal_effect"))
    assert macro.observation_count == 3
    assert macro.feature_vector.numel() > 0
    # Reflections should be emitted with kind == "chunk_compiled".
    kinds = [r.get("kind") for r in result["reflections"]]
    assert "chunk_compiled" in kinds


def test_compiler_run_once_does_nothing_below_threshold(tmp_path):
    rows = [_row(0, "memory_lookup")]
    db = tmp_path / "broca.sqlite"
    reg = MacroChunkRegistry(db, namespace="dmn")
    mind = _StubMind(rows)
    compiler = DMNChunkingCompiler(
        mind,
        registry=reg,
        config=ChunkingDetectionConfig(min_motif_length=2, min_repetitions=3),
    )
    result = compiler.run_once()
    assert result["compiled"] == 0
    assert reg.count() == 0


def test_compiler_run_once_caps_at_max_macros_per_tick(tmp_path):
    # Construct two distinct motifs each repeating 3+ times; cap=1.
    rows = []
    jid = 0
    for _ in range(3):
        rows.append(_row(jid, "intent_a", confidence=0.5)); jid += 1
        rows.append(_row(jid, "intent_b", confidence=0.5)); jid += 1
    for _ in range(3):
        rows.append(_row(jid, "intent_c", confidence=0.5)); jid += 1
        rows.append(_row(jid, "intent_d", confidence=0.5)); jid += 1

    db = tmp_path / "broca.sqlite"
    reg = MacroChunkRegistry(db, namespace="dmn")
    mind = _StubMind(rows)
    compiler = DMNChunkingCompiler(
        mind,
        registry=reg,
        config=ChunkingDetectionConfig(
            window_size=32,
            min_motif_length=2,
            max_motif_length=2,
            min_repetitions=3,
            max_macros_per_tick=1,
        ),
    )
    result = compiler.run_once()
    assert result["compiled"] == 1
    assert reg.count() == 1


def test_macro_frame_features_pads_to_broca_dim():
    # A short feature vector should be zero-padded to BROCA_FEATURE_DIM.
    short = torch.tensor([1.0, 2.0, 3.0])
    macro = CompiledMacro(
        name="x",
        pattern=("a", "b"),
        observation_count=2,
        avg_confidence=0.4,
        feature_vector=short,
        last_seen_at=time.time(),
    )
    feats = macro_frame_features(macro)
    assert feats.shape == (BROCA_FEATURE_DIM,)
    # First three slots preserved.
    assert torch.allclose(feats[:3], short)
    # Tail is zero-padded.
    assert torch.allclose(feats[3:], torch.zeros(BROCA_FEATURE_DIM - 3))
