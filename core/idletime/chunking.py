"""DMN Chunking Compiler — proceduralization of repeated cognitive motifs.

The Default Mode Network notices that the same multi-step reasoning sequence
keeps firing across many episodes (e.g. ``memory_lookup`` → ``causal_effect``
→ ``active_action``) and *compiles* that motif into a single, reusable
"macro" cognitive frame.

Mathematically the compiled macro is the mean of the cognitive feature
vectors produced by every instance of the motif: a single point in
``COGNITIVE_FRAME_DIM`` space that captures the average "understanding" the
substrate had each time it walked the motif.  The macro is persisted to
SQLite so subsequent sessions inherit it.

When the user produces a new utterance, the substrate compares both recent
intent prefixes and continuous cognitive-frame geometry against the registry;
if a known macro matches, the substrate can short-circuit the slow multi-step
routing by injecting the macro's compiled feature vector directly into the
residual stream.  This is the "System 2 → System 1" transition.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..frame.continuous_frame import BROCA_FEATURE_DIM


logger = logging.getLogger(__name__)


def _vec_to_blob(v: torch.Tensor) -> tuple[bytes, int]:
    arr = v.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
    arr = np.ascontiguousarray(arr)
    return arr.tobytes(), int(arr.size)


def _blob_to_vec(blob: bytes, dim: int) -> torch.Tensor:
    arr = np.frombuffer(blob, dtype=np.float32, count=dim)
    return torch.from_numpy(arr.copy())


def _align_vec_to_dim(vec: torch.Tensor, dim: int) -> torch.Tensor:
    """Pad or truncate a 1-D feature vector to ``dim`` (same rules as :func:`macro_frame_features`)."""

    v = vec.detach().float().reshape(-1)
    n = v.numel()
    if n == dim:
        return v
    if n > dim:
        return v[:dim].clone()
    out = torch.zeros(dim, dtype=torch.float32)
    out[:n] = v
    return out


@dataclass
class CompiledMacro:
    """A proceduralized cognitive shortcut compiled from repeated motifs."""

    name: str
    pattern: tuple[str, ...]
    observation_count: int
    avg_confidence: float
    feature_vector: torch.Tensor
    last_seen_at: float
    member_episodes: list[int] = field(default_factory=list)
    id: int | None = None

    def matches_prefix(self, recent_intents: Sequence[str]) -> bool:
        """The macro's prefix matches whenever the last ``len(pattern)-1`` intents form the head of the pattern."""

        plen = len(self.pattern)
        if plen <= 1:
            return False
        head = self.pattern[: plen - 1]
        if len(recent_intents) < len(head):
            return False
        tail = tuple(recent_intents[-len(head) :])
        return tail == head

    def predicted_next_intent(self) -> str:
        return self.pattern[-1] if self.pattern else "unknown"


def _macro_name_for_pattern(pattern: Sequence[str]) -> str:
    """Stable identifier derived from the intent sequence."""

    cleaned = [str(p).strip().replace(" ", "_") or "unknown" for p in pattern]
    return "macro_" + "__".join(cleaned)


class MacroChunkRegistry:
    """SQLite-backed registry of proceduralized motifs.

    Lives in the same SQLite file as the rest of the substrate (e.g. ``broca_substrate.sqlite``)
    so a substrate handed to a fresh process inherits prior chunking.
    """

    def __init__(self, path: str | Path, *, namespace: str = "main"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.namespace = str(namespace)
        self._prefix_match_cache: dict[tuple[str, ...], CompiledMacro | None] = {}
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path)
        con.execute("PRAGMA journal_mode=WAL")
        return con

    def _init_schema(self) -> None:
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS macro_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    namespace TEXT NOT NULL,
                    name TEXT NOT NULL,
                    pattern_json TEXT NOT NULL,
                    observation_count INTEGER NOT NULL,
                    avg_confidence REAL NOT NULL,
                    feature_dim INTEGER NOT NULL,
                    feature_blob BLOB NOT NULL,
                    member_episodes_json TEXT NOT NULL,
                    last_seen_at REAL NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    UNIQUE(namespace, name)
                )
                """
            )
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_macro_chunks_namespace ON macro_chunks(namespace)"
            )

    def upsert(self, macro: CompiledMacro) -> int:
        blob, dim = _vec_to_blob(macro.feature_vector)
        now = time.time()
        with self._connect() as con:
            row = con.execute(
                "SELECT id, observation_count, avg_confidence, feature_dim, feature_blob, member_episodes_json, created_at "
                "FROM macro_chunks WHERE namespace=? AND name=?",
                (self.namespace, macro.name),
            ).fetchone()
            if row is None:
                cur = con.execute(
                    """
                    INSERT INTO macro_chunks(namespace, name, pattern_json, observation_count, avg_confidence,
                        feature_dim, feature_blob, member_episodes_json, last_seen_at, created_at, updated_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        self.namespace,
                        macro.name,
                        json.dumps(list(macro.pattern)),
                        int(macro.observation_count),
                        float(macro.avg_confidence),
                        int(dim),
                        blob,
                        json.dumps(list(macro.member_episodes)),
                        float(macro.last_seen_at),
                        now,
                        now,
                    ),
                )
                rid = int(cur.lastrowid)
            else:
                # Online running-mean update: blend the new observation with prior compiled state.
                rid = int(row[0])
                prior_count = int(row[1])
                prior_conf = float(row[2])
                prior_dim = int(row[3])
                prior_vec = _blob_to_vec(row[4], prior_dim)
                prior_episodes = json.loads(row[5])
                new_count = prior_count + int(macro.observation_count)
                # Treat ``observation_count`` as the count of episodes *this update* collapsed.
                # The merged mean is a weighted average of prior_vec and macro.feature_vector.
                w_old = float(prior_count) / float(max(1, new_count))
                w_new = float(macro.observation_count) / float(max(1, new_count))
                new_aligned = _align_vec_to_dim(macro.feature_vector, prior_dim)
                merged_vec = prior_vec * w_old + new_aligned * w_new
                merged_conf = prior_conf * w_old + float(macro.avg_confidence) * w_new
                merged_episodes_set: list[int] = list(prior_episodes)
                seen = set(int(e) for e in merged_episodes_set)
                for e in macro.member_episodes:
                    ei = int(e)
                    if ei not in seen:
                        seen.add(ei)
                        merged_episodes_set.append(ei)
                blob, dim = _vec_to_blob(merged_vec)
                con.execute(
                    """
                    UPDATE macro_chunks SET pattern_json=?, observation_count=?, avg_confidence=?,
                        feature_dim=?, feature_blob=?, member_episodes_json=?, last_seen_at=?, updated_at=?
                    WHERE id=?
                    """,
                    (
                        json.dumps(list(macro.pattern)),
                        int(new_count),
                        float(merged_conf),
                        int(dim),
                        blob,
                        json.dumps(merged_episodes_set),
                        float(macro.last_seen_at),
                        now,
                        rid,
                    ),
                )
            logger.debug(
                "MacroChunkRegistry.upsert: id=%d ns=%s name=%s pattern_len=%d count_step=%d",
                rid,
                self.namespace,
                macro.name,
                len(macro.pattern),
                macro.observation_count,
            )
            macro.id = rid
            self._prefix_match_cache.clear()
            return rid

    def all_macros(self) -> list[CompiledMacro]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT id, name, pattern_json, observation_count, avg_confidence, feature_dim, "
                "feature_blob, member_episodes_json, last_seen_at FROM macro_chunks "
                "WHERE namespace=? ORDER BY observation_count DESC, id ASC",
                (self.namespace,),
            ).fetchall()
        out: list[CompiledMacro] = []
        for row in rows:
            out.append(
                CompiledMacro(
                    id=int(row[0]),
                    name=str(row[1]),
                    pattern=tuple(json.loads(row[2])),
                    observation_count=int(row[3]),
                    avg_confidence=float(row[4]),
                    feature_vector=_blob_to_vec(row[6], int(row[5])),
                    member_episodes=[int(e) for e in json.loads(row[7])],
                    last_seen_at=float(row[8]),
                )
            )
        return out

    def get(self, name: str) -> CompiledMacro | None:
        with self._connect() as con:
            row = con.execute(
                "SELECT id, name, pattern_json, observation_count, avg_confidence, feature_dim, "
                "feature_blob, member_episodes_json, last_seen_at FROM macro_chunks "
                "WHERE namespace=? AND name=?",
                (self.namespace, name),
            ).fetchone()
        if row is None:
            return None
        return CompiledMacro(
            id=int(row[0]),
            name=str(row[1]),
            pattern=tuple(json.loads(row[2])),
            observation_count=int(row[3]),
            avg_confidence=float(row[4]),
            feature_vector=_blob_to_vec(row[6], int(row[5])),
            member_episodes=[int(e) for e in json.loads(row[7])],
            last_seen_at=float(row[8]),
        )

    def remove(self, name: str) -> bool:
        with self._connect() as con:
            cur = con.execute(
                "DELETE FROM macro_chunks WHERE namespace=? AND name=?",
                (self.namespace, name),
            )
        cleared = int(cur.rowcount or 0) > 0
        if cleared:
            self._prefix_match_cache.clear()
        return cleared

    def count(self) -> int:
        with self._connect() as con:
            row = con.execute(
                "SELECT COUNT(*) FROM macro_chunks WHERE namespace=?", (self.namespace,)
            ).fetchone()
        return int(row[0]) if row else 0

    def find_macro_matching_prefix(self, recent_intents: Sequence[str]) -> CompiledMacro | None:
        """Return the most-observed macro whose prefix matches the tail of ``recent_intents``.

        Multiple macros may share a prefix; the most-frequently observed one wins
        because it represents the strongest proceduralized shortcut.
        """

        if not recent_intents:
            return None
        cache_key = tuple(recent_intents)
        if cache_key in self._prefix_match_cache:
            cached = self._prefix_match_cache[cache_key]
            return cached if cached else None

        candidates: list[CompiledMacro] = []
        for m in self.all_macros():
            if m.matches_prefix(recent_intents):
                candidates.append(m)
        if not candidates:
            self._prefix_match_cache[cache_key] = None
            return None
        candidates.sort(
            key=lambda m: (m.observation_count, m.avg_confidence), reverse=True
        )
        best = candidates[0]
        self._prefix_match_cache[cache_key] = best
        return best

    def find_macro_by_features(self, query: torch.Tensor, *, min_cosine: float) -> CompiledMacro | None:
        """Return the strongest macro whose feature basin contains ``query``."""

        q = F.normalize(query.detach().float().reshape(1, -1), dim=-1)
        best: tuple[CompiledMacro, float] | None = None
        for macro in self.all_macros():
            feat = _align_vec_to_dim(macro.feature_vector, q.shape[-1])
            score = float((q @ F.normalize(feat.reshape(1, -1), dim=-1).T).item())
            if score < float(min_cosine):
                continue
            if best is None or (score, macro.observation_count, macro.avg_confidence) > (
                best[1],
                best[0].observation_count,
                best[0].avg_confidence,
            ):
                best = (macro, score)
        return best[0] if best is not None else None


def _frame_features_from_row(row: dict, *, text_encoder=None) -> torch.Tensor:
    from ..frame.continuous_frame import pack_cognitive_frame

    return pack_cognitive_frame(
        str(row.get("intent", "")),
        str(row.get("subject", "")),
        str(row.get("answer", "")),
        float(row.get("confidence", 0.0)),
        row.get("evidence") if isinstance(row.get("evidence"), dict) else None,
        text_encoder=text_encoder,
    )


def _row_multimodal_features(row: dict, *, text_encoder=None) -> torch.Tensor:
    """BROCA-width continuous state vector including affect and perceptual ambiguity."""

    out = _align_vec_to_dim(
        _frame_features_from_row(row, text_encoder=text_encoder),
        BROCA_FEATURE_DIM,
    )
    ev = _journal_evidence_dict(row)
    affect = ev.get("affect")
    if isinstance(affect, dict):
        vals = [
            float(affect.get("valence", 0.0) or 0.0),
            float(affect.get("arousal", 0.0) or 0.0),
            float(affect.get("entropy", 0.0) or 0.0),
            float(affect.get("certainty", 0.0) or 0.0),
        ]
        out[-8:-4] = torch.tensor(vals, dtype=torch.float32)
    ambiguity = _perceptual_ambiguity(ev)
    out[-4] = float(ambiguity)
    return out


def _perceptual_ambiguity(evidence: dict) -> float:
    explicit = evidence.get("friston_ambiguity", evidence.get("visual_ambiguity", None))
    if explicit is not None:
        value = float(explicit)
        if not math.isfinite(value):
            raise ValueError("perceptual ambiguity must be finite")
        return max(0.0, min(1.0, value))
    encoder_outputs = evidence.get("encoder_outputs")
    if not isinstance(encoder_outputs, dict) or not encoder_outputs:
        return 0.0
    confidences: list[float] = []
    for payload in encoder_outputs.values():
        if not isinstance(payload, dict):
            continue
        raw = payload.get("confidence")
        if raw is None:
            continue
        conf = float(raw)
        if not math.isfinite(conf):
            raise ValueError("encoder confidence must be finite")
        confidences.append(max(0.0, min(1.0, conf)))
    if not confidences:
        return 0.0
    return 1.0 - min(confidences)


def _trajectory_signature(features: Sequence[torch.Tensor]) -> torch.Tensor:
    if not features:
        raise ValueError("_trajectory_signature requires at least one feature vector")
    return F.normalize(torch.cat([F.normalize(f.float().flatten(), dim=0) for f in features], dim=0), dim=0)


def _macro_name_for_feature_centroid(feature_vector: torch.Tensor) -> str:
    arr = feature_vector.detach().cpu().float().numpy().astype(np.float32, copy=False)
    digest = hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]
    return f"macro_multimodal_{digest}"


@dataclass
class ChunkingDetectionConfig:
    """Tunable thresholds for motif detection during DMN idle ticks."""

    window_size: int = 64                 # how many recent journal rows to scan
    min_motif_length: int = 2             # smallest meaningful motif
    max_motif_length: int = 5             # longest motif to test (limits combinatorial cost)
    min_repetitions: int = 3              # need ≥ N occurrences to compile
    max_macros_per_tick: int = 4          # ceiling so a single tick can't dominate the registry
    hawkes_salience_cap: float = 24.0     # clamp λ(intent)/baseline when multiplying salience
    surprise_gap_scale: float = 2.5       # scales lexical_surprise_gap from journal evidence
    salience_oneshot_threshold: float = 10.0  # Σ row multipliers across the motif window
    hopfield_weight_min_for_oneshot: float = 0.38  # retrieval concentration floor when memory non-empty


def _journal_evidence_dict(row: dict) -> dict:
    ev = row.get("evidence")
    return ev if isinstance(ev, dict) else {}


def _motif_is_compilable(pattern: Sequence[str]) -> bool:
    return all(_intent_is_compilable(intent) for intent in pattern)


def _intent_is_compilable(intent: str) -> bool:
    value = str(intent or "").strip().lower()
    return bool(value) and value != "unknown" and not value.endswith("_pending")


def _row_salience_multiplier(mind: Any, row: dict, cfg: ChunkingDetectionConfig) -> float:
    """Hawkes-normalised intensity × lexical surprise bump for one journal row."""

    mult = 1.0
    intent = str(row.get("intent", "") or "unknown")
    hk = getattr(mind, "hawkes", None)
    if hk is not None:
        lam = float(hk.intensity(intent))
        baseline = float(getattr(hk, "baseline", 0.05))
        mult *= min(cfg.hawkes_salience_cap, max(1.0, lam / max(baseline, 1e-6)))
    ev = _journal_evidence_dict(row)
    gap = float(ev.get("lexical_surprise_gap") or 0.0)
    mult *= 1.0 + cfg.surprise_gap_scale * max(0.0, gap)
    return float(mult)


def _window_salience(
    mind: Any,
    rows: list[dict],
    start: int,
    length: int,
    cfg: ChunkingDetectionConfig,
) -> float:
    if start < 0 or length < 0:
        raise ValueError(f"_window_salience: invalid window start={start!r} length={length!r} (must be non-negative)")
    end = start + length
    if end > len(rows):
        raise ValueError(
            f"_window_salience: window [{start}, {end}) extends past rows (len(rows)={len(rows)})"
        )
    return sum(
        _row_salience_multiplier(mind, rows[start + k], cfg) for k in range(length)
    )


def _hopfield_concentration_ok(
    mind: Any,
    mean_feat: torch.Tensor,
    cfg: ChunkingDetectionConfig,
) -> bool:
    """Reject one-shot macros when Hopfield mass stays diffuse (likely stochastic noise)."""

    hm = getattr(mind, "hopfield_memory", None)
    if hm is None:
        return True
    if len(hm) == 0:
        return True
    q = _align_vec_to_dim(mean_feat, hm.d_model).to(device=hm.device, dtype=hm.dtype)
    _, w = hm.retrieve(q)
    wmax = float(w.max().item()) if w.numel() else 0.0
    return wmax >= cfg.hopfield_weight_min_for_oneshot


def find_salience_forced_motifs(
    intents: Sequence[str],
    rows: list[dict],
    mind: Any,
    cfg: ChunkingDetectionConfig,
    *,
    min_motif_length: int,
    max_motif_length: int,
) -> dict[tuple[str, ...], tuple[float, int]]:
    """Return motif → (salience score, start index) for high-salience windows."""

    best: dict[tuple[str, ...], tuple[float, int]] = {}
    n = len(intents)
    for L in range(max_motif_length, min_motif_length - 1, -1):
        if L > n:
            continue
        for s in range(n - L + 1):
            pat = tuple(intents[s : s + L])
            if not _motif_is_compilable(pat):
                continue
            sal = _window_salience(mind, rows, s, L, cfg)
            if sal < cfg.salience_oneshot_threshold:
                continue
            feats: list[torch.Tensor] = []
            for k in range(L):
                try:
                    feats.append(
                        _frame_features_from_row(rows[s + k], text_encoder=getattr(mind, "text_encoder", None))
                    )
                except Exception:
                    logger.warning(
                        "find_salience_forced_motifs: skipping frame at row index %s (%s)",
                        s + k,
                        rows[s + k].get("id", "?"),
                        exc_info=True,
                    )
            if not feats:
                continue
            mean_feat = torch.stack(feats, dim=0).mean(dim=0)
            if not _hopfield_concentration_ok(mind, mean_feat, cfg):
                continue
            prev = best.get(pat)
            if prev is None or sal > prev[0]:
                best[pat] = (sal, s)
    return best


def find_multimodal_trajectory_motifs(
    rows: list[dict],
    mind: Any,
    cfg: ChunkingDetectionConfig,
    *,
    text_encoder=None,
) -> list[tuple[list[int], torch.Tensor, float, list[tuple[str, ...]]]]:
    """Cluster repeated continuous state trajectories independent of exact intent tuples."""

    out: list[tuple[list[int], torch.Tensor, float, list[tuple[str, ...]]]] = []
    n = len(rows)
    for L in range(cfg.max_motif_length, cfg.min_motif_length - 1, -1):
        if L > n:
            continue
        windows: list[tuple[int, tuple[str, ...], torch.Tensor, torch.Tensor]] = []
        for s in range(n - L + 1):
            features = [
                _row_multimodal_features(rows[s + offset], text_encoder=text_encoder)
                for offset in range(L)
            ]
            signature = _trajectory_signature(features)
            mean_feat = torch.stack(features, dim=0).mean(dim=0)
            pattern = tuple(str(rows[s + offset].get("intent", "") or "unknown") for offset in range(L))
            windows.append((s, pattern, signature, mean_feat))
        if len(windows) < cfg.min_repetitions:
            continue
        threshold = _derived_cluster_threshold([w[2] for w in windows], cfg)
        clusters: list[dict[str, Any]] = []
        for start, pattern, signature, mean_feat in windows:
            best_ix = -1
            best_score = -1.0
            for ix, cluster in enumerate(clusters):
                score = float(torch.dot(signature, cluster["centroid"]).item())
                if score > best_score:
                    best_score = score
                    best_ix = ix
            if best_ix >= 0 and best_score >= threshold:
                cluster = clusters[best_ix]
                cluster["starts"].append(start)
                cluster["patterns"].append(pattern)
                cluster["features"].append(mean_feat)
                stacked = torch.stack(cluster["signatures"] + [signature], dim=0)
                cluster["centroid"] = F.normalize(stacked.mean(dim=0), dim=0)
                cluster["signatures"].append(signature)
                continue
            clusters.append(
                {
                    "starts": [start],
                    "patterns": [pattern],
                    "signatures": [signature],
                    "features": [mean_feat],
                    "centroid": signature,
                }
            )
        for cluster in clusters:
            starts = _non_overlapping_starts(cluster["starts"], L)
            if len(starts) < cfg.min_repetitions:
                continue
            selected = [
                cluster["features"][cluster["starts"].index(start)]
                for start in starts
            ]
            centroid_feature = torch.stack(selected, dim=0).mean(dim=0)
            if not _hopfield_concentration_ok(mind, centroid_feature, cfg):
                continue
            out.append(
                (
                    starts,
                    centroid_feature,
                    float(len(starts)),
                    list(cluster["patterns"]),
                )
            )
    out.sort(key=lambda item: (-item[2], -len(item[0])))
    return out


def _derived_cluster_threshold(signatures: Sequence[torch.Tensor], cfg: ChunkingDetectionConfig) -> float:
    if len(signatures) < 2:
        return 1.0
    sims: list[float] = []
    for i in range(len(signatures)):
        for j in range(i + 1, len(signatures)):
            sims.append(float(torch.dot(signatures[i], signatures[j]).item()))
    if not sims:
        return 1.0
    mu = sum(sims) / len(sims)
    var = sum((x - mu) ** 2 for x in sims) / len(sims)
    return max(float(cfg.hopfield_weight_min_for_oneshot), min(0.999, mu + math.sqrt(var)))


def _non_overlapping_starts(starts: Sequence[int], length: int) -> list[int]:
    out: list[int] = []
    last_end = -1
    for start in sorted(int(s) for s in starts):
        if start < last_end:
            continue
        out.append(start)
        last_end = start + int(length)
    return out


class DMNChunkingCompiler:
    """Detects repeated intent motifs in the workspace journal and compiles them into macros.

    The compiler is driven by the DMN's idle tick (one call to :meth:`run_once`
    per tick).  Motif detection combines (a) exact frequency repetition — slide
    windows over recent intents and compile patterns meeting ``min_repetitions``
    — with (b) **salience forcing**, where Hawkes intensity and lexical surprise
    gaps multiply per-row weight so a rare burst can satisfy the compiler in one
    shot when Hopfield retrieval concentrates on a stable basin.

    Long patterns are reported before short ones so the registry favors the most
    specific motif when sub-patterns coincide.
    """

    def __init__(
        self,
        mind,
        *,
        registry: MacroChunkRegistry,
        config: ChunkingDetectionConfig | None = None,
    ):
        self.mind = mind
        self.registry = registry
        self.config = config or ChunkingDetectionConfig()
        self.iterations: int = 0

    @staticmethod
    def find_repeated_motifs(
        intents: Sequence[str],
        *,
        min_motif_length: int,
        max_motif_length: int,
        min_repetitions: int,
    ) -> list[tuple[tuple[str, ...], list[int]]]:
        """Return ``(pattern, [start_indices])`` tuples for patterns repeating ≥ ``min_repetitions`` times.

        Counts non-overlapping occurrences via a left-to-right greedy scan so
        that "AAA" is counted as one occurrence of "AA" rather than two
        overlapping ones.  Long patterns are reported before short ones so the
        registry favors the most specific motif when sub-patterns coincide.
        """

        out: list[tuple[tuple[str, ...], list[int]]] = []
        if not intents:
            return out
        n = len(intents)
        for L in range(max_motif_length, min_motif_length - 1, -1):
            if L > n:
                continue
            counts: dict[tuple[str, ...], list[int]] = {}
            i = 0
            while i + L <= n:
                pat = tuple(intents[i : i + L])
                if not _motif_is_compilable(pat):
                    i += 1
                    continue
                counts.setdefault(pat, []).append(i)
                i += 1
            for pat, starts in counts.items():
                # Compute non-overlapping greedy occurrence indices.
                non_overlap: list[int] = []
                last_end = -1
                for s in starts:
                    if s >= last_end:
                        non_overlap.append(s)
                        last_end = s + L
                if len(non_overlap) >= min_repetitions:
                    out.append((pat, non_overlap))
        return out

    def _gather_recent_intents(self) -> tuple[list[dict], list[str]]:
        try:
            rows = self.mind.journal.recent(limit=self.config.window_size)
        except Exception:
            logger.exception("DMNChunkingCompiler: journal.recent failed")
            return [], []
        intents = [str(r.get("intent", "") or "unknown") for r in rows]
        return rows, intents

    def run_once(self) -> dict:
        """Single DMN tick: detect motifs, compile macros, persist to registry.

        Returns telemetry suitable for the DMN's phase summary dict.
        """

        cfg = self.config
        rows, intents = self._gather_recent_intents()
        if len(intents) < cfg.min_motif_length:
            return {
                "scanned": len(intents),
                "candidates": 0,
                "compiled": 0,
                "reflections": [],
            }

        motifs_freq = self.find_repeated_motifs(
            intents,
            min_motif_length=cfg.min_motif_length,
            max_motif_length=cfg.max_motif_length,
            min_repetitions=cfg.min_repetitions,
        )
        salience_best = find_salience_forced_motifs(
            intents,
            rows,
            self.mind,
            cfg,
            min_motif_length=cfg.min_motif_length,
            max_motif_length=cfg.max_motif_length,
        )
        text_encoder = getattr(self.mind, "text_encoder", None)
        multimodal_motifs = find_multimodal_trajectory_motifs(
            rows,
            self.mind,
            cfg,
            text_encoder=text_encoder,
        )
        freq_patterns = {pat for pat, _ in motifs_freq}
        work: list[tuple[tuple[str, ...], list[int], str, float, torch.Tensor | None, list[tuple[str, ...]]]] = [
            (pat, starts, "frequency", float(len(starts)), None, []) for pat, starts in motifs_freq
        ]
        for pat, (sal, s) in salience_best.items():
            if pat in freq_patterns:
                continue
            work.append((pat, [s], "salience", float(sal), None, []))
        for starts, centroid_feature, score, member_patterns in multimodal_motifs:
            length = len(member_patterns[0]) if member_patterns else cfg.min_motif_length
            pat = tuple(f"multimodal_{i}" for i in range(length))
            work.append((pat, starts, "multimodal_trajectory", float(score), centroid_feature, member_patterns))
        work.sort(key=lambda item: (-item[3], item[2] == "salience"))

        compiled: list[CompiledMacro] = []
        reflections: list[dict] = []
        for pat, starts, compile_via, priority_score, centroid_feature, member_patterns in work[: cfg.max_macros_per_tick]:
            # Compute the mean feature vector across all instances of the motif.
            instance_feats: list[torch.Tensor] = []
            instance_ids: list[int] = []
            confs: list[float] = []
            for s in starts:
                for offset in range(len(pat)):
                    row = rows[s + offset]
                    raw_id = row.get("id", None)
                    try:
                        eid = int(raw_id) if raw_id is not None else None
                    except (TypeError, ValueError):
                        eid = None
                    if eid is None or eid < 0:
                        logger.warning(
                            "DMNChunkingCompiler: skipping journal row without valid non-negative integer id keys=%s",
                            list(row.keys()),
                        )
                        continue
                    instance_ids.append(eid)
                    confs.append(float(row.get("confidence", 0.0)))
                    try:
                        instance_feats.append(
                            _frame_features_from_row(row, text_encoder=text_encoder)
                        )
                    except Exception:
                        logger.exception(
                            "DMNChunkingCompiler: feature pack failed for episode %s",
                            row.get("id"),
                        )
            if not instance_feats and centroid_feature is None:
                continue
            mean_feat = centroid_feature if centroid_feature is not None else torch.stack(instance_feats, dim=0).mean(dim=0)
            avg_conf = sum(confs) / max(1, len(confs))
            obs_count = len(starts) if compile_via == "frequency" else min(1000, max(1, int(round(priority_score))))
            macro = CompiledMacro(
                name=(
                    _macro_name_for_feature_centroid(mean_feat)
                    if compile_via == "multimodal_trajectory"
                    else _macro_name_for_pattern(pat)
                ),
                pattern=tuple(pat),
                observation_count=obs_count,
                avg_confidence=float(avg_conf),
                feature_vector=mean_feat,
                last_seen_at=time.time(),
                member_episodes=instance_ids,
            )
            self.registry.upsert(macro)
            compiled.append(macro)
            reflections.append(
                {
                    "kind": "chunk_compiled",
                    "compile_via": compile_via,
                    "macro_name": macro.name,
                    "pattern": list(macro.pattern),
                    "observation_count": macro.observation_count,
                    "avg_confidence": float(macro.avg_confidence),
                    "member_episodes": list(macro.member_episodes),
                    "salience_score": float(priority_score),
                    "member_patterns": [list(p) for p in member_patterns],
                }
            )
            logger.info(
                "DMN.chunking.compiled: name=%s pattern=%s via=%s n=%d avg_conf=%.3f salience=%.3f",
                macro.name,
                list(macro.pattern),
                compile_via,
                macro.observation_count,
                float(macro.avg_confidence),
                float(priority_score),
            )

        self.iterations += 1
        return {
            "scanned": len(intents),
            "candidates": len(motifs_freq) + len(salience_best) + len(multimodal_motifs),
            "compiled": len(compiled),
            "reflections": reflections,
        }


def macro_frame_features(macro: CompiledMacro) -> torch.Tensor:
    """Return the macro's compiled feature vector reshaped to ``BROCA_FEATURE_DIM``.

    Pads or truncates as needed so a stale macro persisted under a previous
    feature dimension still loads cleanly. VSA injection tail is zero for
    macros (loaded motifs predate the current utterance's holographic bind).
    """

    return _align_vec_to_dim(macro.feature_vector, BROCA_FEATURE_DIM)
