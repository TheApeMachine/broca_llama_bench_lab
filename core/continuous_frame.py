"""Stable semantic sketch vectors for cognitive frames.

Instead of fixed one-hot slots per intent/entity/value, arbitrary UTF-8 strings
are mapped into a fixed-dimensional frozen subword feature projection. When a
host model is available, the projection uses its frozen tokenizer embeddings;
otherwise it falls back to lexical subword features. Numeric faculty scalars are
appended unchanged.
"""

from __future__ import annotations

import math
import re
from typing import Any, Callable

import torch

SKETCH_DIM = 128
SKETCH_SEEDS = 8
NGRAM_MIN = 3
NGRAM_MAX = 5

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

NUMERIC_FEATURE_FIELDS = (
    "confidence",
    "p_do_positive",
    "p_do_negative",
    "ate",
    "policy_listen",
    "policy_open_left",
    "policy_open_right",
    "delta_ce",
    "bias",
)

NUMERIC_TAIL_LEN = len(NUMERIC_FEATURE_FIELDS)
COGNITIVE_FRAME_DIM = SKETCH_DIM * 3 + NUMERIC_TAIL_LEN
TextEncoder = Callable[[str], torch.Tensor]


def _tokens(text: str) -> list[str]:
    toks = _TOKEN_RE.findall(text.strip().lower())
    if toks:
        return toks
    raw = text.strip().lower()
    return [raw] if raw else []


def _stable_u64(feature: str, seed: int) -> int:
    """Small deterministic non-cryptographic hash for signed feature projection."""

    h = (0xCBF29CE484222325 ^ ((int(seed) + 1) * 0x9E3779B185EBCA87)) & 0xFFFFFFFFFFFFFFFF
    for b in feature.encode("utf-8", errors="replace"):
        h ^= int(b)
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def _feature_hash(feature: str, seed: int, dim: int) -> tuple[int, float]:
    h = _stable_u64(feature, seed)
    idx = h % int(dim)
    sign = 1.0 if (h >> 63) & 1 else -1.0
    return idx, sign


def _subword_features(text: str) -> list[tuple[str, float]]:
    toks = _tokens(text)
    feats: list[tuple[str, float]] = []
    for tok in toks:
        feats.append((f"tok:{tok}", 0.55))

        bounded = f"<{tok}>"
        for n in range(NGRAM_MIN, NGRAM_MAX + 1):
            if len(bounded) < n:
                continue
            grams = [bounded[i : i + n] for i in range(len(bounded) - n + 1)]
            if not grams:
                continue
            w = 1.0 / math.sqrt(float(len(grams)))
            feats.extend((f"char{n}:{g}", w) for g in grams)

        if len(tok) >= 4:
            feats.append((f"prefix:{tok[:3]}", 0.35))
            feats.append((f"suffix:{tok[-3:]}", 0.35))

    for a, b in zip(toks, toks[1:]):
        feats.append((f"pair:{a}:{b}", 0.45))
    return feats


def semantic_subword_sketch(text: str, *, dim: int = SKETCH_DIM, n_seeds: int = SKETCH_SEEDS) -> torch.Tensor:
    """Frozen lexical subword projection for ``text`` (stable across processes)."""

    v = torch.zeros(dim, dtype=torch.float32)
    feats = _subword_features(text)
    if not feats:
        return v
    seeds = max(1, int(n_seeds))
    seed_scale = math.sqrt(float(seeds))
    for feature, weight in feats:
        for seed in range(seeds):
            idx, sign = _feature_hash(feature, seed, dim)
            v[idx] += sign * float(weight) / seed_scale
    norm = v.norm()
    if float(norm) > 1e-12:
        v = v / norm
    return v


def stable_sketch(text: str, *, dim: int = SKETCH_DIM, n_seeds: int = SKETCH_SEEDS) -> torch.Tensor:
    """Backward-compatible name for the semantic subword projection."""

    return semantic_subword_sketch(text, dim=dim, n_seeds=n_seeds)


class FrozenSubwordProjector:
    """Project frozen tokenizer embedding averages into the frame sketch space.

    Projection output dim is fixed to :data:`SKETCH_DIM` so ``pack_cognitive_frame`` /
    ``_encode_or_sketch`` agree on vector size.

    By default ``embedding_weight`` is stored as ``embedding_weight.detach()`` (no copy).
    The caller must not mutate that tensor afterward. Pass ``clone_embedding=True`` to
    take an independent copy when the underlying weight buffer may be updated in place.
    """

    def __init__(
        self,
        tokenizer: Any,
        embedding_weight: torch.Tensor,
        *,
        seed: int = 0,
        clone_embedding: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.dim = int(SKETCH_DIM)
        w = embedding_weight.detach()
        self.embedding_weight = w.clone() if clone_embedding else w
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed))
        self.projection = torch.empty((self.embedding_weight.shape[1], self.dim), dtype=torch.float32)
        self.projection.normal_(mean=0.0, std=1.0 / math.sqrt(float(self.dim)), generator=g)

    def _encode(self, text: str) -> list[int]:
        try:
            ids = self.tokenizer.encode(str(text), add_special_tokens=False)
        except TypeError:
            ids = self.tokenizer.encode(str(text))
        n_vocab = self.embedding_weight.shape[0]
        return [int(i) for i in ids if 0 <= int(i) < n_vocab]

    def __call__(self, text: str) -> torch.Tensor:
        ids = self._encode(text)
        if not ids:
            return stable_sketch(text, dim=self.dim)
        idx = torch.tensor(ids, dtype=torch.long, device=self.embedding_weight.device)
        pooled = self.embedding_weight.index_select(0, idx).float().mean(dim=0).cpu()
        z = pooled @ self.projection
        z = z + 0.15 * stable_sketch(text, dim=self.dim)
        norm = z.norm()
        if float(norm) > 1e-12:
            z = z / norm
        return z.to(dtype=torch.float32)


def frozen_subword_projector_from_model(
    model: Any,
    tokenizer: Any,
    *,
    seed: int = 0,
    clone_embedding: bool = False,
) -> FrozenSubwordProjector | None:
    """Build a frame-space projector from a host model's frozen input embeddings."""

    host = getattr(model, "llm", model)
    get_input_embeddings = getattr(host, "get_input_embeddings", None)
    emb = get_input_embeddings() if callable(get_input_embeddings) else None
    if emb is None:
        inner = getattr(host, "model", None)
        emb = getattr(inner, "embed_tokens", None)
    weight = getattr(emb, "weight", None)
    if weight is None:
        return None
    return FrozenSubwordProjector(tokenizer, weight, seed=seed, clone_embedding=clone_embedding)


def _encode_or_sketch(text: str, text_encoder: TextEncoder | None) -> torch.Tensor:
    v = text_encoder(str(text)) if text_encoder is not None else stable_sketch(str(text))
    if not isinstance(v, torch.Tensor):
        v = torch.tensor(v, dtype=torch.float32)
    v = v.detach().float().cpu().view(-1)
    if v.numel() != SKETCH_DIM:
        raise ValueError(f"text encoder returned dim {v.numel()}, expected {SKETCH_DIM}")
    return v


def numeric_tail(confidence: float, evidence: dict[str, Any] | None) -> torch.Tensor:
    ev = evidence or {}
    policy = ev.get("policy_posterior", {}) or {}
    return torch.tensor(
        [
            float(confidence),
            float(ev.get("p_do_positive", 0.0)),
            float(ev.get("p_do_negative", 0.0)),
            float(ev.get("ate", 0.0)),
            float(policy.get("listen", 0.0)),
            float(policy.get("open_left", 0.0)),
            float(policy.get("open_right", 0.0)),
            float(ev.get("delta_ce", 0.0)),
            1.0,
        ],
        dtype=torch.float32,
    )


def pack_cognitive_frame(
    intent: str,
    subject: str,
    answer: str,
    confidence: float,
    evidence: dict[str, Any] | None,
    *,
    text_encoder: TextEncoder | None = None,
) -> torch.Tensor:
    """Concatenate sketch(intent), sketch(subject), sketch(answer), numeric tail."""

    parts = torch.cat(
        [
            _encode_or_sketch(str(intent), text_encoder),
            _encode_or_sketch(str(subject), text_encoder),
            _encode_or_sketch(str(answer), text_encoder),
            numeric_tail(confidence, evidence),
        ],
        dim=0,
    )
    return parts
