"""Lexical subword projector for sketch-space encoding.

A frozen, deterministic, model-free :class:`TextEncoder`. Hashes character
n-grams, token unigrams, prefixes, suffixes, and adjacent token pairs into
a fixed-width signed-feature vector, then renormalizes.
"""

from __future__ import annotations

import math
import re

import torch

from .dimensions import FrameDimensions


class SubwordProjector:
    """Deterministic subword-feature projector that satisfies :class:`TextEncoder`.

    The projection is stable across processes for the same input — the hash
    salt and seed counts are baked into the class and must not be tuned per
    run. Two callers with the same text will produce bit-identical vectors.
    """

    _TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
    _FNV_OFFSET = 0xCBF29CE484222325
    _FNV_PRIME = 0x100000001B3
    _GOLDEN_GAMMA = 0x9E3779B185EBCA87
    _U64_MASK = 0xFFFFFFFFFFFFFFFF

    def __init__(self, *, dim: int | None = None, seeds: int | None = None) -> None:
        self.dim = int(dim) if dim is not None else FrameDimensions.SKETCH_DIM
        self.seeds = int(seeds) if seeds is not None else FrameDimensions.SKETCH_SEEDS
        if self.dim <= 0:
            raise ValueError(f"SubwordProjector.dim must be positive, got {self.dim}")
        if self.seeds <= 0:
            raise ValueError(f"SubwordProjector.seeds must be positive, got {self.seeds}")

    def __call__(self, text: str) -> torch.Tensor:
        return self.encode(text)

    def encode(self, text: str) -> torch.Tensor:
        v = torch.zeros(self.dim, dtype=torch.float32)
        feats = self._subword_features(text)
        if not feats:
            return v
        seed_scale = math.sqrt(float(self.seeds))
        for feature, weight in feats:
            for seed in range(self.seeds):
                idx, sign = self._feature_hash(feature, seed)
                v[idx] += sign * float(weight) / seed_scale
        norm = v.norm()
        if float(norm) > 1e-12:
            v = v / norm
        return v

    def _tokens(self, text: str) -> list[str]:
        toks = self._TOKEN_RE.findall(text.strip().lower())
        if toks:
            return toks
        raw = text.strip().lower()
        return [raw] if raw else []

    def _stable_u64(self, feature: str, seed: int) -> int:
        h = (self._FNV_OFFSET ^ ((int(seed) + 1) * self._GOLDEN_GAMMA)) & self._U64_MASK
        for b in feature.encode("utf-8", errors="replace"):
            h ^= int(b)
            h = (h * self._FNV_PRIME) & self._U64_MASK
        return h

    def _feature_hash(self, feature: str, seed: int) -> tuple[int, float]:
        h = self._stable_u64(feature, seed)
        idx = h % int(self.dim)
        sign = 1.0 if (h >> 63) & 1 else -1.0
        return idx, sign

    def _subword_features(self, text: str) -> list[tuple[str, float]]:
        toks = self._tokens(text)
        feats: list[tuple[str, float]] = []
        for tok in toks:
            feats.append((f"tok:{tok}", 0.55))
            bounded = f"<{tok}>"
            for n in range(FrameDimensions.NGRAM_MIN, FrameDimensions.NGRAM_MAX + 1):
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
