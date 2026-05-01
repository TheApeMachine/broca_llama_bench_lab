from __future__ import annotations

import math
import sys

import torch

from core.continuous_frame import COGNITIVE_FRAME_DIM
from core.vision import VisionEncoder


def _gradient_image() -> torch.Tensor:
    h = w = 64
    grid = torch.linspace(0.0, 1.0, w).unsqueeze(0).expand(h, w)
    return torch.stack([grid, grid, grid], dim=0)


def _checker_image() -> torch.Tensor:
    h = w = 64
    rows = torch.arange(h).unsqueeze(1).expand(h, w)
    cols = torch.arange(w).unsqueeze(0).expand(h, w)
    pattern = ((rows // 8) + (cols // 8)) % 2
    img = pattern.to(torch.float32)
    return torch.stack([img, img, img], dim=0)


def _flat_image() -> torch.Tensor:
    return torch.full((3, 64, 64), 0.5)


def test_encode_image_returns_cognitive_frame_layout():
    enc = VisionEncoder(use_real_model=False)
    out = enc.encode_image(_gradient_image())
    assert out.shape == (COGNITIVE_FRAME_DIM,), out.shape
    assert torch.isfinite(out).all()


def test_encode_image_is_deterministic():
    enc = VisionEncoder(use_real_model=False)
    img = _gradient_image()
    a = enc.encode_image(img)
    b = enc.encode_image(img)
    assert torch.allclose(a, b)


def test_distinct_images_produce_distinct_embeddings():
    enc = VisionEncoder(use_real_model=False)
    grad = enc.encode_image(_gradient_image())
    checker = enc.encode_image(_checker_image())
    flat = enc.encode_image(_flat_image())
    cos_gc = float(torch.nn.functional.cosine_similarity(grad, checker, dim=0).item())
    cos_gf = float(torch.nn.functional.cosine_similarity(grad, flat, dim=0).item())
    cos_cf = float(torch.nn.functional.cosine_similarity(checker, flat, dim=0).item())
    # Distinct images should not map to identical vectors.
    assert cos_gc < 0.999
    assert cos_gf < 0.999
    assert cos_cf < 0.999


def test_ambiguity_separates_uniform_from_textured():
    enc = VisionEncoder(use_real_model=False)
    h_flat = enc.ambiguity(_flat_image(), n_buckets=16)
    h_checker = enc.ambiguity(_checker_image(), n_buckets=16)
    h_grad = enc.ambiguity(_gradient_image(), n_buckets=16)
    # A flat image collapses onto one bucket → low entropy.
    assert h_flat < 0.5
    # A checker fills two buckets → ~log(2) ≈ 0.69 nats.
    assert 0.5 < h_checker < 1.0
    # A gradient spreads across all buckets → close to log(16).
    assert h_grad > 1.5


def test_real_model_falls_back_silently_when_transformers_unavailable():
    from unittest.mock import patch

    with patch.dict(sys.modules, {"transformers": None}):
        enc = VisionEncoder(use_real_model=True)
    assert enc.is_real is False
    out = enc.encode_image(_gradient_image())
    assert out.shape == (COGNITIVE_FRAME_DIM,)
