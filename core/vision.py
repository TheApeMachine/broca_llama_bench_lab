"""Occipital peripheral: a vision encoder with a graceful no-deps fallback.

Real models (CLIP, SigLIP) are loaded lazily via ``transformers`` when the user
opts in. When the heavy stack isn't installed the module falls back to a
deterministic perceptual hash + frequency sketch that maps any PIL/torch image
into the same ``COGNITIVE_FRAME_DIM`` space as the rest of the substrate, so
the rest of the architecture can interoperate with the encoder without a hard
dependency.

The encoder always exposes:

* ``encode_image(image) -> torch.Tensor`` — fixed-dim cognitive-frame vector.
* ``ambiguity(image) -> float`` — Friston-style observation entropy. The active
  inference agent uses this to decide whether to ask for a closer image.

For the production model, set ``VISION_MODEL`` (default
``openai/clip-vit-base-patch32``) and pass ``use_real_model=True`` to the
constructor; the loader silently falls back if transformers is unavailable.
"""

from __future__ import annotations

import logging
import math
import os
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .continuous_frame import COGNITIVE_FRAME_DIM, SKETCH_DIM, stable_sketch

logger = logging.getLogger(__name__)


def _to_tensor(image: Any) -> torch.Tensor:
    """Normalize an arbitrary image input to a [3, H, W] float tensor in [0, 1]."""

    if isinstance(image, torch.Tensor):
        t = image.detach().float()
        if t.numel() > 0 and float(t.max().item()) > 1.0:
            t = t / 255.0
    else:
        try:
            from PIL import Image  # type: ignore
        except ImportError as exc:  # pragma: no cover - PIL is a typical dep
            raise RuntimeError(
                "PIL/Pillow is required to decode non-tensor images"
            ) from exc
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert("RGB")
        elif isinstance(image, (bytes, bytearray)):
            img = Image.open(BytesIO(image)).convert("RGB")
        elif isinstance(image, Image.Image):
            img = image.convert("RGB")
        else:
            raise TypeError(f"unsupported image type: {type(image)!r}")
        import numpy as np  # numpy ships with torch

        arr = np.asarray(img, dtype="float32") / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    if t.ndim == 2:
        t = t.unsqueeze(0).repeat(3, 1, 1)
    if t.shape[0] == 1:
        t = t.repeat(3, 1, 1)
    if t.shape[0] != 3:
        raise ValueError(f"expected 3-channel image, got shape {tuple(t.shape)}")
    return t.to(dtype=torch.float32).clamp(0.0, 1.0)


def _phash_sketch(image: torch.Tensor, *, dim: int = SKETCH_DIM) -> torch.Tensor:
    """Deterministic perceptual sketch for the no-deps path.

    The first mosaic version used a hand-rolled DCT plus FFT. That is elegant,
    but on some CPU/OpenMP combinations it can become pathologically slow when
    run after many torch-heavy tests. The substrate does not need a perfect
    pHash here; it needs a stable visual vector. This version keeps the same
    signals — luminance distribution, low-frequency layout, and texture — but
    computes them with small pooling/difference operations only.
    """

    g = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
    g = F.adaptive_avg_pool2d(g.unsqueeze(0).unsqueeze(0), (32, 32)).squeeze()
    bins = torch.clamp((g.flatten() * 16).to(torch.long), 0, 15)
    hist = torch.bincount(bins, minlength=16).to(torch.float32)
    hist = hist / hist.sum().clamp_min(1e-9)

    # Low-frequency spatial layout, equivalent in spirit to a tiny pHash block
    # but without building a DCT basis or invoking FFT kernels.
    low_freq = F.adaptive_avg_pool2d(g.unsqueeze(0).unsqueeze(0), (8, 8)).flatten()
    low_freq = low_freq - low_freq.mean()

    dx = (g[:, 1:] - g[:, :-1]).abs()
    dy = (g[1:, :] - g[:-1, :]).abs()
    texture = torch.tensor(
        [
            float(dx.mean().item()),
            float(dy.mean().item()),
            float(dx.std(unbiased=False).item()),
            float(dy.std(unbiased=False).item()),
            float(g.mean().item()),
            float(g.std(unbiased=False).item()),
            float(g.min().item()),
            float(g.max().item()),
        ],
        dtype=torch.float32,
        device=g.device,
    )

    feature = torch.cat([hist, low_freq, texture]).nan_to_num(0.0)
    feature = F.normalize(feature, dim=0)
    digest = " ".join(f"{x:.4f}" for x in feature.detach().cpu().tolist())
    sketch = stable_sketch(digest, dim=dim)
    pad = torch.zeros(dim, dtype=torch.float32)
    cpu_feature = feature.detach().cpu().to(torch.float32)
    pad[: min(dim, cpu_feature.shape[0])] = cpu_feature[: min(dim, cpu_feature.shape[0])]
    blended = F.normalize(0.6 * pad + 0.4 * sketch, dim=0)
    return blended


class VisionEncoder:
    """Substrate-shaped vision encoder.

    The output is laid out the same way as :func:`pack_cognitive_frame`:
    ``[sketch(intent="vision"), sketch(image), sketch(scene), numeric_tail]``
    so it feeds directly into ``TrainableBrocaGraft`` and ``feature_graft``
    without any additional bridge code.
    """

    def __init__(
        self,
        *,
        model_id: str | None = None,
        use_real_model: bool = False,
        device: torch.device | str | None = None,
    ) -> None:
        self.model_id = model_id or os.environ.get(
            "VISION_MODEL", "openai/clip-vit-base-patch32"
        )
        self.device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        self._model = None
        self._processor = None
        self._real = False
        self._pil_available = False
        try:
            import PIL.Image  # type: ignore  # noqa: F401

            self._pil_available = True
        except ImportError:
            logger.warning(
                "VisionEncoder: PIL not installed; real-model tensor inputs will fall back "
                "to perceptual hashing until Pillow is available"
            )
        if use_real_model:
            self._try_load_real()
        logger.info(
            "VisionEncoder.init: model_id=%s real=%s device=%s",
            self.model_id,
            self._real,
            self.device,
        )

    @property
    def is_real(self) -> bool:
        return self._real

    def _try_load_real(self) -> None:
        try:
            from transformers import AutoModel, AutoProcessor  # type: ignore
        except ImportError:
            logger.warning(
                "VisionEncoder: transformers not available; falling back to perceptual sketch"
            )
            self._real = False
            return
        try:
            self._processor = AutoProcessor.from_pretrained(self.model_id)
            self._model = (
                AutoModel.from_pretrained(self.model_id).to(self.device).eval()
            )
            self._real = True
        except (FileNotFoundError, OSError, RuntimeError) as exc:  # pragma: no cover
            logger.warning(
                "VisionEncoder: failed to load %s [%s]: %s; using perceptual sketch",
                self.model_id,
                type(exc).__name__,
                exc,
            )
            self._real = False

    @torch.no_grad()
    def encode_image(self, image: Any) -> torch.Tensor:
        """Map an image to the substrate's cognitive-frame layout."""

        if self._real and self._model is not None and self._processor is not None:
            pil = image if hasattr(image, "convert") else None
            inputs = None
            if isinstance(image, torch.Tensor):
                if not self._pil_available:
                    tensor_fb = _to_tensor(image)
                    sketch_fb = _phash_sketch(tensor_fb)
                    return _embed_to_cognitive_frame(sketch_fb)
                t = image.detach().float().cpu()
                if t.ndim == 3:
                    t = t.unsqueeze(0)
                if t.numel() > 0 and float(t.max().item()) > 1.0:
                    t = t / 255.0
                t = t.clamp(0.0, 1.0)
                from PIL import Image as PILImage  # type: ignore

                pil_images: list[Any] = []
                for bi in range(int(t.shape[0])):
                    arr = (
                        (t[bi].clamp(0.0, 1.0) * 255.0)
                        .clamp(0, 255)
                        .to(dtype=torch.uint8)
                        .permute(1, 2, 0)
                        .contiguous()
                        .numpy()
                    )
                    pil_images.append(PILImage.fromarray(arr, mode="RGB"))
                inputs = self._processor(images=pil_images, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            elif pil is None:
                from PIL import Image as PILOpen  # type: ignore

                if isinstance(image, (str, Path)):
                    pil = PILOpen.open(image).convert("RGB")
                elif isinstance(image, (bytes, bytearray)):
                    pil = PILOpen.open(BytesIO(image)).convert("RGB")
                if pil is not None:
                    raw = self._processor(images=pil, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in dict(raw).items()}
            if inputs is not None:
                feats = (
                    self._model.get_image_features(**inputs)
                    if hasattr(self._model, "get_image_features")
                    else self._model(**inputs).pooler_output
                )
                embed = F.normalize(
                    feats[0].detach().cpu().to(torch.float32), dim=0
                )
                return _embed_to_cognitive_frame(embed)
        # No-deps perceptual fallback path.
        tensor = _to_tensor(image)
        sketch = _phash_sketch(tensor)
        return _embed_to_cognitive_frame(sketch)

    def ambiguity(self, image: Any, *, n_buckets: int = 16) -> float:
        """Friston-style observation entropy of the image's luminance distribution.

        Uniform images (a blank wall, a JPEG artifact) yield log(n_buckets);
        well-exposed photographs collapse onto fewer modes. Returned in nats.
        """

        tensor = _to_tensor(image)
        gray = 0.299 * tensor[0] + 0.587 * tensor[1] + 0.114 * tensor[2]
        nb = int(n_buckets)
        bins = torch.clamp((gray.flatten() * nb).to(torch.long), 0, nb - 1)
        hist = torch.bincount(bins, minlength=nb).to(torch.float32)
        p = (hist / hist.sum().clamp_min(1e-9)).clamp_min(1e-12)
        h = float(-(p * p.log()).sum().item())
        logger.debug(
            "VisionEncoder.ambiguity: H=%.4f nats max=%.4f", h, math.log(int(n_buckets))
        )
        return h


def _embed_to_cognitive_frame(embed: torch.Tensor) -> torch.Tensor:
    """Project an arbitrary-dim embedding into the cognitive-frame layout."""

    e = embed.detach().to(torch.float32).flatten()
    if e.numel() < SKETCH_DIM:
        e = torch.cat([e, torch.zeros(SKETCH_DIM - e.numel(), dtype=torch.float32)])
    base = F.normalize(e[:SKETCH_DIM], dim=0)
    intent = stable_sketch("vision", dim=SKETCH_DIM)
    scene = stable_sketch(f"scene:{base[:8].tolist()}", dim=SKETCH_DIM)
    tail_len = COGNITIVE_FRAME_DIM - 3 * SKETCH_DIM
    if tail_len < 0:
        raise ValueError(
            f"COGNITIVE_FRAME_DIM ({COGNITIVE_FRAME_DIM}) must be >= 3 * SKETCH_DIM ({3 * SKETCH_DIM}); "
            "check continuous_frame layout constants."
        )
    tail = torch.zeros(tail_len, dtype=torch.float32)
    if tail_len > 0:
        tail[0] = 1.0  # confidence of the visual observation channel
        if tail_len > 8:
            tail[8] = float(base.norm().item())
    out = torch.cat([intent, base, scene, tail])
    return out
