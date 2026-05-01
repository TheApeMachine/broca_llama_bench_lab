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

For the production model, set ``ASI_BROCA_VISION_MODEL`` (default
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
        t = image
    else:
        try:
            from PIL import Image  # type: ignore
        except ImportError as exc:  # pragma: no cover - PIL is a typical dep
            raise RuntimeError("PIL/Pillow is required to decode non-tensor images") from exc
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
    """Deterministic perceptual + DCT-frequency sketch for the no-deps path.

    Combines:
      * a spatial luminance histogram (16 buckets) for global appearance,
      * the low-frequency DCT block (8×8) on a downscaled grayscale copy for
        compositional structure (this is the classical pHash signal),
      * a discrete Fourier magnitude profile (8 radii) for texture orientation.
    """

    g = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
    target = 64
    g = F.adaptive_avg_pool2d(g.unsqueeze(0).unsqueeze(0), (target, target)).squeeze()
    hist = torch.histc(g, bins=16, min=0.0, max=1.0)
    hist = hist / hist.sum().clamp_min(1e-9)
    # 2D DCT-II via separable matrices — small enough at 64×64 to be cheap.
    n = target
    k = torch.arange(n, dtype=torch.float32)
    basis = torch.cos(math.pi / n * (k.unsqueeze(1) + 0.5) * k.unsqueeze(0))
    basis[0] = basis[0] / math.sqrt(2.0)
    basis = basis * math.sqrt(2.0 / n)
    dct2 = basis @ g @ basis.t()
    low_freq = dct2[:8, :8].flatten()
    fft = torch.fft.rfft2(g)
    mag = fft.abs()
    yy, xx = torch.meshgrid(torch.arange(mag.shape[0], dtype=torch.float32), torch.arange(mag.shape[1], dtype=torch.float32), indexing="ij")
    radius = torch.sqrt(yy ** 2 + xx ** 2)
    radial = torch.stack([mag[(radius >= r) & (radius < r + 4)].mean() if mag[(radius >= r) & (radius < r + 4)].numel() else torch.tensor(0.0) for r in torch.arange(0, 32, 4)])
    feature = torch.cat([hist, low_freq, radial.nan_to_num(0.0)])
    feature = F.normalize(feature, dim=0)
    # Project the perceptual features into the substrate's hashed sketch space
    # by mixing with a stable text-sketch derived from the feature digest.
    digest = " ".join(f"{x:.4f}" for x in feature.tolist())
    sketch = stable_sketch(digest, dim=dim)
    pad = torch.zeros(dim, dtype=torch.float32)
    pad[: min(dim, feature.shape[0])] = feature[: min(dim, feature.shape[0])]
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
        self.model_id = model_id or os.environ.get("ASI_BROCA_VISION_MODEL", "openai/clip-vit-base-patch32")
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self._model = None
        self._processor = None
        self._real = False
        if use_real_model:
            self._try_load_real()
        logger.info("VisionEncoder.init: model_id=%s real=%s device=%s", self.model_id, self._real, self.device)

    @property
    def is_real(self) -> bool:
        return self._real

    def _try_load_real(self) -> None:
        try:
            from transformers import AutoModel, AutoProcessor  # type: ignore
        except ImportError:
            logger.warning("VisionEncoder: transformers not available; falling back to perceptual sketch")
            self._real = False
            return
        try:
            self._processor = AutoProcessor.from_pretrained(self.model_id)
            self._model = AutoModel.from_pretrained(self.model_id).to(self.device).eval()
            self._real = True
        except Exception as exc:  # pragma: no cover - network/model issues
            logger.warning("VisionEncoder: failed to load %s (%s); using perceptual sketch", self.model_id, exc)
            self._real = False

    @torch.no_grad()
    def encode_image(self, image: Any) -> torch.Tensor:
        """Map an image to the substrate's cognitive-frame layout."""

        if self._real and self._model is not None and self._processor is not None:
            try:
                from PIL import Image  # type: ignore
            except ImportError:
                self._real = False
            else:
                pil = image if hasattr(image, "convert") else None
                if pil is None:
                    if isinstance(image, (str, Path)):
                        pil = Image.open(image).convert("RGB")
                    elif isinstance(image, (bytes, bytearray)):
                        pil = Image.open(BytesIO(image)).convert("RGB")
                if pil is not None:
                    inputs = self._processor(images=pil, return_tensors="pt").to(self.device)
                    feats = self._model.get_image_features(**inputs) if hasattr(self._model, "get_image_features") else self._model(**inputs).pooler_output
                    embed = F.normalize(feats[0].detach().cpu().to(torch.float32), dim=0)
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
        hist = torch.histc(gray, bins=int(n_buckets), min=0.0, max=1.0)
        p = (hist / hist.sum().clamp_min(1e-9)).clamp_min(1e-12)
        h = float(-(p * p.log()).sum().item())
        logger.debug("VisionEncoder.ambiguity: H=%.4f nats max=%.4f", h, math.log(int(n_buckets)))
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
    tail = torch.zeros(tail_len, dtype=torch.float32)
    if tail_len > 0:
        tail[0] = 1.0  # confidence of the visual observation channel
        if tail_len > 8:
            tail[8] = float(base.norm().item())
    out = torch.cat([intent, base, scene, tail])
    return out
