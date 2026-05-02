"""Perception organs: frozen vision encoders as visual cortex.

Brain analogies:
- V-JEPA (dorsal stream / area MT): motion, temporal dynamics, world prediction
- I-JEPA (ventral stream / V4-IT): object recognition, semantic features
- DINOv2 (primary visual cortex V1-V2): general-purpose visual features
- Depth Anything (parietal 'where' pathway): spatial reasoning, depth

All models are ViT-based, produce fixed-dim feature vectors, and run as a
single forward pass (no autoregressive decoding). On M4 Max with 128GB,
all four can be loaded simultaneously (~5.5 GB total).

Models:
- V-JEPA2: facebook/vjepa2-vith-fpc64-256 (632M, 1.3GB fp16)
- I-JEPA: facebook/ijepa_vith14_1k (632M, 1.3GB fp16)
- DINOv2: facebook/dinov2-giant (1.05B, 2.2GB fp16) or dinov2-large (307M)
- Depth Anything: depth-anything/Depth-Anything-V2-Large-hf (335M, 0.7GB fp16)
"""

from __future__ import annotations

import logging
import time
from typing import Any

import torch
import torch.nn.functional as F

from ..system.event_bus import get_default_bus
from .base import BaseOrgan, OrganOutput

logger = logging.getLogger(__name__)


def _publish(topic: str, payload: dict) -> None:
    try:
        get_default_bus().publish(topic, payload)
    except Exception:
        pass

# Model IDs
_VJEPA_MODEL = "facebook/vjepa2-vith-fpc64-256"
_IJEPA_MODEL = "facebook/ijepa_vith14_1k"
_DINOV2_MODEL = "facebook/dinov2-large"  # Large (307M) as default; Giant (1.05B) available
_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Large-hf"

# Output dimensions
_VIT_H_DIM = 1280  # ViT-Huge hidden size (V-JEPA, I-JEPA)
_VIT_L_DIM = 1024  # ViT-Large hidden size (DINOv2-large, Depth)
_VIT_G_DIM = 1536  # ViT-Giant hidden size (DINOv2-giant)


class DINOv2Organ(BaseOrgan):
    """Frozen DINOv2 encoder — general-purpose visual features.

    Brain analogy: Primary + early association visual cortex (V1-V4).
    Produces rich patch-level and global visual representations without
    any task-specific training.

    Usage:
        organ = DINOv2Organ()
        organ.load()
        output = organ.encode(image_tensor)  # [3, H, W] -> OrganOutput
    """

    def __init__(self, *, model_id: str | None = None, device: str | None = None):
        mid = model_id or _DINOV2_MODEL
        dim = _VIT_G_DIM if "giant" in mid else _VIT_L_DIM
        super().__init__(name="visual_cortex", model_id=mid, output_dim=dim, device=device)

    def _load_model(self) -> None:
        from transformers import AutoModel, AutoImageProcessor
        self._processor = AutoImageProcessor.from_pretrained(self._model_id)
        self._model = AutoModel.from_pretrained(self._model_id).to(self.device).eval()
        # Freeze
        for param in self._model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode(self, image: Any) -> OrganOutput:
        """Encode an image to a global feature vector.

        Args:
            image: PIL Image, torch tensor [3,H,W], or file path.

        Returns:
            OrganOutput with features=[d_model] global CLS token representation.
        """
        self._ensure_loaded()
        start = time.time()

        # Process image
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self._model(**inputs)
        # DINOv2 returns last_hidden_state; CLS token is index 0
        cls_features = outputs.last_hidden_state[:, 0]  # [1, d_model]
        features = F.normalize(cls_features[0].cpu().float(), dim=0)

        elapsed = (time.time() - start) * 1000
        self._record_call(elapsed, method="encode")
        n_patches = int(outputs.last_hidden_state.shape[1] - 1)
        _publish(
            "organ.perception.visual",
            {
                "stream": "visual_cortex",
                "n_patches": n_patches,
                "feature_dim": int(features.numel()),
                "latency_ms": elapsed,
            },
        )

        return OrganOutput(
            features=features,
            metadata={"model": self._model_id, "n_patches": n_patches},
            confidence=1.0,
            latency_ms=elapsed,
            organ_name=self._name,
        )

    @torch.no_grad()
    def encode_patches(self, image: Any) -> tuple[torch.Tensor, OrganOutput]:
        """Return both patch-level features and global CLS token.

        Returns:
            (patch_features [n_patches, d_model], global OrganOutput)
        """
        self._ensure_loaded()
        start = time.time()

        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self._model(**inputs)
        all_features = outputs.last_hidden_state[0].cpu().float()  # [1+n_patches, d_model]
        cls_features = F.normalize(all_features[0], dim=0)
        patch_features = all_features[1:]  # [n_patches, d_model]

        elapsed = (time.time() - start) * 1000
        self._record_call(elapsed, method="encode_patches")
        _publish(
            "organ.perception.visual",
            {
                "stream": "visual_cortex",
                "mode": "patches",
                "n_patches": int(patch_features.shape[0]),
                "latency_ms": elapsed,
            },
        )

        output = OrganOutput(
            features=cls_features,
            metadata={"n_patches": patch_features.shape[0]},
            confidence=1.0,
            latency_ms=elapsed,
            organ_name=self._name,
        )
        return patch_features, output

    def process(self, image: Any, **kwargs: Any) -> OrganOutput:
        return self.encode(image)


class IJEPAOrgan(BaseOrgan):
    """Frozen I-JEPA encoder — semantic visual features via predictive learning.

    Brain analogy: Ventral visual stream (V4 / inferotemporal cortex).
    Learns high-level semantic structure by predicting masked patch
    representations, NOT pixels. This means features encode meaning
    (object identity, category, scene) rather than texture.

    Usage:
        organ = IJEPAOrgan()
        organ.load()
        output = organ.encode(image_tensor)
    """

    def __init__(self, *, model_id: str | None = None, device: str | None = None):
        super().__init__(
            name="ventral_stream",
            model_id=model_id or _IJEPA_MODEL,
            output_dim=_VIT_H_DIM,
            device=device,
        )

    def _load_model(self) -> None:
        from transformers import AutoModel, AutoImageProcessor
        self._processor = AutoImageProcessor.from_pretrained(self._model_id)
        self._model = AutoModel.from_pretrained(self._model_id).to(self.device).eval()
        for param in self._model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode(self, image: Any) -> OrganOutput:
        """Encode image to semantic feature vector."""
        self._ensure_loaded()
        start = time.time()

        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self._model(**inputs)
        # Average pool over patch tokens for global representation
        hidden = outputs.last_hidden_state[0]  # [n_tokens, d]
        # Skip CLS if present, average patches
        if hidden.shape[0] > 1:
            features = F.normalize(hidden[1:].mean(dim=0).cpu().float(), dim=0)
        else:
            features = F.normalize(hidden[0].cpu().float(), dim=0)

        elapsed = (time.time() - start) * 1000
        self._record_call(elapsed, method="encode")
        _publish(
            "organ.perception.ventral",
            {
                "stream": "ventral_stream",
                "n_tokens": int(hidden.shape[0]),
                "feature_dim": int(features.numel()),
                "latency_ms": elapsed,
            },
        )

        return OrganOutput(
            features=features,
            metadata={"model": self._model_id},
            confidence=1.0,
            latency_ms=elapsed,
            organ_name=self._name,
        )

    def process(self, image: Any, **kwargs: Any) -> OrganOutput:
        return self.encode(image)


class VJEPAOrgan(BaseOrgan):
    """Frozen V-JEPA encoder — temporal/motion features from video.

    Brain analogy: Dorsal visual stream (area MT/MST).
    Predicts future frame representations from context — learns the
    dynamics of the visual world without pixel reconstruction.

    This is LeCun's world model: it answers "what happens next?" in
    abstract representation space.

    Usage:
        organ = VJEPAOrgan()
        organ.load()
        output = organ.encode_frames(video_frames)  # [T, 3, H, W]
    """

    def __init__(self, *, model_id: str | None = None, device: str | None = None):
        super().__init__(
            name="dorsal_stream",
            model_id=model_id or _VJEPA_MODEL,
            output_dim=_VIT_H_DIM,
            device=device,
        )

    def _load_model(self) -> None:
        """Load V-JEPA2 model."""
        try:
            from transformers import AutoModel, AutoVideoProcessor
            self._processor = AutoVideoProcessor.from_pretrained(self._model_id)
            self._model = AutoModel.from_pretrained(self._model_id).to(self.device).eval()
        except Exception:
            # V-JEPA2 may require specific model class
            from transformers import AutoModel, AutoProcessor
            self._processor = AutoProcessor.from_pretrained(self._model_id)
            self._model = AutoModel.from_pretrained(self._model_id).to(self.device).eval()
        for param in self._model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode_frames(self, frames: torch.Tensor | list) -> OrganOutput:
        """Encode video frames to temporal feature vector.

        Args:
            frames: Video tensor [T, 3, H, W] or list of PIL images.

        Returns:
            OrganOutput with features representing temporal dynamics.
        """
        self._ensure_loaded()
        start = time.time()

        inputs = self._processor(videos=frames, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self._model(**inputs)
        # Pool over spatial and temporal dimensions
        hidden = outputs.last_hidden_state[0]  # [n_tokens, d]
        features = F.normalize(hidden.mean(dim=0).cpu().float(), dim=0)

        elapsed = (time.time() - start) * 1000
        self._record_call(elapsed, method="encode_frames")
        n_frames = len(frames) if hasattr(frames, "__len__") else 0
        _publish(
            "organ.perception.dorsal",
            {
                "stream": "dorsal_stream",
                "mode": "frames",
                "n_frames": int(n_frames),
                "n_tokens": int(hidden.shape[0]),
                "latency_ms": elapsed,
            },
        )

        return OrganOutput(
            features=features,
            metadata={"model": self._model_id, "n_frames": n_frames},
            confidence=1.0,
            latency_ms=elapsed,
            organ_name=self._name,
        )

    @torch.no_grad()
    def encode_single_frame(self, image: Any) -> OrganOutput:
        """Encode a single image through the video model (treats as 1-frame clip)."""
        self._ensure_loaded()
        start = time.time()

        # Wrap single image as 1-frame video
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self._model(**inputs)
        hidden = outputs.last_hidden_state[0]
        features = F.normalize(hidden.mean(dim=0).cpu().float(), dim=0)

        elapsed = (time.time() - start) * 1000
        self._record_call(elapsed, method="encode_single_frame")
        _publish(
            "organ.perception.dorsal",
            {
                "stream": "dorsal_stream",
                "mode": "single_frame",
                "n_tokens": int(hidden.shape[0]),
                "latency_ms": elapsed,
            },
        )

        return OrganOutput(
            features=features,
            metadata={"model": self._model_id, "mode": "single_frame"},
            confidence=1.0,
            latency_ms=elapsed,
            organ_name=self._name,
        )

    def process(self, input_data: Any, **kwargs: Any) -> OrganOutput:
        if isinstance(input_data, torch.Tensor) and input_data.ndim == 4:
            return self.encode_frames(input_data)
        return self.encode_single_frame(input_data)


class DepthOrgan(BaseOrgan):
    """Frozen Depth Anything V2 — monocular depth estimation.

    Brain analogy: Parietal 'where' pathway.
    Gives the substrate a sense of spatial geometry, distance, and layout.
    The depth map can be used for spatial reasoning queries.

    Usage:
        organ = DepthOrgan()
        organ.load()
        output = organ.estimate_depth(image)
        # output.metadata["depth_map"] = [H, W] tensor
    """

    def __init__(self, *, model_id: str | None = None, device: str | None = None):
        super().__init__(
            name="spatial_cortex",
            model_id=model_id or _DEPTH_MODEL,
            output_dim=_VIT_L_DIM,
            device=device,
        )

    def _load_model(self) -> None:
        from transformers import AutoModelForDepthEstimation, AutoImageProcessor
        self._processor = AutoImageProcessor.from_pretrained(self._model_id)
        self._model = AutoModelForDepthEstimation.from_pretrained(self._model_id).to(self.device).eval()
        for param in self._model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def estimate_depth(self, image: Any) -> OrganOutput:
        """Estimate depth map from a single image.

        Returns:
            OrganOutput with:
            - features: global depth statistics vector
            - metadata["depth_map"]: [H, W] relative depth tensor
            - metadata["depth_stats"]: {mean, std, min, max, median}
        """
        self._ensure_loaded()
        start = time.time()

        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self._model(**inputs)
        depth_map = outputs.predicted_depth[0].cpu().float()  # [H, W]

        # Normalize depth to [0, 1]
        d_min, d_max = depth_map.min(), depth_map.max()
        if d_max - d_min > 1e-6:
            depth_normalized = (depth_map - d_min) / (d_max - d_min)
        else:
            depth_normalized = torch.zeros_like(depth_map)

        # Compute spatial statistics as feature vector
        stats = {
            "mean": float(depth_normalized.mean().item()),
            "std": float(depth_normalized.std().item()),
            "min": float(depth_normalized.min().item()),
            "max": float(depth_normalized.max().item()),
            "median": float(depth_normalized.median().item()),
        }

        # Build a feature vector from depth histogram + spatial layout
        # 16 histogram bins + 64 spatial (8x8 grid averages) = 80-dim
        bins = torch.clamp((depth_normalized.flatten() * 16).long(), 0, 15)
        hist = torch.bincount(bins, minlength=16).float()
        hist = hist / hist.sum().clamp_min(1e-6)
        spatial = F.adaptive_avg_pool2d(depth_normalized.unsqueeze(0).unsqueeze(0), (8, 8)).flatten()
        features = F.normalize(torch.cat([hist, spatial]), dim=0)

        # Pad to output_dim
        padded = torch.zeros(self._output_dim, dtype=torch.float32)
        padded[:features.shape[0]] = features

        elapsed = (time.time() - start) * 1000
        self._record_call(elapsed, method="estimate_depth")
        _publish(
            "organ.perception.spatial",
            {
                "stream": "spatial_cortex",
                "depth_stats": stats,
                "depth_shape": list(depth_map.shape),
                "latency_ms": elapsed,
            },
        )

        return OrganOutput(
            features=padded,
            metadata={
                "depth_map": depth_normalized,
                "depth_stats": stats,
                "model": self._model_id,
            },
            confidence=1.0,
            latency_ms=elapsed,
            organ_name=self._name,
        )

    def process(self, image: Any, **kwargs: Any) -> OrganOutput:
        return self.estimate_depth(image)


# Convenience factory
def create_perception_organs(
    *,
    device: str | None = None,
    include: set[str] | None = None,
) -> dict[str, BaseOrgan]:
    """Create all perception organs (or a subset).

    Args:
        device: Torch device string.
        include: Set of organ names to include. None = all.
            Valid names: "visual_cortex", "ventral_stream", "dorsal_stream", "spatial_cortex"

    Returns:
        Dict mapping organ name to organ instance (not yet loaded).
    """
    all_organs = {
        "visual_cortex": lambda: DINOv2Organ(device=device),
        "ventral_stream": lambda: IJEPAOrgan(device=device),
        "dorsal_stream": lambda: VJEPAOrgan(device=device),
        "spatial_cortex": lambda: DepthOrgan(device=device),
    }

    if include is None:
        include = set(all_organs.keys())

    return {name: factory() for name, factory in all_organs.items() if name in include}
