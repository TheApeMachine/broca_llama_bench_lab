"""Binding organ: ImageBind for multi-sensory association and cross-modal retrieval.

Brain analogy: Multi-sensory association cortex (superior temporal sulcus, insula).
Binds what you see with what you hear, feel, and sense into a unified embedding
space — enabling cross-modal transfer (audio → image retrieval, depth → text, etc.).

Model: nielsr/imagebind-huge (1.13B params, ~2.3GB fp16)
- Single shared embedding space across 6 modalities
- Zero-shot cross-modal retrieval without paired multi-sensory training data
- Modalities: image, text, audio, depth, thermal, IMU

License: CC-BY-NC-SA 4.0 (NON-COMMERCIAL ONLY)

The organ enables the substrate to:
- Bind visual observations with audio context (video understanding)
- Ground language in multi-sensory experience
- Retrieve memories across modalities (hear a sound → recall an image)
- Compute cross-modal similarity for the Hopfield associative memory
"""

from __future__ import annotations

import logging
import time
from typing import Any, Literal

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

_IMAGEBIND_MODEL = "nielsr/imagebind-huge"
_IMAGEBIND_DIM = 1024  # ImageBind's shared embedding dimension


Modality = Literal["image", "text", "audio", "depth", "thermal", "imu"]


class BindingOrgan(BaseOrgan):
    """Frozen ImageBind model for multi-sensory embedding.

    Maps inputs from any of 6 modalities into a shared 1024-dim space
    where cross-modal similarity is meaningful. Two inputs from different
    modalities that co-occur in nature (e.g., a dog image and a bark sound)
    will have high cosine similarity.

    Usage:
        organ = BindingOrgan()
        organ.load()

        # Encode different modalities into same space
        img_emb = organ.encode_image(pil_image)
        txt_emb = organ.encode_text("a dog barking")
        aud_emb = organ.encode_audio(bark_audio)

        # Cross-modal similarity
        sim = torch.cosine_similarity(img_emb.features, aud_emb.features, dim=0)
        # High similarity if image and audio are semantically related

    Note: CC-BY-NC-SA license — non-commercial use only.
    """

    def __init__(self, *, model_id: str | None = None, device: str | None = None):
        super().__init__(
            name="association_cortex",
            model_id=model_id or _IMAGEBIND_MODEL,
            output_dim=_IMAGEBIND_DIM,
            device=device,
        )

    def _load_model(self) -> None:
        """Load ImageBind model."""
        try:
            from imagebind.models import imagebind_model
            from imagebind.models.imagebind_model import ModalityType
        except ImportError as exc:
            raise ImportError(
                "BindingOrgan requires the native `imagebind` package "
                "(pip install git+https://github.com/facebookresearch/ImageBind). "
                "The transformers backend is not used because it cannot encode all wired modalities."
            ) from exc

        self._model = imagebind_model.imagebind_huge(pretrained=True).to(self.device).eval()
        self._modality_type = ModalityType
        self._backend = "imagebind_native"

        # Freeze
        for param in self._model.parameters():
            param.requires_grad = False
        logger.info("BindingOrgan: loaded via %s backend", self._backend)

    @torch.no_grad()
    def encode_image(self, image: Any) -> OrganOutput:
        """Encode an image into the shared embedding space."""
        self._ensure_loaded()
        start = time.time()

        if self._backend != "imagebind_native":
            raise RuntimeError("BindingOrgan must be loaded with the native ImageBind backend")

        from imagebind import data as ib_data
        if isinstance(image, str):
            inputs = {self._modality_type.VISION: ib_data.load_and_transform_vision_data([image], self.device)}
        else:
            inputs = {self._modality_type.VISION: ib_data.load_and_transform_vision_data_from_pil([image], self.device)}
        embeddings = self._model(inputs)
        features = F.normalize(embeddings[self._modality_type.VISION][0].cpu().float(), dim=0)

        elapsed = (time.time() - start) * 1000
        self._record_call(elapsed, method="encode_image")
        _publish(
            "organ.binding.encode",
            {"modality": "image", "latency_ms": elapsed, "feature_dim": int(features.numel())},
        )
        return OrganOutput(features=features, metadata={"modality": "image"}, latency_ms=elapsed, organ_name=self._name)

    @torch.no_grad()
    def encode_text(self, text: str) -> OrganOutput:
        """Encode text into the shared embedding space."""
        self._ensure_loaded()
        start = time.time()

        if self._backend != "imagebind_native":
            raise RuntimeError("BindingOrgan must be loaded with the native ImageBind backend")

        from imagebind import data as ib_data
        inputs = {self._modality_type.TEXT: ib_data.load_and_transform_text([text], self.device)}
        embeddings = self._model(inputs)
        features = F.normalize(embeddings[self._modality_type.TEXT][0].cpu().float(), dim=0)

        elapsed = (time.time() - start) * 1000
        self._record_call(elapsed, method="encode_text")
        _publish(
            "organ.binding.encode",
            {
                "modality": "text",
                "text": text[:120],
                "latency_ms": elapsed,
                "feature_dim": int(features.numel()),
            },
        )
        return OrganOutput(features=features, metadata={"modality": "text", "text": text[:100]}, latency_ms=elapsed, organ_name=self._name)

    @torch.no_grad()
    def encode_audio(self, audio: Any) -> OrganOutput:
        """Encode audio into the shared embedding space."""
        self._ensure_loaded()
        start = time.time()

        if self._backend != "imagebind_native":
            raise RuntimeError("BindingOrgan must be loaded with the native ImageBind backend")

        from imagebind import data as ib_data
        if isinstance(audio, str):
            inputs = {self._modality_type.AUDIO: ib_data.load_and_transform_audio_data([audio], self.device)}
        else:
            inputs = {self._modality_type.AUDIO: audio.unsqueeze(0).to(self.device) if isinstance(audio, torch.Tensor) else audio}
        embeddings = self._model(inputs)
        features = F.normalize(embeddings[self._modality_type.AUDIO][0].cpu().float(), dim=0)

        elapsed = (time.time() - start) * 1000
        self._record_call(elapsed, method="encode_audio")
        _publish(
            "organ.binding.encode",
            {"modality": "audio", "latency_ms": elapsed, "feature_dim": int(features.numel())},
        )
        return OrganOutput(features=features, metadata={"modality": "audio"}, latency_ms=elapsed, organ_name=self._name)

    @torch.no_grad()
    def cross_modal_similarity(self, output_a: OrganOutput, output_b: OrganOutput) -> float:
        """Compute cosine similarity between two organ outputs from any modalities."""
        if output_a.features is None or output_b.features is None:
            raise ValueError("cross_modal_similarity requires features from both outputs")
        return float(F.cosine_similarity(
            output_a.features.unsqueeze(0),
            output_b.features.unsqueeze(0),
        ).item())

    def process(self, input_data: Any, **kwargs: Any) -> OrganOutput:
        """Route to appropriate encoder based on input type."""
        modality = kwargs.get("modality", "auto")

        if modality == "text" or isinstance(input_data, str):
            return self.encode_text(input_data)
        elif modality == "audio":
            return self.encode_audio(input_data)
        else:
            return self.encode_image(input_data)
