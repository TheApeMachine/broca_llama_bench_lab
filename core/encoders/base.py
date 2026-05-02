"""Base protocol for frozen specialist encoders.

Every encoder follows the same lifecycle:
1. Construct (specifies model ID, device preference — does NOT load weights)
2. Load (downloads/loads weights on first use — lazy)
3. Process (accepts modality-specific input, returns EncoderOutput)
4. Project (optional: map encoder output to substrate's cognitive frame dim)

The EncoderRegistry manages all loaded encoders and provides device budgeting.
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch
import torch.nn.functional as F

from ..system.event_bus import get_default_bus

logger = logging.getLogger(__name__)


def _publish(topic: str, payload: dict) -> None:
    try:
        get_default_bus().publish(topic, payload)
    except Exception:
        pass


@dataclass
class EncoderOutput:
    """Uniform output from any frozen encoder.

    Attributes:
        features: The encoder's raw output embedding [d_encoder].
        projected: Optional projection to substrate dim [d_substrate].
        metadata: Encoder-specific structured output (entities, emotions, etc).
        confidence: How confident the encoder is in its output (0-1).
        latency_ms: How long the encoder took to process.
        encoder_name: Which encoder produced this.
    """
    features: torch.Tensor | None = None
    projected: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    latency_ms: float = 0.0
    encoder_name: str = ""


@runtime_checkable
class Encoder(Protocol):
    """Protocol that all frozen specialist encoders implement."""

    @property
    def name(self) -> str:
        """Stable encoder id (e.g. 'visual_cortex', 'auditory_cortex')."""
        ...

    @property
    def model_id(self) -> str:
        """HuggingFace model ID or path."""
        ...

    @property
    def is_loaded(self) -> bool:
        """Whether weights are currently in memory."""
        ...

    @property
    def output_dim(self) -> int:
        """Dimensionality of the encoder's feature output."""
        ...

    def load(self, device: torch.device | None = None) -> None:
        """Load model weights. Idempotent."""
        ...

    def unload(self) -> None:
        """Release model weights from memory."""
        ...


class BaseEncoder(ABC):
    """Abstract base class for encoder implementations.

    Handles common concerns: lazy loading, device management, timing.
    Subclasses implement _load_model() and the modality-specific process methods.
    """

    def __init__(
        self,
        *,
        name: str,
        model_id: str,
        output_dim: int,
        device: torch.device | str | None = None,
    ):
        self._name = name
        self._model_id = model_id
        self._output_dim = output_dim
        self._device = torch.device(device) if device else None
        self._model: Any = None
        self._processor: Any = None
        self._loaded = False
        self._load_lock = threading.Lock()
        self._total_calls = 0
        self._total_latency_ms = 0.0

    @property
    def name(self) -> str:
        return self._name

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def device(self) -> torch.device:
        if self._device is not None:
            return self._device
        from ..system.device import pick_torch_device
        return pick_torch_device()

    @property
    def stats(self) -> dict[str, Any]:
        avg = self._total_latency_ms / max(1, self._total_calls)
        return {
            "name": self._name,
            "model_id": self._model_id,
            "loaded": self._loaded,
            "device": str(self.device),
            "output_dim": self._output_dim,
            "total_calls": self._total_calls,
            "avg_latency_ms": round(avg, 2),
        }

    def load(self, device: torch.device | None = None) -> None:
        """Load model weights. Thread-safe and idempotent."""
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            if device is not None:
                self._device = device
            start = time.time()
            self._load_model()
            self._loaded = True
            elapsed = (time.time() - start) * 1000
            logger.info(
                "Encoder loaded: %s (%s) on %s in %.0fms",
                self._name, self._model_id, self.device, elapsed,
            )
            _publish(
                "encoder.load",
                {
                    "name": self._name,
                    "model_id": self._model_id,
                    "device": str(self.device),
                    "load_ms": elapsed,
                },
            )

    def unload(self) -> None:
        """Release model weights."""
        with self._load_lock:
            self._model = None
            self._processor = None
            self._loaded = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Encoder unloaded: %s", self._name)
            _publish("encoder.unload", {"name": self._name})

    def _ensure_loaded(self) -> None:
        """Load on first use if not already loaded."""
        if not self._loaded:
            self.load()

    def _record_call(self, latency_ms: float, *, method: str | None = None) -> None:
        self._total_calls += 1
        self._total_latency_ms += latency_ms
        _publish(
            "encoder.call",
            {
                "name": self._name,
                "method": method or "process",
                "latency_ms": float(latency_ms),
                "total_calls": self._total_calls,
                "avg_latency_ms": self._total_latency_ms / max(1, self._total_calls),
            },
        )

    @abstractmethod
    def _load_model(self) -> None:
        """Subclass implements actual model loading."""
        ...


class EncoderRegistry:
    """Central registry managing all loaded encoders.

    Provides:
    - Lazy loading of encoders on first access
    - Device budgeting (track total VRAM usage)
    - Bulk load/unload for session management
    - Stats reporting
    """

    def __init__(self, *, default_device: torch.device | str | None = None):
        self._encoders: dict[str, BaseEncoder] = {}
        self._default_device = (
            torch.device(default_device) if default_device else None
        )
        self._lock = threading.Lock()

    @property
    def default_device(self) -> torch.device:
        if self._default_device is not None:
            return self._default_device
        from ..system.device import pick_torch_device
        return pick_torch_device()

    def register(self, encoder: BaseEncoder) -> None:
        """Register an encoder (does not load it)."""
        with self._lock:
            self._encoders[encoder.name] = encoder
            logger.debug("Registered encoder: %s (%s)", encoder.name, encoder.model_id)

    def get(self, name: str) -> BaseEncoder | None:
        """Get an encoder by name. Returns None if not registered."""
        return self._encoders.get(name)

    def get_or_load(self, name: str) -> BaseEncoder | None:
        """Get an encoder, loading it if needed."""
        encoder = self._encoders.get(name)
        if encoder is not None and not encoder.is_loaded:
            encoder.load(self.default_device)
        return encoder

    def load_all(self) -> None:
        """Load all registered encoders."""
        for encoder in self._encoders.values():
            if not encoder.is_loaded:
                encoder.load(self.default_device)

    def unload_all(self) -> None:
        """Unload all encoders to free memory."""
        for encoder in self._encoders.values():
            if encoder.is_loaded:
                encoder.unload()

    @property
    def loaded_encoders(self) -> list[str]:
        return [name for name, o in self._encoders.items() if o.is_loaded]

    @property
    def all_encoders(self) -> list[str]:
        return list(self._encoders.keys())

    def stats(self) -> dict[str, Any]:
        """Report stats for all encoders."""
        return {
            "n_registered": len(self._encoders),
            "n_loaded": len(self.loaded_encoders),
            "default_device": str(self.default_device),
            "encoders": {name: encoder.stats for name, encoder in self._encoders.items()},
        }

    def __contains__(self, name: str) -> bool:
        return name in self._encoders

    def __len__(self) -> int:
        return len(self._encoders)
