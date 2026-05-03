"""Auditory encoder: frozen Whisper encoder for speech and audio perception.

Processes audio input into representations for the substrate.

Model: openai/whisper-large-v3-turbo (809M, ~1.6GB fp16)
- Full 32-layer encoder + 4 decoder layers (vs 32 in full large-v3)
- Near-identical accuracy at 50% memory cost
- Supports 99 languages

Provides:
- Transcription (speech → text for routing through the substrate)
- Audio embeddings (encoder features for cross-modal binding)
- Language detection
"""

from __future__ import annotations

import logging
import time
from typing import Any

import torch
import torch.nn.functional as F

from ..workspace import WorkspacePublisher
from .base import BaseEncoder, EncoderOutput

logger = logging.getLogger(__name__)

_WHISPER_MODEL = "openai/whisper-large-v3-turbo"
_WHISPER_DIM = 1280  # Whisper-large encoder hidden size


class AuditoryEncoder(BaseEncoder):
    """Frozen Whisper encoder for audio/speech perception.

    Usage:
        organ = AuditoryEncoder()
        organ.load()

        # Transcribe speech
        text = organ.transcribe(audio_array, sampling_rate=16000)

        # Get audio embeddings (for cross-modal integration)
        output = organ.encode(audio_array, sampling_rate=16000)

        # Full pipeline
        output = organ.process(audio_array, sampling_rate=16000)
        # output.metadata["transcription"] = "Hello world"
        # output.features = [1280] encoder representation
    """

    def __init__(self, *, model_id: str | None = None, device: str | None = None):
        super().__init__(
            name="auditory_cortex",
            model_id=model_id or _WHISPER_MODEL,
            output_dim=_WHISPER_DIM,
            device=device,
        )
        self._pipe: Any = None

    def _load_model(self) -> None:
        """Load Whisper model + processor."""
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        import torch as _torch

        dtype = _torch.float16 if self.device.type in ("cuda", "mps") else _torch.float32

        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self._model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(self.device).eval()

        self._processor = AutoProcessor.from_pretrained(self._model_id)

        # Freeze
        for param in self._model.parameters():
            param.requires_grad = False

        # Also create a pipeline for easy transcription
        from transformers import pipeline
        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=self._model,
            tokenizer=self._processor.tokenizer,
            feature_extractor=self._processor.feature_extractor,
            torch_dtype=dtype,
            device=self.device,
        )

    @torch.no_grad()
    def transcribe(
        self,
        audio: Any,
        *,
        sampling_rate: int = 16000,
        language: str | None = None,
        return_timestamps: bool = False,
    ) -> str:
        """Transcribe audio to text.

        Args:
            audio: numpy array, torch tensor, or file path.
            sampling_rate: Audio sample rate (Whisper expects 16kHz).
            language: Force language code (e.g. "en"). None = auto-detect.
            return_timestamps: Include word-level timestamps.

        Returns:
            Transcribed text string.
        """
        self._ensure_loaded()
        start = time.time()

        generate_kwargs = {}
        if language:
            generate_kwargs["language"] = language

        result = self._pipe(
            audio,
            generate_kwargs=generate_kwargs,
            return_timestamps=return_timestamps,
        )

        text = result.get("text", "") if isinstance(result, dict) else str(result)
        latency = (time.time() - start) * 1000
        self._record_call(latency, method="transcribe")
        stripped = text.strip()
        WorkspacePublisher.emit(
            "encoder.auditory.transcribe",
            {
                "transcription": stripped[:160],
                "language": language or "auto",
                "sampling_rate": int(sampling_rate),
                "latency_ms": latency,
            },
        )
        return stripped

    @torch.no_grad()
    def encode(
        self,
        audio: Any,
        *,
        sampling_rate: int = 16000,
    ) -> EncoderOutput:
        """Encode audio to a feature vector via the Whisper encoder.

        Returns the encoder's pooled hidden states — useful for cross-modal
        binding (e.g., with ImageBind or the substrate's VSA).

        Args:
            audio: numpy array of audio samples.
            sampling_rate: Sample rate (16kHz expected).

        Returns:
            EncoderOutput with features=[1280] encoder representation.
        """
        self._ensure_loaded()
        start = time.time()

        # Process audio through feature extractor
        inputs = self._processor.feature_extractor(
            audio, sampling_rate=sampling_rate, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Run encoder only (no decoder)
        encoder = self._model.get_encoder()
        encoder_outputs = encoder(**inputs)

        # Pool over time dimension for global audio representation
        hidden = encoder_outputs.last_hidden_state[0]  # [T, d_model]
        features = F.normalize(hidden.mean(dim=0).cpu().float(), dim=0)

        elapsed = (time.time() - start) * 1000
        self._record_call(elapsed, method="encode")
        WorkspacePublisher.emit(
            "encoder.auditory.encode",
            {
                "n_frames": int(hidden.shape[0]),
                "feature_dim": int(features.numel()),
                "sampling_rate": int(sampling_rate),
                "latency_ms": elapsed,
            },
        )

        return EncoderOutput(
            features=features,
            metadata={"model": self._model_id, "n_frames": hidden.shape[0]},
            confidence=1.0,
            latency_ms=elapsed,
            encoder_name=self._name,
        )

    def process(self, audio: Any, **kwargs: Any) -> EncoderOutput:
        """Full pipeline: transcribe + encode."""
        self._ensure_loaded()
        start = time.time()

        sampling_rate = kwargs.get("sampling_rate", 16000)

        # Transcribe
        text = self.transcribe(audio, sampling_rate=sampling_rate)

        # Encode
        output = self.encode(audio, sampling_rate=sampling_rate)

        # Augment metadata with transcription
        output.metadata["transcription"] = text
        output.metadata["language"] = kwargs.get("language", "auto")

        return output
