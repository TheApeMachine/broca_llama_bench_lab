"""Multimodal frozen-encoder routing for substrate observations."""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from ..encoders.base import BaseEncoder, EncoderOutput, EncoderRegistry
from ..cognition.observation import CognitiveObservation


class MultimodalPerceptionPipeline:
    """Runs frozen modality encoders and returns typed substrate observations."""

    def __init__(
        self,
        *,
        device: torch.device | str | None = None,
        encoders: Mapping[str, BaseEncoder] | None = None,
    ) -> None:
        self.registry = EncoderRegistry(default_device=device)
        if encoders is None:
            self._register_default_encoders(device=device)
        else:
            for encoder in encoders.values():
                self.registry.register(encoder)

    @property
    def registered_encoders(self) -> list[str]:
        return self.registry.all_encoders

    @property
    def loaded_encoders(self) -> list[str]:
        return self.registry.loaded_encoders

    def stats(self) -> dict[str, Any]:
        return self.registry.stats()

    def perceive_image(self, image: Any, *, source: str = "image") -> CognitiveObservation:
        outputs = {
            "visual_cortex": self._run("visual_cortex", "encode", image),
            "ventral_stream": self._run("ventral_stream", "encode", image),
            "spatial_cortex": self._run("spatial_cortex", "estimate_depth", image),
            "association_cortex": self._run("association_cortex", "encode_image", image),
        }
        return self._observation(
            "image",
            source,
            "visual scene",
            outputs,
            evidence={"streams": sorted(outputs)},
        )

    def perceive_video(self, frames: Any, *, source: str = "video") -> CognitiveObservation:
        first = self._first_frame(frames)
        outputs = {
            "dorsal_stream": self._run("dorsal_stream", "encode_frames", frames),
            "visual_cortex": self._run("visual_cortex", "encode", first),
            "ventral_stream": self._run("ventral_stream", "encode", first),
            "spatial_cortex": self._run("spatial_cortex", "estimate_depth", first),
            "association_cortex": self._run("association_cortex", "encode_image", first),
        }
        return self._observation(
            "video",
            source,
            "temporal visual scene",
            outputs,
            evidence={"streams": sorted(outputs)},
        )

    def perceive_audio(
        self,
        audio: Any,
        *,
        sampling_rate: int = 16000,
        source: str = "audio",
        language: str | None = None,
    ) -> CognitiveObservation:
        auditory = self._encoder("auditory_cortex")
        transcript = str(
            self._call(auditory, "transcribe", audio, sampling_rate=int(sampling_rate), language=language)
        ).strip()
        audio_output = self._call(auditory, "encode", audio, sampling_rate=int(sampling_rate))
        if not isinstance(audio_output, EncoderOutput):
            raise TypeError("AuditoryEncoder.encode must return EncoderOutput")

        outputs = {
            "auditory_cortex": audio_output,
            "association_cortex_audio": self._run("association_cortex", "encode_audio", audio),
        }
        if transcript:
            outputs["association_cortex_text"] = self._run("association_cortex", "encode_text", transcript)

        return self._observation(
            "audio",
            source,
            transcript or "audio observation",
            outputs,
            evidence={
                "transcription": transcript,
                "sampling_rate": int(sampling_rate),
                "language": language or "auto",
                "streams": sorted(outputs),
            },
        )

    def _register_default_encoders(self, *, device: torch.device | str | None) -> None:
        from ..encoders.auditory import AuditoryEncoder
        from ..encoders.binding import BindingEncoder
        from ..encoders.perception import DINOv2Encoder, DepthEncoder, IJEPAEncoder, VJEPAEncoder

        for encoder in (
            DINOv2Encoder(device=str(device) if device is not None else None),
            IJEPAEncoder(device=str(device) if device is not None else None),
            VJEPAEncoder(device=str(device) if device is not None else None),
            DepthEncoder(device=str(device) if device is not None else None),
            AuditoryEncoder(device=str(device) if device is not None else None),
            BindingEncoder(device=str(device) if device is not None else None),
        ):
            self.registry.register(encoder)

    def _encoder(self, name: str) -> BaseEncoder:
        encoder = self.registry.get_or_load(name)
        if encoder is None:
            raise RuntimeError(f"multimodal encoder {name!r} is not registered")
        return encoder

    def _run(self, name: str, method: str, *args: Any, **kwargs: Any) -> EncoderOutput:
        encoder = self._encoder(name)
        output = self._call(encoder, method, *args, **kwargs)
        if not isinstance(output, EncoderOutput):
            raise TypeError(f"{name}.{method} must return EncoderOutput")
        if output.features is None:
            raise ValueError(f"{name}.{method} returned no features")
        return output

    @staticmethod
    def _call(encoder: BaseEncoder, method: str, *args: Any, **kwargs: Any) -> Any:
        fn = getattr(encoder, method, None)
        if not callable(fn):
            raise RuntimeError(f"encoder {encoder.name!r} does not implement {method!r}")
        return fn(*args, **kwargs)

    def _observation(
        self,
        modality: str,
        source: str,
        answer: str,
        outputs: Mapping[str, EncoderOutput],
        *,
        evidence: dict[str, Any],
    ) -> CognitiveObservation:
        if not outputs:
            raise ValueError("multimodal observation requires at least one encoder output")
        features = self._combine_features(outputs)
        confidence = min(self._confidence(output) for output in outputs.values())
        encoder_evidence = {
            name: self._output_evidence(output)
            for name, output in outputs.items()
        }
        return CognitiveObservation(
            modality=modality,
            source=source,
            features=features,
            confidence=confidence,
            answer=answer,
            evidence={
                **dict(evidence),
                "encoder_outputs": encoder_evidence,
            },
        )

    @staticmethod
    def _combine_features(outputs: Mapping[str, EncoderOutput]) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        for output in outputs.values():
            if output.features is None:
                raise ValueError(f"encoder output {output.encoder_name!r} has no features")
            part = output.features.detach().float().cpu().flatten()
            if part.numel() <= 0:
                raise ValueError(f"encoder output {output.encoder_name!r} has empty features")
            if not torch.isfinite(part).all():
                raise ValueError(f"encoder output {output.encoder_name!r} has non-finite features")
            parts.append(F.normalize(part, dim=0))
        return F.normalize(torch.cat(parts), dim=0)

    @staticmethod
    def _confidence(output: EncoderOutput) -> float:
        conf = float(output.confidence)
        if not math.isfinite(conf):
            raise ValueError(f"encoder output {output.encoder_name!r} confidence is not finite")
        if conf < 0.0 or conf > 1.0:
            raise ValueError(f"encoder output {output.encoder_name!r} confidence must be in [0, 1]")
        return conf

    def _output_evidence(self, output: EncoderOutput) -> dict[str, Any]:
        if output.features is None:
            raise ValueError(f"encoder output {output.encoder_name!r} has no features")
        return {
            "encoder_name": output.encoder_name,
            "feature_dim": int(output.features.numel()),
            "confidence": self._confidence(output),
            "latency_ms": float(output.latency_ms),
            "metadata": self._json_safe(output.metadata),
        }

    def _json_safe(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, bool)):
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("metadata contains non-finite float")
            return value
        if isinstance(value, torch.Tensor):
            t = value.detach().float().cpu()
            if not torch.isfinite(t).all():
                raise ValueError("metadata tensor contains non-finite values")
            return {
                "tensor_shape": list(t.shape),
                "tensor_norm": float(t.norm().item()),
            }
        if isinstance(value, Mapping):
            return {str(k): self._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(v) for v in value]
        return str(value)

    @staticmethod
    def _first_frame(frames: Any) -> Any:
        if isinstance(frames, torch.Tensor):
            if frames.ndim < 4 or int(frames.shape[0]) <= 0:
                raise ValueError("video tensor must have shape [T, ...] with T > 0")
            return frames[0]
        if isinstance(frames, (list, tuple)):
            if not frames:
                raise ValueError("video frame list must be non-empty")
            return frames[0]
        raise TypeError(f"unsupported video frames type: {type(frames)!r}")
