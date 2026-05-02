"""Multimodal organ routing for substrate observations."""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from ..organs.base import BaseOrgan, OrganOutput, OrganRegistry
from .observation import CognitiveObservation


class MultimodalPerceptionPipeline:
    """Runs frozen sensory organs and returns typed substrate observations."""

    def __init__(
        self,
        *,
        device: torch.device | str | None = None,
        organs: Mapping[str, BaseOrgan] | None = None,
    ) -> None:
        self.registry = OrganRegistry(default_device=device)
        if organs is None:
            self._register_default_organs(device=device)
        else:
            for organ in organs.values():
                self.registry.register(organ)

    @property
    def registered_organs(self) -> list[str]:
        return self.registry.all_organs

    @property
    def loaded_organs(self) -> list[str]:
        return self.registry.loaded_organs

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
        auditory = self._organ("auditory_cortex")
        transcript = str(
            self._call(auditory, "transcribe", audio, sampling_rate=int(sampling_rate), language=language)
        ).strip()
        audio_output = self._call(auditory, "encode", audio, sampling_rate=int(sampling_rate))
        if not isinstance(audio_output, OrganOutput):
            raise TypeError("AuditoryOrgan.encode must return OrganOutput")

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

    def _register_default_organs(self, *, device: torch.device | str | None) -> None:
        from ..organs.auditory import AuditoryOrgan
        from ..organs.binding import BindingOrgan
        from ..organs.perception import DINOv2Organ, DepthOrgan, IJEPAOrgan, VJEPAOrgan

        for organ in (
            DINOv2Organ(device=str(device) if device is not None else None),
            IJEPAOrgan(device=str(device) if device is not None else None),
            VJEPAOrgan(device=str(device) if device is not None else None),
            DepthOrgan(device=str(device) if device is not None else None),
            AuditoryOrgan(device=str(device) if device is not None else None),
            BindingOrgan(device=str(device) if device is not None else None),
        ):
            self.registry.register(organ)

    def _organ(self, name: str) -> BaseOrgan:
        organ = self.registry.get_or_load(name)
        if organ is None:
            raise RuntimeError(f"multimodal organ {name!r} is not registered")
        return organ

    def _run(self, name: str, method: str, *args: Any, **kwargs: Any) -> OrganOutput:
        organ = self._organ(name)
        output = self._call(organ, method, *args, **kwargs)
        if not isinstance(output, OrganOutput):
            raise TypeError(f"{name}.{method} must return OrganOutput")
        if output.features is None:
            raise ValueError(f"{name}.{method} returned no features")
        return output

    @staticmethod
    def _call(organ: BaseOrgan, method: str, *args: Any, **kwargs: Any) -> Any:
        fn = getattr(organ, method, None)
        if not callable(fn):
            raise RuntimeError(f"organ {organ.name!r} does not implement {method!r}")
        return fn(*args, **kwargs)

    def _observation(
        self,
        modality: str,
        source: str,
        answer: str,
        outputs: Mapping[str, OrganOutput],
        *,
        evidence: dict[str, Any],
    ) -> CognitiveObservation:
        if not outputs:
            raise ValueError("multimodal observation requires at least one organ output")
        features = self._combine_features(outputs)
        confidence = min(self._confidence(output) for output in outputs.values())
        organ_evidence = {
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
                "organ_outputs": organ_evidence,
            },
        )

    @staticmethod
    def _combine_features(outputs: Mapping[str, OrganOutput]) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        for output in outputs.values():
            if output.features is None:
                raise ValueError(f"organ output {output.organ_name!r} has no features")
            part = output.features.detach().float().cpu().flatten()
            if part.numel() <= 0:
                raise ValueError(f"organ output {output.organ_name!r} has empty features")
            if not torch.isfinite(part).all():
                raise ValueError(f"organ output {output.organ_name!r} has non-finite features")
            parts.append(F.normalize(part, dim=0))
        return F.normalize(torch.cat(parts), dim=0)

    @staticmethod
    def _confidence(output: OrganOutput) -> float:
        conf = float(output.confidence)
        if not math.isfinite(conf):
            raise ValueError(f"organ output {output.organ_name!r} confidence is not finite")
        if conf < 0.0 or conf > 1.0:
            raise ValueError(f"organ output {output.organ_name!r} confidence must be in [0, 1]")
        return conf

    def _output_evidence(self, output: OrganOutput) -> dict[str, Any]:
        if output.features is None:
            raise ValueError(f"organ output {output.organ_name!r} has no features")
        return {
            "organ_name": output.organ_name,
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
