"""GLiClass-backed hierarchical zero-shot text classification."""

from __future__ import annotations

import time
from typing import Any, Sequence

from transformers import AutoTokenizer

from ..system.event_bus import get_default_bus
from .base import BaseEncoder, EncoderOutput


CLASSIFICATION_MODEL_ID = "knowledgator/gliclass-small-v1.0"


class SemanticClassificationEncoder(BaseEncoder):
    """Frozen GLiClass encoder for prompt-conditioned hierarchical labels."""

    def __init__(
        self,
        *,
        model_id: str | None = None,
        device: str | None = None,
    ):
        super().__init__(
            name="semantic_classification",
            model_id=model_id or CLASSIFICATION_MODEL_ID,
            output_dim=0,
            device=device,
        )
        self._pipeline: Any = None

    def _load_model(self) -> None:
        from gliclass import GLiClassModel, ZeroShotClassificationPipeline

        self._model = GLiClassModel.from_pretrained(self._model_id)
        self._processor = AutoTokenizer.from_pretrained(self._model_id)
        self._pipeline = ZeroShotClassificationPipeline(
            self._model,
            self._processor,
            classification_type="multi-label",
            device=str(self.device),
            progress_bar=False,
        )

    def classify_axes(
        self,
        text: str,
        labels: dict[str, Sequence[str]],
        *,
        prompt: str,
        examples: Sequence[dict[str, Any]],
    ) -> dict[str, dict[str, float]]:
        """Classify one text against hierarchical axes and return all scores."""

        if not text.strip():
            raise ValueError("SemanticClassificationEncoder.classify_axes requires non-empty text")
        self._validate_labels(labels)
        self._ensure_loaded()
        start = time.time()
        raw_batch = self._pipeline(
            text,
            labels,
            examples=[dict(ex) for ex in examples],
            prompt=prompt,
            return_hierarchical=True,
        )
        if not isinstance(raw_batch, list) or len(raw_batch) != 1:
            raise RuntimeError(
                f"SemanticClassificationEncoder.classify_axes expected one result, got {raw_batch!r}"
            )
        axes = self._normalize_axes(raw_batch[0], labels)
        latency = (time.time() - start) * 1000
        self._record_call(latency, method="classify_axes")
        self._publish(
            "encoder.semantic_classification.axes",
            {
                "text": text[:120],
                "axes": axes,
                "latency_ms": latency,
            },
        )
        return axes

    def process(self, text: str, **kwargs: Any) -> EncoderOutput:
        labels = kwargs.get("labels")
        prompt = kwargs.get("prompt")
        examples = kwargs.get("examples")
        if not isinstance(labels, dict):
            raise ValueError("SemanticClassificationEncoder.process requires dict labels")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("SemanticClassificationEncoder.process requires a non-empty prompt")
        if not isinstance(examples, Sequence) or isinstance(examples, (str, bytes)):
            raise ValueError("SemanticClassificationEncoder.process requires examples")
        start = time.time()
        axes = self.classify_axes(text, labels, prompt=prompt, examples=examples)
        elapsed = (time.time() - start) * 1000
        return EncoderOutput(
            metadata={"axes": axes},
            confidence=max((score for axis in axes.values() for score in axis.values()), default=0.0),
            latency_ms=elapsed,
            encoder_name=self._name,
        )

    @staticmethod
    def _validate_labels(labels: dict[str, Sequence[str]]) -> None:
        if not labels:
            raise ValueError("SemanticClassificationEncoder.classify_axes requires labels")
        for axis, axis_labels in labels.items():
            if not isinstance(axis, str) or not axis.strip():
                raise ValueError(
                    f"SemanticClassificationEncoder requires non-empty axis names, got {axis!r}"
                )
            if not isinstance(axis_labels, Sequence) or isinstance(axis_labels, (str, bytes)):
                raise ValueError(
                    f"SemanticClassificationEncoder requires a label sequence for axis {axis!r}"
                )
            if not axis_labels:
                raise ValueError(
                    f"SemanticClassificationEncoder requires at least one label for axis {axis!r}"
                )
            for label in axis_labels:
                if not isinstance(label, str) or not label.strip():
                    raise ValueError(
                        f"SemanticClassificationEncoder requires non-empty labels for axis {axis!r}"
                    )

    def _normalize_axes(
        self,
        raw: Any,
        labels: dict[str, Sequence[str]],
    ) -> dict[str, dict[str, float]]:
        if not isinstance(raw, dict):
            raise RuntimeError(f"SemanticClassificationEncoder expected dict output, got {raw!r}")
        out: dict[str, dict[str, float]] = {}
        for axis, axis_labels in labels.items():
            raw_axis = raw.get(axis)
            if not isinstance(raw_axis, dict):
                raise RuntimeError(
                    f"SemanticClassificationEncoder missing hierarchical axis {axis!r} in {raw!r}"
                )
            axis_out: dict[str, float] = {}
            for label in axis_labels:
                if label not in raw_axis:
                    raise RuntimeError(
                        f"SemanticClassificationEncoder missing label {axis}.{label} in {raw_axis!r}"
                    )
                axis_out[str(label)] = float(raw_axis[label])
            out[str(axis)] = axis_out
        return out

    @staticmethod
    def _publish(topic: str, payload: dict[str, Any]) -> None:
        get_default_bus().publish(topic, payload)
