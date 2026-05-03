"""GLiClass-backed hierarchical zero-shot text classification.

A forward hook on the underlying DeBERTa encoder captures
``last_hidden_state`` for substrate working-memory publication. The hook is
read-only — it never mutates the encoder output.
"""

from __future__ import annotations

import time
from typing import Any, Sequence

import torch

from ..workspace import WorkspacePublisher
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
        from transformers import AutoTokenizer

        self._model = GLiClassModel.from_pretrained(self._model_id)
        self._processor = AutoTokenizer.from_pretrained(self._model_id)
        self._pipeline = ZeroShotClassificationPipeline(
            self._model,
            self._processor,
            classification_type="multi-label",
            device=str(self.device),
            progress_bar=False,
        )

        self._last_hidden: torch.Tensor | None = None
        self._install_hidden_state_hook()

    def _install_hidden_state_hook(self) -> None:
        encoder = self._locate_underlying_encoder()

        if not callable(getattr(encoder, "register_forward_hook", None)):
            raise RuntimeError(
                "SemanticClassificationEncoder requires an encoder module that supports "
                f"register_forward_hook; loaded model {self._model_id!r} does not."
            )

        def _hook(_module: Any, _input: Any, output: Any) -> None:
            tensor = self._extract_hidden_tensor(output)
            self._last_hidden = tensor.detach()

        encoder.register_forward_hook(_hook)

    def _locate_underlying_encoder(self) -> Any:
        for path in ("encoder", "model", "backbone", "transformer"):
            candidate = getattr(self._model, path, None)

            if candidate is not None and callable(getattr(candidate, "register_forward_hook", None)):
                return candidate

        raise RuntimeError(
            f"SemanticClassificationEncoder: cannot locate underlying transformer on {self._model_id!r}"
        )

    @staticmethod
    def _extract_hidden_tensor(output: Any) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output

        for attr in ("last_hidden_state", "hidden_states", "logits"):
            t = getattr(output, attr, None)

            if isinstance(t, torch.Tensor):
                return t

        if isinstance(output, (list, tuple)) and output and isinstance(output[0], torch.Tensor):
            return output[0]

        raise RuntimeError(
            f"SemanticClassificationEncoder hidden-state hook: unrecognised output type {type(output).__name__}"
        )

    @property
    def last_hidden(self) -> torch.Tensor:
        if self._model is None:
            raise RuntimeError("SemanticClassificationEncoder.last_hidden: model not loaded")

        if self._last_hidden is None:
            raise RuntimeError(
                "SemanticClassificationEncoder.last_hidden: no hidden state captured yet — call classify_axes first"
            )

        return self._last_hidden

    @property
    def has_captured_hidden(self) -> bool:
        return self._model is not None and self._last_hidden is not None

    @property
    def input_embedding_matrix(self) -> torch.Tensor:
        if self._model is None:
            raise RuntimeError("SemanticClassificationEncoder.input_embedding_matrix: model not loaded")

        encoder = self._locate_underlying_encoder()
        embeddings = getattr(encoder, "embeddings", None)
        word_embeddings = getattr(embeddings, "word_embeddings", None) if embeddings is not None else None

        if word_embeddings is None:
            raise RuntimeError(
                f"SemanticClassificationEncoder.input_embedding_matrix: cannot locate word_embeddings on {self._model_id!r}"
            )

        weight = getattr(word_embeddings, "weight", None)

        if not isinstance(weight, torch.Tensor):
            raise RuntimeError(
                "SemanticClassificationEncoder.input_embedding_matrix: word_embeddings.weight is not a tensor"
            )

        return weight.detach()

    @property
    def hidden_dim(self) -> int:
        return int(self.input_embedding_matrix.shape[-1])

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
        WorkspacePublisher.emit(
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
