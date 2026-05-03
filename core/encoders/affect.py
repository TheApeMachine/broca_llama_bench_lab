"""Affect encoder: emotion detection and pragmatic intent classification.

Replaces regex-based sentiment patterns (_POSITIVE_SENTIMENT, _NEGATIVE_SENTIMENT)
with proper neural models that detect 28 fine-grained emotions and arbitrary
dynamic intent labels.

Models:
- GoEmotions (SamLowe/roberta-base-go_emotions): 28-label multi-label emotion
  detection trained on 58k Reddit comments. 125M params, <5ms on CPU.
- comprehend_it (Knowledgator/comprehend_it-base): Zero-shot NLI-based
  classification with arbitrary dynamic labels. 184M params.

This encoder exposes emotions as cognitive signals that the substrate uses for:
- Preference learning (DirichletPreference updates)
- Active inference (modulating exploration/exploitation via arousal)
- Intrinsic cues (confusion → clarifying question, frustration → strategy shift)
- Hawkes process excitation (surprise spikes temporal dynamics)
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch

from ..workspace import WorkspacePublisher
from .base import BaseEncoder, EncoderOutput

logger = logging.getLogger(__name__)

# Primary emotion model
_GOEMOTION_MODEL = "SamLowe/roberta-base-go_emotions"
# Zero-shot classifier for custom labels
_COMPREHEND_MODEL = "Knowledgator/comprehend_it-base"

# GoEmotions labels that map to cognitive states useful for the substrate
COGNITIVE_STATE_LABELS = frozenset({
    "confusion", "curiosity", "nervousness", "excitement",
    "disapproval", "annoyance", "realization", "surprise",
})

# Mapping from GoEmotions labels to substrate signals
EMOTION_TO_SIGNAL: dict[str, str] = {
    # Positive preference signals
    "gratitude": "positive_preference",
    "approval": "positive_preference",
    "admiration": "positive_preference",
    "joy": "positive_preference",
    "love": "positive_preference",
    "optimism": "positive_preference",
    "pride": "positive_preference",
    "excitement": "positive_preference",
    # Negative preference signals
    "annoyance": "negative_preference",
    "anger": "negative_preference",
    "disapproval": "negative_preference",
    "disgust": "negative_preference",
    "disappointment": "negative_preference",
    # Epistemic signals (affect active inference)
    "confusion": "epistemic_uncertainty",
    "curiosity": "epistemic_drive",
    "realization": "epistemic_update",
    "surprise": "prediction_error",
    # Arousal signals (modulate exploration)
    "nervousness": "high_arousal_negative",
    "fear": "high_arousal_negative",
    "excitement": "high_arousal_positive",
    "amusement": "moderate_arousal_positive",
    # Disengagement
    "sadness": "low_arousal_negative",
    "remorse": "low_arousal_negative",
}


@dataclass
class EmotionScore:
    """A single emotion detection with score."""
    label: str
    score: float
    signal: str = ""  # Mapped substrate signal


@dataclass
class AffectState:
    """Complete affective state from the encoder."""
    dominant_emotion: str = "neutral"
    dominant_score: float = 0.0
    confidences: list[EmotionScore] = field(default_factory=list)
    emotions: list[EmotionScore] = field(default_factory=list)
    cognitive_states: dict[str, float] = field(default_factory=dict)
    valence: float = 0.0  # -1 (negative) to +1 (positive)
    arousal: float = 0.0  # 0 (calm) to 1 (excited)
    entropy: float = 0.0
    certainty: float = 0.0
    preference_signal: str = ""  # "positive_preference" or "negative_preference" or ""
    preference_strength: float = 0.0

    def distribution(self) -> dict[str, float]:
        """Return the full GoEmotions confidence vector keyed by label."""

        return {item.label: float(item.score) for item in self.confidences}


class AffectEncoder(BaseEncoder):
    """Frozen emotion/affect detection model.

    Detects 28 fine-grained emotions per utterance and maps them to
    substrate-actionable signals for preference learning, active inference
    modulation, and intrinsic cue generation.

    Usage:
        organ = AffectEncoder()
        organ.load()
        state = organ.detect("I can't believe they did that, it's so frustrating!")
        # state.dominant_emotion = "annoyance"
        # state.cognitive_states = {"annoyance": 0.71}
        # state.preference_signal = "negative_preference"
    """

    def __init__(
        self,
        *,
        model_id: str | None = None,
        device: str | None = None,
        threshold: float = 0.15,
        use_onnx: bool = True,
    ):
        super().__init__(
            name="affect",
            model_id=model_id or _GOEMOTION_MODEL,
            output_dim=31,  # 28 GoEmotions labels + valence/arousal/certainty
            device=device,
        )
        self._threshold = threshold
        self._use_onnx = use_onnx
        self._pipeline: Any = None
        self._zero_shot: Any = None
        self._label_order: list[str] = []

    def _load_model(self) -> None:
        """Load the configured GoEmotions backend without silent fallback."""

        try:
            from transformers import AutoTokenizer, pipeline
        except ImportError as exc:
            raise ImportError(
                "AffectEncoder requires `transformers`. Install with: pip install transformers"
            ) from exc

        if self._use_onnx:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            from transformers import AutoConfig

            onnx_model_id = self._model_id + "-onnx" if not self._model_id.endswith("-onnx") else self._model_id
            model_path = hf_hub_download(
                repo_id=onnx_model_id,
                filename="onnx/model_quantized.onnx",
            )
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
            self._config = AutoConfig.from_pretrained(self._model_id)
            self._label_order = [
                str(self._config.id2label[i])
                for i in sorted(getattr(self._config, "id2label", {}))
            ]
            self._session = ort.InferenceSession(
                model_path,
                providers=["CPUExecutionProvider"],
            )
            self._pipeline = self._run_onnx_text_classification
            logger.info("AffectEncoder: loaded ONNX quantized %s", onnx_model_id)
            return

        self._pipeline = pipeline(
            "text-classification",
            model=self._model_id,
            top_k=None,
            device=self.device if self.device.type != "cpu" else -1,
        )
        config = getattr(getattr(self._pipeline, "model", None), "config", None)
        self._label_order = [
            str(config.id2label[i])
            for i in sorted(getattr(config, "id2label", {}))
        ] if config is not None else []
        logger.info("AffectEncoder: loaded standard %s", self._model_id)

    def _run_onnx_text_classification(self, text: str) -> list[dict[str, float | str]]:
        encoded = self._tokenizer(
            text,
            truncation=True,
            max_length=512,
            return_tensors="np",
        )
        input_names = {inp.name for inp in self._session.get_inputs()}
        inputs = {name: encoded[name] for name in input_names if name in encoded}
        missing = input_names.difference(inputs)
        if missing:
            raise RuntimeError(f"AffectEncoder ONNX session missing tokenizer inputs: {sorted(missing)}")
        logits = self._session.run(None, inputs)[0][0].astype(np.float32)
        scores = 1.0 / (1.0 + np.exp(-logits))
        id2label = getattr(self._config, "id2label", {})
        return [
            {
                "label": str(id2label.get(i, f"LABEL_{i}")),
                "score": float(score),
            }
            for i, score in enumerate(scores)
        ]

    def _load_zero_shot(self) -> None:
        """Lazy-load zero-shot classifier for custom labels."""
        if self._zero_shot is not None:
            return
        from transformers import pipeline

        self._zero_shot = pipeline(
            "zero-shot-classification",
            model=_COMPREHEND_MODEL,
            device=self.device if self.device.type != "cpu" else -1,
        )

    def detect(self, text: str, *, threshold: float | None = None) -> AffectState:
        """Detect emotional state from text.

        Returns a rich AffectState with dominant emotion, cognitive states,
        valence/arousal estimates, and substrate preference signals.
        """
        self._ensure_loaded()
        start = time.time()
        thresh = threshold if threshold is not None else self._threshold

        state = AffectState()

        if not text or not text.strip():
            self._record_call((time.time() - start) * 1000)
            return state

        raw_scores = self._pipeline(text[:512])
        if isinstance(raw_scores, list) and raw_scores and isinstance(raw_scores[0], list):
            raw_scores = raw_scores[0]

        all_scores: dict[str, float] = {}
        for item in raw_scores:
            label = item["label"]
            score = float(item["score"])
            all_scores[label] = score
            signal = EMOTION_TO_SIGNAL.get(label, "")
            state.confidences.append(EmotionScore(label=label, score=score, signal=signal))

        state.confidences.sort(key=lambda e: e.score, reverse=True)
        state.emotions = [item for item in state.confidences if item.score >= thresh]
        state.emotions.sort(key=lambda e: e.score, reverse=True)
        state.entropy = self._distribution_entropy(state.confidences)
        state.certainty = self._distribution_certainty(state.confidences, entropy=state.entropy)

        if state.confidences:
            state.dominant_emotion = state.confidences[0].label
            state.dominant_score = state.confidences[0].score

        state.cognitive_states = {
            label: score for label, score in all_scores.items()
            if label in COGNITIVE_STATE_LABELS and score >= thresh
        }

        positive = sum(
            all_scores.get(e, 0.0) for e in
            ["joy", "love", "gratitude", "approval", "admiration", "optimism", "excitement", "amusement", "pride", "relief", "caring"]
        )
        negative = sum(
            all_scores.get(e, 0.0) for e in
            ["anger", "annoyance", "disgust", "disappointment", "disapproval", "fear", "grief", "nervousness", "remorse", "sadness", "embarrassment"]
        )
        total = positive + negative
        state.valence = (positive - negative) / max(total, 1e-6) if total > 0.01 else 0.0

        high_arousal = sum(
            all_scores.get(e, 0.0) for e in
            ["anger", "excitement", "fear", "surprise", "nervousness", "disgust"]
        )
        low_arousal = sum(
            all_scores.get(e, 0.0) for e in
            ["sadness", "relief", "neutral", "caring"]
        )
        state.arousal = min(1.0, high_arousal / max(high_arousal + low_arousal, 1e-6))

        pos_pref = sum(
            s.score for s in state.confidences if s.signal == "positive_preference"
        )
        neg_pref = sum(
            s.score for s in state.confidences if s.signal == "negative_preference"
        )
        pref_total = pos_pref + neg_pref
        if pos_pref > neg_pref and pos_pref >= thresh:
            state.preference_signal = "positive_preference"
            state.preference_strength = pos_pref / max(pref_total, 1e-12)
        elif neg_pref > pos_pref and neg_pref >= thresh:
            state.preference_signal = "negative_preference"
            state.preference_strength = neg_pref / max(pref_total, 1e-12)

        latency = (time.time() - start) * 1000
        self._record_call(latency, method="detect")
        WorkspacePublisher.emit(
            "encoder.affect",
            {
                "text": text[:120],
                "dominant_emotion": state.dominant_emotion,
                "dominant_score": state.dominant_score,
                "entropy": state.entropy,
                "certainty": state.certainty,
                "valence": state.valence,
                "arousal": state.arousal,
                "preference_signal": state.preference_signal,
                "preference_strength": state.preference_strength,
                "cognitive_states": dict(state.cognitive_states),
                "top_emotions": [(e.label, e.score) for e in state.emotions[:5]],
                "confidences": [(e.label, e.score) for e in state.confidences],
                "latency_ms": latency,
            },
        )
        return state

    @staticmethod
    def _distribution_entropy(scores: Sequence[EmotionScore]) -> float:
        total = sum(max(0.0, float(item.score)) for item in scores)
        if total <= 0.0:
            return 0.0
        entropy = 0.0
        for item in scores:
            p = max(0.0, float(item.score)) / total
            if p > 0.0:
                entropy -= p * math.log(p)
        return float(entropy)

    @staticmethod
    def _distribution_certainty(scores: Sequence[EmotionScore], *, entropy: float) -> float:
        n = len(scores)
        if n <= 1:
            return 1.0 if n == 1 else 0.0
        max_entropy = math.log(n)
        if max_entropy <= 0.0:
            return 0.0
        return max(0.0, min(1.0, 1.0 - float(entropy) / max_entropy))

    def classify_custom(
        self,
        text: str,
        *,
        labels: Sequence[str],
        multi_label: bool = True,
        threshold: float = 0.3,
    ) -> list[tuple[str, float]]:
        """Zero-shot classification with arbitrary custom labels.

        Uses comprehend_it-base (NLI-based) for labels not in GoEmotions taxonomy.
        Useful for detecting: sarcasm, frustration, enthusiasm, skepticism, etc.
        """
        self._ensure_loaded()
        self._load_zero_shot()
        start = time.time()

        raw = self._zero_shot(
            text[:512],
            candidate_labels=list(labels),
            multi_label=multi_label,
        )
        results = list(zip(raw["labels"], raw["scores"]))
        results = [(l, s) for l, s in results if s >= threshold]
        results.sort(key=lambda x: x[1], reverse=True)

        self._record_call((time.time() - start) * 1000)
        return results

    def process(self, text: str, **kwargs: Any) -> EncoderOutput:
        """Unified encoder process entrypoint."""
        state = self.detect(text)

        # Encode affect state as a feature vector for substrate integration
        import torch

        distribution = state.distribution()
        labels = self._label_order or sorted(distribution)
        features = torch.tensor(
            [distribution.get(label, 0.0) for label in labels]
            + [state.valence, state.arousal, state.certainty],
            dtype=torch.float32,
        )
        features[-3] = state.valence
        features[-2] = state.arousal
        features[-1] = state.certainty

        return EncoderOutput(
            features=features,
            metadata={
                "dominant_emotion": state.dominant_emotion,
                "dominant_score": state.dominant_score,
                "confidences": [(e.label, e.score) for e in state.confidences],
                "cognitive_states": state.cognitive_states,
                "valence": state.valence,
                "arousal": state.arousal,
                "entropy": state.entropy,
                "certainty": state.certainty,
                "preference_signal": state.preference_signal,
                "preference_strength": state.preference_strength,
                "emotions": [(e.label, e.score) for e in state.emotions[:5]],
            },
            confidence=state.certainty,
            latency_ms=0.0,  # Already recorded internally
            encoder_name=self._name,
        )
