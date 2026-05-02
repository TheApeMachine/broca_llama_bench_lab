"""Affect organ: emotion detection and pragmatic intent classification.

Replaces regex-based sentiment patterns (_POSITIVE_SENTIMENT, _NEGATIVE_SENTIMENT)
with proper neural models that detect 28 fine-grained emotions and arbitrary
dynamic intent labels.

Brain analogy: Limbic system / insula — emotional processing, interoception,
and subjective feeling states that modulate decision-making.

Models:
- GoEmotions (SamLowe/roberta-base-go_emotions): 28-label multi-label emotion
  detection trained on 58k Reddit comments. 125M params, <5ms on CPU.
- comprehend_it (Knowledgator/comprehend_it-base): Zero-shot NLI-based
  classification with arbitrary dynamic labels. 184M params.

The organ exposes emotions as cognitive signals that the substrate uses for:
- Preference learning (DirichletPreference updates)
- Active inference (modulating exploration/exploitation via arousal)
- Intrinsic cues (confusion → clarifying question, frustration → strategy shift)
- Hawkes process excitation (surprise spikes temporal dynamics)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch

from ..system.event_bus import get_default_bus
from .base import BaseOrgan, OrganOutput

logger = logging.getLogger(__name__)


def _publish(topic: str, payload: dict) -> None:
    try:
        get_default_bus().publish(topic, payload)
    except Exception:
        pass

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
    """Complete affective state from the organ."""
    dominant_emotion: str = "neutral"
    dominant_score: float = 0.0
    emotions: list[EmotionScore] = field(default_factory=list)
    cognitive_states: dict[str, float] = field(default_factory=dict)
    valence: float = 0.0  # -1 (negative) to +1 (positive)
    arousal: float = 0.0  # 0 (calm) to 1 (excited)
    preference_signal: str = ""  # "positive_preference" or "negative_preference" or ""
    preference_strength: float = 0.0


class AffectOrgan(BaseOrgan):
    """Frozen emotion/affect detection model.

    Detects 28 fine-grained emotions per utterance and maps them to
    substrate-actionable signals for preference learning, active inference
    modulation, and intrinsic cue generation.

    Usage:
        organ = AffectOrgan()
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
            output_dim=28,  # 28 GoEmotions labels
            device=device,
        )
        self._threshold = threshold
        self._use_onnx = use_onnx
        self._pipeline: Any = None
        self._zero_shot: Any = None

    def _load_model(self) -> None:
        """Load the configured GoEmotions backend without silent fallback."""

        try:
            from transformers import AutoTokenizer, pipeline
        except ImportError as exc:
            raise ImportError(
                "AffectOrgan requires `transformers`. Install with: pip install transformers"
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
            self._session = ort.InferenceSession(
                model_path,
                providers=["CPUExecutionProvider"],
            )
            self._pipeline = self._run_onnx_text_classification
            logger.info("AffectOrgan: loaded ONNX quantized %s", onnx_model_id)
            return

        self._pipeline = pipeline(
            "text-classification",
            model=self._model_id,
            top_k=None,
            device=self.device if self.device.type != "cpu" else -1,
        )
        logger.info("AffectOrgan: loaded standard %s", self._model_id)

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
            raise RuntimeError(f"AffectOrgan ONNX session missing tokenizer inputs: {sorted(missing)}")
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
            if score >= thresh:
                signal = EMOTION_TO_SIGNAL.get(label, "")
                state.emotions.append(EmotionScore(label=label, score=score, signal=signal))

        state.emotions.sort(key=lambda e: e.score, reverse=True)

        if state.emotions:
            state.dominant_emotion = state.emotions[0].label
            state.dominant_score = state.emotions[0].score
        elif all_scores:
            best = max(all_scores, key=all_scores.get)
            state.dominant_emotion = best
            state.dominant_score = all_scores[best]

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
            s.score for s in state.emotions if s.signal == "positive_preference"
        )
        neg_pref = sum(
            s.score for s in state.emotions if s.signal == "negative_preference"
        )
        if pos_pref > neg_pref and pos_pref > 0.3:
            state.preference_signal = "positive_preference"
            state.preference_strength = pos_pref
        elif neg_pref > pos_pref and neg_pref > 0.3:
            state.preference_signal = "negative_preference"
            state.preference_strength = neg_pref

        latency = (time.time() - start) * 1000
        self._record_call(latency, method="detect")
        _publish(
            "organ.affect",
            {
                "text": text[:120],
                "dominant_emotion": state.dominant_emotion,
                "dominant_score": state.dominant_score,
                "valence": state.valence,
                "arousal": state.arousal,
                "preference_signal": state.preference_signal,
                "preference_strength": state.preference_strength,
                "cognitive_states": dict(state.cognitive_states),
                "top_emotions": [(e.label, e.score) for e in state.emotions[:5]],
                "latency_ms": latency,
            },
        )
        return state

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

    def process(self, text: str, **kwargs: Any) -> OrganOutput:
        """Unified organ interface."""
        state = self.detect(text)

        # Encode affect state as a feature vector for substrate integration
        import torch
        # 28-dim emotion vector + 3 summary dims (valence, arousal, dominant_score)
        features = torch.zeros(self._output_dim + 3, dtype=torch.float32)
        # Fill in emotion scores by their index in the model's label list
        for emo in state.emotions:
            # Use hash-based indexing since we don't have the exact label order
            idx = hash(emo.label) % self._output_dim
            features[idx] = max(features[idx].item(), emo.score)
        features[-3] = state.valence
        features[-2] = state.arousal
        features[-1] = state.dominant_score

        return OrganOutput(
            features=features,
            metadata={
                "dominant_emotion": state.dominant_emotion,
                "dominant_score": state.dominant_score,
                "cognitive_states": state.cognitive_states,
                "valence": state.valence,
                "arousal": state.arousal,
                "preference_signal": state.preference_signal,
                "preference_strength": state.preference_strength,
                "emotions": [(e.label, e.score) for e in state.emotions[:5]],
            },
            confidence=state.dominant_score,
            latency_ms=0.0,  # Already recorded internally
            organ_name=self._name,
        )
