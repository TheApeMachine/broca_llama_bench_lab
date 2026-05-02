from __future__ import annotations

import types
from pathlib import Path
from typing import Any

import pytest
import torch

import core.cognition.substrate as substrate_mod
from core.cli import build_substrate_controller
from core.cognition.observation import CognitiveObservation
from core.cognition.substrate import SubstrateController

from conftest import make_stub_llm_pair


class FakeHost:
    cfg = types.SimpleNamespace(d_model=8)

    def __init__(self) -> None:
        self.grafts: list[tuple[str, Any]] = []
        self.llm, self._stub_tokenizer = make_stub_llm_pair()

    def add_graft(self, slot: str, graft: Any) -> None:
        self.grafts.append((slot, graft))


class FakeTokenizer:
    def __init__(self, stub_inner: Any) -> None:
        self.inner = stub_inner


class StubMultimodalPipeline:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    @property
    def registered_encoders(self) -> list[str]:
        return ["visual_cortex", "auditory_cortex", "association_cortex"]

    @property
    def loaded_encoders(self) -> list[str]:
        return []

    def stats(self) -> dict[str, Any]:
        return {
            "n_registered": len(self.registered_encoders),
            "n_loaded": 0,
            "encoders": {},
        }

    def perceive_image(self, image: Any, *, source: str = "image") -> CognitiveObservation:
        _ = image
        self.calls.append(("image", source))
        return CognitiveObservation(
            modality="image",
            source=source,
            features=torch.tensor([1.0, 2.0, 3.0]),
            confidence=0.87,
            answer="visual scene",
            evidence={"streams": ["visual_cortex", "association_cortex"]},
        )

    def perceive_video(self, frames: Any, *, source: str = "video") -> CognitiveObservation:
        _ = frames
        self.calls.append(("video", source))
        return CognitiveObservation(
            modality="video",
            source=source,
            features=torch.tensor([3.0, 2.0, 1.0]),
            confidence=0.81,
            answer="temporal visual scene",
            evidence={"streams": ["dorsal_stream", "association_cortex"]},
        )

    def perceive_audio(
        self,
        audio: Any,
        *,
        sampling_rate: int = 16000,
        source: str = "audio",
        language: str | None = None,
    ) -> CognitiveObservation:
        _ = audio, language
        self.calls.append(("audio", source))
        return CognitiveObservation(
            modality="audio",
            source=source,
            features=torch.tensor([0.25, 0.5, 1.0]),
            confidence=0.93,
            answer="ada is in rome .",
            evidence={
                "transcription": "ada is in rome .",
                "sampling_rate": int(sampling_rate),
                "streams": ["auditory_cortex", "association_cortex_text"],
            },
        )


@pytest.fixture
def fake_host_loader(monkeypatch: pytest.MonkeyPatch):
    def _make() -> FakeHost:
        host = FakeHost()
        tokenizer = FakeTokenizer(host._stub_tokenizer)
        monkeypatch.setattr(
            substrate_mod,
            "load_llama_broca_host",
            lambda *args, **kwargs: (host, tokenizer),
        )
        return host

    return _make


def _build_mind(tmp_path: Path) -> SubstrateController:
    return build_substrate_controller(
        seed=0,
        db_path=tmp_path / "multimodal.sqlite",
        namespace="multimodal",
        device="cpu",
        hf_token=False,
    )


def test_controller_registers_full_multimodal_pipeline(tmp_path: Path, fake_host_loader) -> None:
    fake_host_loader()
    mind = _build_mind(tmp_path)

    assert {
        "visual_cortex",
        "ventral_stream",
        "dorsal_stream",
        "spatial_cortex",
        "auditory_cortex",
        "association_cortex",
    }.issubset(set(mind.multimodal_perception.registered_encoders))
    assert mind.snapshot()["encoders"]["n_registered"] >= 6


def test_perceive_image_commits_workspace_journal_and_hopfield(
    tmp_path: Path,
    fake_host_loader,
) -> None:
    fake_host_loader()
    mind = _build_mind(tmp_path)
    pipeline = StubMultimodalPipeline()
    mind.multimodal_perception = pipeline  # type: ignore[assignment]

    frame = mind.perceive_image(torch.ones(3, 4, 4), source="unit-image")

    assert pipeline.calls == [("image", "unit-image")]
    assert frame.intent == "perception_image"
    assert frame.subject == "image"
    assert frame.answer == "visual scene"
    assert frame.confidence == pytest.approx(0.87)
    assert frame.evidence["source"] == "unit-image"
    assert frame.evidence["feature_dim"] == 3
    assert frame.evidence["allows_storage"] is False
    assert mind.workspace.latest is frame
    assert mind.journal.count() == 1
    assert len(mind.hopfield_memory) == 1

    row = mind.journal.fetch(int(frame.evidence["journal_id"]))
    assert row is not None
    assert row["utterance"] == "[image:unit-image] visual scene"


def test_perceive_audio_routes_transcription_into_language_memory(
    tmp_path: Path,
    fake_host_loader,
) -> None:
    fake_host_loader()
    mind = _build_mind(tmp_path)
    mind.multimodal_perception = StubMultimodalPipeline()  # type: ignore[assignment]

    frame = mind.perceive_audio(torch.ones(16000), source="unit-audio")

    assert frame.intent == "perception_audio"
    assert frame.answer == "ada is in rome ."
    assert mind.journal.count() == 2
    assert len(mind.hopfield_memory) == 1

    reflections = mind.process_deferred_relation_ingest()
    assert reflections[0]["status"] == "memory_write"
    assert len(mind.hopfield_memory) == 2

    rec = mind.memory.get("ada", "is_in")
    assert rec is not None
    assert rec[0] == "rome"
