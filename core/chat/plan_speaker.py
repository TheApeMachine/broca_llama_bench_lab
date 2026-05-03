"""PlanSpeaker — plan-forced surface generation.

Retained for benchmark code that scores the substrate's ability to produce
specific tokens. Conversational use goes through :mod:`core.chat.orchestrator`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from ..frame import CognitiveFrame
from ..generation import PlanForcedGenerator
from ..numeric import Probability


if TYPE_CHECKING:
    from ..substrate.controller import SubstrateController


class PlanSpeaker:
    """Plan-forced surface generation against the substrate's host."""

    def __init__(self, mind: "SubstrateController") -> None:
        self._mind = mind
        self.probability = Probability()

    @staticmethod
    def motor_replay_messages_plan_forced(
        frame: CognitiveFrame, plan_words: Sequence[str]
    ) -> list[dict[str, str]]:
        chunks = (
            f"intent={frame.intent}",
            f"subject={frame.subject or ''}",
            f"answer={frame.answer or ''}",
            f"plan={' '.join(plan_words)}",
        )
        return [{"role": "user", "content": " | ".join(chunks)}]

    def speak(self, frame: CognitiveFrame) -> str:
        mind = self._mind
        plan_words = frame.speech_plan()
        broca_features = mind.broca_features_from_frame(frame)
        text_out, token_ids, inertia_tail = PlanForcedGenerator.generate(
            mind.host,
            mind.tokenizer,
            plan_words,
            broca_features=broca_features,
        )
        confidence = self.probability.unit_interval(frame.confidence)
        msgs = self.motor_replay_messages_plan_forced(frame, plan_words)
        mind.motor_replay_recorder.record(
            msgs,
            generated_token_ids=token_ids,
            broca_features=broca_features,
            substrate_confidence=confidence,
            substrate_inertia=inertia_tail,
        )
        return text_out
