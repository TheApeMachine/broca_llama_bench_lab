"""PlanSpeaker — plan-forced surface generation.

Retained for benchmark code that scores the substrate's ability to produce
specific tokens. Conversational use goes through
:class:`ChatOrchestrator`. The plan-forced path emits one token per planned
word, biased by :class:`LexicalPlanGraft`, and records the run as a motor-
training target so REM-time training can fit the residual graft to the
emitted tokens.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

from ..frame import CognitiveFrame
from ..generation import PlanForcedGenerator


if TYPE_CHECKING:
    from .substrate import SubstrateController


def _motor_replay_messages_plan_forced(
    frame: CognitiveFrame, plan_words: Sequence[str]
) -> list[dict[str, str]]:
    """One user turn synthesizing lexical-plan context for REM chat-template supervision."""

    chunks = (
        f"intent={frame.intent}",
        f"subject={frame.subject or ''}",
        f"answer={frame.answer or ''}",
        f"plan={' '.join(plan_words)}",
    )
    return [{"role": "user", "content": " | ".join(chunks)}]


class PlanSpeaker:
    """Plan-forced surface generation against the substrate's host."""

    def __init__(self, mind: "SubstrateController") -> None:
        self._mind = mind

    def speak(self, frame: CognitiveFrame) -> str:
        from .chat_orchestrator import ChatOrchestrator

        mind = self._mind
        plan_words = frame.speech_plan()
        broca_features = mind.broca_features_from_frame(frame)
        text_out, token_ids, inertia_tail = PlanForcedGenerator.generate(
            mind.host,
            mind.tokenizer,
            plan_words,
            broca_features=broca_features,
        )
        confidence = max(0.0, min(1.0, float(frame.confidence)))
        msgs = _motor_replay_messages_plan_forced(frame, plan_words)
        ChatOrchestrator(mind)._record_motor_replay(
            msgs,
            generated_token_ids=token_ids,
            broca_features=broca_features,
            substrate_confidence=confidence,
            substrate_inertia=inertia_tail,
        )
        return text_out
