"""Substrate-owned chat reply turn."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Callable, Sequence

from ..frame import CognitiveFrame
from ..generation import ChatDecoder
from ..grafts import ChatGraftPlan, FrameGraftProjection
from ..learning import MotorReplayRecorder
from .chat_completion import ChatCompletion

if TYPE_CHECKING:
    from .controller import SubstrateController


class SubstrateChatTurn:
    """Composes substrate planning, decode, replay, and lifecycle recording."""

    def __init__(
        self,
        *,
        substrate: "SubstrateController",
        grafts: FrameGraftProjection,
        decoder: ChatDecoder,
        replay: MotorReplayRecorder,
    ) -> None:
        self._substrate = substrate
        self._grafts = grafts
        self._decoder = decoder
        self._replay = replay

    def reply(
        self,
        frame: CognitiveFrame,
        messages: Sequence[dict[str, str]],
        *,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        on_token: Callable[[str], None] | None,
    ) -> str:
        plan = self._grafts.chat_plan(frame, requested_temperature=temperature)
        self._start(plan)
        text, generated_token_ids, substrate_inertia = self._decoder.stream(
            messages,
            plan=plan,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            on_token=on_token,
        )
        self._replay.record(
            messages,
            generated_token_ids=generated_token_ids,
            broca_features=plan.broca_features,
            substrate_confidence=plan.confidence,
            substrate_inertia=substrate_inertia,
        )
        self._complete(plan, text=text)
        return text

    def _start(self, plan: ChatGraftPlan) -> None:
        substrate = self._substrate
        substrate.session.last_chat_meta = plan.start_metadata(timestamp=time.time())
        substrate.event_bus.publish("chat.start", dict(substrate.session.last_chat_meta))

    def _complete(self, plan: ChatGraftPlan, *, text: str) -> None:
        substrate = self._substrate
        if not substrate.session.last_chat_meta:
            raise RuntimeError("SubstrateChatTurn._complete requires start metadata")
        if substrate.session.last_affect is None:
            raise RuntimeError("SubstrateChatTurn._complete requires recorded user affect")
        if substrate.session.last_user_affect_trace_id is None:
            raise RuntimeError(
                "SubstrateChatTurn._complete requires recorded user affect trace id"
            )

        assistant_affect = substrate.affect_encoder.detect(text)

        affect_alignment = substrate.affect_trace.alignment(
            substrate.session.last_affect,
            assistant_affect,
        )

        assistant_affect_trace_id = substrate.affect_trace.record(
            role="assistant",
            text=text,
            affect=assistant_affect,
            response_to_id=substrate.session.last_user_affect_trace_id,
            alignment=affect_alignment,
        )

        completion = ChatCompletion(
            plan=plan,
            text=text,
            assistant_affect=assistant_affect,
            affect_alignment=dict(affect_alignment),
            assistant_affect_trace_id=int(assistant_affect_trace_id),
            user_affect_trace_id=int(substrate.session.last_user_affect_trace_id),
        )
        substrate.session.last_chat_meta = {
            **substrate.session.last_chat_meta,
            **completion.meta_patch(),
        }

        substrate.event_bus.publish("chat.complete", completion.event_payload())
