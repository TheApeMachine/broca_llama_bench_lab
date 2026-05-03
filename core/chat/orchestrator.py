"""Chat orchestrator."""

from __future__ import annotations

from typing import Callable, Sequence

from ..frame import CognitiveFrame


class ChatOrchestrator:
    """Conversation boundary for one chat reply."""

    def __init__(self, mind: "SubstrateController") -> None:
        self._mind = mind

    def run(
        self,
        messages: Sequence[dict[str, str]],
        *,
        max_new_tokens: int = 256,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9,
        on_token: Callable[[str], None] | None = None,
    ) -> tuple[CognitiveFrame, str]:
        mind = self._mind
        msgs = [dict(m) for m in messages]

        if not msgs or msgs[-1].get("role") != "user":
            raise ValueError("ChatOrchestrator.run expects messages ending with a user turn")

        user_text = str(msgs[-1].get("content", "")).strip()
        frame = mind.comprehend(user_text)

        text = mind.chat_turn.reply(
            frame,
            msgs,
            max_new_tokens=int(max_new_tokens),
            do_sample=bool(do_sample),
            temperature=float(temperature),
            top_p=float(top_p),
            on_token=on_token,
        )

        return frame, text
