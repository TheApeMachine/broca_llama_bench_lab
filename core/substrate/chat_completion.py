"""Chat completion model."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from ..affect.evidence import AffectEvidence
from ..grafts import ChatGraftPlan


class ChatCompletion(BaseModel):
    """Assistant completion metadata for a chat turn."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    plan: ChatGraftPlan
    text: str
    assistant_affect: Any
    affect_alignment: dict[str, Any]
    assistant_affect_trace_id: int
    user_affect_trace_id: int

    def meta_patch(self) -> dict[str, Any]:
        return {
            "assistant_affect": AffectEvidence.as_dict(self.assistant_affect),
            "affect_alignment": dict(self.affect_alignment),
            "assistant_affect_trace_id": int(self.assistant_affect_trace_id),
            "user_affect_trace_id": int(self.user_affect_trace_id),
        }

    def event_payload(self) -> dict[str, Any]:
        return {
            "intent": self.plan.frame.intent,
            "confidence": float(self.plan.confidence),
            "affect_alignment": float(self.affect_alignment["alignment"]),
            "reply_chars": len(self.text),
            "reply_preview": self.text[:200],
        }
