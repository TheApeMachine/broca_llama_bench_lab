"""IntrinsicCue — a high-urgency signal raised by the substrate to itself.

The DMN raises cues when it notices ambiguity, contradiction, or a question
the substrate should ask the user before proceeding. The cognitive router
inspects intrinsic cues at each turn and may bias the LLM toward a
clarifying utterance.

Cues are not events. Events are pub/sub messages that any subscriber may
read; cues are first-person interrupts the substrate writes to its own
blackboard.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class IntrinsicCue:
    urgency: float
    faculty: str
    evidence: dict = field(default_factory=dict)
    source: str | None = None
