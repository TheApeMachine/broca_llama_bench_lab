"""GlobalWorkspace — the substrate's per-turn cognitive frame buffer.

Holds the recent stream of :class:`CognitiveFrame` instances published by
non-language faculties (memory readouts, causal verdicts, active-inference
decisions). Maintains a sliding ``working`` window sized as
``max(8, ⌈4 √n_frames⌉)`` so working memory grows logarithmically in total
frames seen this session — bounded but not fixed.

When two recent frames carry complementary instruments (a ``memory_*``
readout and a ``causal_effect`` verdict), :meth:`post_frame` automatically
calls :meth:`CognitiveFrame.synthesize_bundle` and adds the synthesis to
the buffer so the cognitive router can pick it up.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict

from ..frame import CognitiveFrame
from .intrinsic_cue import IntrinsicCue


logger = logging.getLogger(__name__)


class GlobalWorkspace:
    """In-memory cognitive frame buffer plus intrinsic cue queue."""

    def __init__(self) -> None:
        self.frames: list[CognitiveFrame] = []
        self.intrinsic_cues: list[IntrinsicCue] = []
        self.working: list[CognitiveFrame] = []

    def _trim_working(self) -> None:
        cap = max(8, int(math.ceil(math.sqrt(max(1, len(self.frames)))) * 4))
        self.working = self.frames[-cap:]

    def post_frame(self, frame: CognitiveFrame) -> CognitiveFrame:
        self.frames.append(frame)
        self._trim_working()
        syn = CognitiveFrame.synthesize_bundle(self.working)
        if syn is not None:
            logger.debug(
                "GlobalWorkspace.post_frame: synthesized intent=%s from working tail",
                syn.intent,
            )
            self.frames.append(syn)
            self._trim_working()
        logger.debug(
            "GlobalWorkspace.post_frame: intent=%s journal_id=%s frames_total=%d",
            frame.intent,
            (frame.evidence or {}).get("journal_id"),
            len(self.frames),
        )
        return frame

    def raise_cue(self, cue: IntrinsicCue) -> None:
        self.intrinsic_cues.append(cue)

    @property
    def latest(self) -> CognitiveFrame | None:
        return self.frames[-1] if self.frames else None

    def snapshot(self) -> list[dict]:
        return [asdict(f) for f in self.frames]
