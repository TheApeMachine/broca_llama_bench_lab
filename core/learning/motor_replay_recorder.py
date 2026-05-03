"""Motor replay recording."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

from ..dmn import DMNConfig
from .motor_replay_item import MotorReplayItem

if TYPE_CHECKING:
    from ..substrate.controller import SubstrateController


class MotorReplayRecorder:
    """Records generated speech plans for later motor learning."""

    def __init__(self, substrate: "SubstrateController") -> None:
        self._substrate = substrate

    def record(
        self,
        messages: Sequence[dict[str, str]],
        *,
        generated_token_ids: Sequence[int],
        broca_features: torch.Tensor | None,
        substrate_confidence: float,
        substrate_inertia: float,
    ) -> None:
        if len(generated_token_ids) == 0:
            return

        substrate = self._substrate
        cap = DMNConfig().sleep_max_replay
        item = MotorReplayItem(
            messages=[dict(message) for message in messages],
            generated_token_ids=[int(token_id) for token_id in generated_token_ids],
            broca_features=broca_features,
            substrate_confidence=float(substrate_confidence),
            substrate_inertia=float(substrate_inertia),
        )

        with substrate.session.cognitive_state_lock:
            substrate.motor_replay.append(item.to_replay_dict())
            if len(substrate.motor_replay) > cap:
                substrate.motor_replay[:] = substrate.motor_replay[-cap:]
