"""SWMResidualGraft — closed-form injection of a Substrate Working Memory slot.

Reads a named slot from the substrate's working memory at every layer step,
projects it through the closed-form :class:`SWMToInputProjection` (Johnson-
Lindenstrauss composed with the orthogonal projector onto the host's input
embedding column space), and adds the result into the residual stream at the
last sequence position with the substrate's standard SNR-derived magnitude.

This is the LatentMAS-style "broadcast the workspace into the agent's input
distribution" primitive, training-free: every parameter of the projection is
derived from the host's own ``W_in`` plus a deterministic JL seed.

The slot to inject is read from ``state['swm_inject_slot']`` so the recursion
controller can advance the slot per round without rewiring the graft.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ..grafting.alignment import SWMToInputProjection
from ..grafting.grafts import (
    BaseGraft,
    DEFAULT_GRAFT_TARGET_SNR,
    snr_magnitude,
    state_confidence,
    state_inertia,
    state_target_snr_scale,
)
from ..swm import SubstrateWorkingMemory


SWM_INJECT_SLOT_KEY: str = "swm_inject_slot"
ACTIVE_THOUGHT_SLOT: str = "active.thought"


class SWMResidualGraft(BaseGraft):
    """Project an SWM slot into the host's residual stream at the last position."""

    def __init__(
        self,
        *,
        swm: SubstrateWorkingMemory,
        projection: SWMToInputProjection,
        default_slot: str | None = None,
        target_snr: float = DEFAULT_GRAFT_TARGET_SNR,
    ) -> None:
        super().__init__()
        self._swm = swm
        self._projection = projection
        self._default_slot = str(default_slot) if default_slot else None
        self.target_snr = float(target_snr)
        self.mixer_priority = 0.45

    @property
    def default_slot(self) -> str | None:
        return self._default_slot

    @property
    def swm(self) -> SubstrateWorkingMemory:
        return self._swm

    @property
    def projection(self) -> SWMToInputProjection:
        return self._projection

    def forward(self, x: torch.Tensor, state: dict) -> torch.Tensor:
        if not self.enabled:
            return x

        explicit_slot = state.get(SWM_INJECT_SLOT_KEY)
        slot_name = explicit_slot or self._default_slot

        if slot_name is None:
            return x

        if not self._swm.has(str(slot_name)):
            # An explicit slot pointer with no backing data is a bug — the
            # caller asked for something that does not exist. The default
            # slot, on the other hand, may legitimately be empty during
            # bootstrap (e.g. a decode happens before any recursion has
            # written its thought); silently skip in that case.
            if explicit_slot is not None:
                raise KeyError(
                    f"SWMResidualGraft: explicit slot {explicit_slot!r} not present in SWM"
                )
            return x

        slot = self._swm.read(str(slot_name))

        if slot.vector.shape[-1] != self._projection.d_in:
            raise ValueError(
                f"SWMResidualGraft: slot {slot_name!r} dim {slot.vector.shape[-1]} "
                f"!= projection d_in {self._projection.d_in}"
            )

        if x.shape[-1] != self._projection.d_out:
            raise ValueError(
                f"SWMResidualGraft: residual last dim {x.shape[-1]} "
                f"!= projection d_out {self._projection.d_out}"
            )

        last_indices = state.get("last_indices")

        if last_indices is None:
            raise RuntimeError("SWMResidualGraft.forward: missing required state key 'last_indices'")

        last = last_indices.to(x.device)
        rows = torch.arange(x.shape[0], device=x.device)
        host_at_last = x[rows, last]

        projected = self._projection.apply(slot.vector.view(1, -1).to(device=x.device, dtype=x.dtype))
        direction = F.normalize(projected.view(-1), dim=0).expand(x.shape[0], -1)

        magnitude = snr_magnitude(
            host_at_last,
            target_snr=self.target_snr,
            confidence=state_confidence(state),
            inertia=state_inertia(state),
            substrate_scale=state_target_snr_scale(state),
        )

        out = x.clone()
        out[rows, last] = host_at_last + direction * magnitude

        return out
