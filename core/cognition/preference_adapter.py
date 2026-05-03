"""PreferenceAdapter — Dirichlet preference + Hawkes observation surface.

The substrate held three small methods that wrapped its preference and
temporal layers; they live here now.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .substrate import SubstrateController


logger = logging.getLogger(__name__)


class PreferenceAdapter:
    """Stateless wrapper around ``mind.spatial_preference`` / ``causal_preference`` / ``hawkes``."""

    def __init__(self, mind: "SubstrateController") -> None:
        self._mind = mind

    def sync_to_pomdp(self) -> None:
        """Push the Dirichlet means into the live POMDPs' C vectors."""

        mind = self._mind
        try:
            mind.pomdp.C = list(mind.spatial_preference.expected_C())
        except (AttributeError, TypeError):
            logger.exception("PreferenceAdapter.sync_to_pomdp: spatial sync failed")
        try:
            mind.causal_pomdp.C = list(mind.causal_preference.expected_C())
        except (AttributeError, TypeError):
            logger.exception("PreferenceAdapter.sync_to_pomdp: causal sync failed")

    def observe_user_feedback(
        self,
        *,
        faculty: str,
        observation_index: int,
        polarity: float,
        weight: float = 1.0,
        reason: str = "",
        conformal_set_size: int | None = None,
        epistemic_ambiguity_floor_strength: float = 0.18,
    ) -> None:
        mind = self._mind
        if faculty == "spatial":
            target = mind.spatial_preference
        elif faculty == "causal":
            target = mind.causal_preference
        else:
            raise ValueError(
                f"PreferenceAdapter.observe_user_feedback: unsupported faculty {faculty!r}; "
                "expected 'spatial' or 'causal'"
            )
        floor: float | None = None
        if polarity < 0 and conformal_set_size is not None and int(conformal_set_size) > 1:
            floor = float(target.prior_strength * epistemic_ambiguity_floor_strength)
        target.update(
            observation_index,
            polarity=polarity,
            weight=weight,
            reason=reason,
            epistemic_alpha_floor=floor,
        )
        self.sync_to_pomdp()
        try:
            mind.preference_persistence.save(faculty, target)
        except (sqlite3.Error, OSError):
            logger.exception(
                "PreferenceAdapter.observe_user_feedback: preference save failed"
            )

    def observe_event(self, channel: str, *, t: float | None = None) -> None:
        """Record an event on the Hawkes layer."""

        self._mind.hawkes.observe(channel, t=t)
