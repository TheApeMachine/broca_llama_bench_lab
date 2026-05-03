"""PreferenceAdapter — Dirichlet preference + Hawkes observation."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class PreferenceAdapter:
    """Dirichlet preference vectors + Hawkes temporal channel."""

    def __init__(
        self,
        *,
        spatial_preference: Any,
        causal_preference: Any,
        hawkes: Any,
        pomdp: Any,
        causal_pomdp: Any,
        preference_persistence: Any,
    ) -> None:
        self._spatial = spatial_preference
        self._causal = causal_preference
        self._hawkes = hawkes
        self._pomdp = pomdp
        self._causal_pomdp = causal_pomdp
        self._pref_persistence = preference_persistence

    def sync_to_pomdp(self) -> None:
        self._pomdp.C = list(self._spatial.expected_C())
        self._causal_pomdp.C = list(self._causal.expected_C())

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
        if faculty == "spatial":
            target = self._spatial
        elif faculty == "causal":
            target = self._causal
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
        self._pref_persistence.save(faculty, target)

    def observe_event(self, channel: str, *, t: float | None = None) -> None:
        self._hawkes.observe(channel, t=t)

    def observe_affect(self, affect: Any) -> None:
        """Translate one ``AffectState`` into Dirichlet feedback on both faculties.

        The substrate's existing affect encoder produces ``preference_signal``
        ("positive_preference" / "negative_preference" / "") and
        ``preference_strength`` ∈ [0, 1] on every utterance. That is the natural
        feedback channel for the Dirichlet prior over preferences ``C``. With no
        explicit user observation index to attribute reward to, we reinforce the
        agent's *current* favorite observation per faculty: positive affect
        strengthens the existing preference, negative affect flattens it. This
        mirrors operant conditioning of a confidence vector by valence.
        """

        signal = str(getattr(affect, "preference_signal", "") or "")

        if signal not in ("positive_preference", "negative_preference"):
            return

        polarity = 1.0 if signal == "positive_preference" else -1.0
        strength = float(getattr(affect, "preference_strength", 0.0) or 0.0)

        if strength <= 0.0:
            return

        for faculty, target in (("spatial", self._spatial), ("causal", self._causal)):
            mean = target.expected_C()

            if not mean:
                continue

            obs_index = max(range(len(mean)), key=lambda i: mean[i])

            self.observe_user_feedback(
                faculty=faculty,
                observation_index=obs_index,
                polarity=polarity,
                weight=strength,
                reason=f"affect:{signal}",
            )

        logger.debug(
            "PreferenceAdapter.observe_affect: signal=%s strength=%.3f spatial_mean=%s causal_mean=%s",
            signal,
            strength,
            [round(x, 3) for x in self._spatial.expected_C()],
            [round(x, 3) for x in self._causal.expected_C()],
        )
