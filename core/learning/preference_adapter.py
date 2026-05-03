"""PreferenceAdapter — Dirichlet preference + Hawkes observation."""

from __future__ import annotations

from typing import Any


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
