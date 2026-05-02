"""Multivariate Hawkes processes for the substrate's temporal layer.

A Hawkes process models *self- and mutually-exciting* event streams: when
event ``i`` arrives at time ``s``, the conditional intensity of every other
event channel jumps by ``α_{ij}`` and decays exponentially at rate ``β``:

    λ_i(t) = μ_i + Σ_j Σ_{s < t, type=j} α_{ij} · exp(-β (t - s))

The intensity itself is the substrate's "working memory" — channels that have
recently fired stay hot for a few seconds, then cool off. This is what gives
the architecture an intuition for the *flow of conversation* without storing
explicit timing in every fact.

This implementation:

* Computes intensities incrementally. Each channel keeps a running cache of
  ``Σ exp(-β (t - s))``; observing a new event multiplies the cache by
  ``exp(-β Δt)`` and adds 1.0. That's O(1) per arrival, regardless of history
  length.
* Supports *negative log-likelihood* on a recorded event sequence so the
  excitation matrix ``α`` can be learned (or at least sanity-checked) from
  observed substrate behavior.
* Persists its excitation matrix and last-event-times so the substrate's
  short-term memory survives process restarts.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .hawkes_em import fit_excitation_em
from .hawkes_validate import normalized_state_entries, require_square_mu_alpha
from .repository import HawkesRepository

logger = logging.getLogger(__name__)

__all__ = [
    "HawkesState",
    "MultivariateHawkesProcess",
    "PersistentHawkes",
    "fit_excitation_em",
]


@dataclass
class HawkesState:
    """Per-channel cache of the exponential excitation sum."""

    last_t: float = 0.0
    cache: list[float] = field(default_factory=list)


class MultivariateHawkesProcess:
    """Exponential-kernel multivariate Hawkes process with online intensity.

    ``mu``      — baseline intensities (per channel).
    ``alpha``   — excitation matrix; ``alpha[i][j]`` is the jump in λ_i caused
                  by an arrival on channel j.
    ``beta``    — exponential decay rate; large β = short memory.

    Channel ordering is decided by the order in which channel names are
    registered. New channels can be added on the fly and existing matrices
    grow accordingly.
    """

    def __init__(self, *, beta: float = 0.5, baseline: float = 0.05):
        self.beta = float(beta)
        self.baseline = float(baseline)
        self.channels: list[str] = []
        self.mu: list[float] = []
        self.alpha: list[list[float]] = []
        self._states: list[HawkesState] = []

    def refit(
        self,
        channels: Sequence[str],
        mu: Sequence[float],
        alpha: Sequence[Sequence[float]],
    ) -> None:
        """Replace channel order and intensity parameters; reset per-channel caches."""

        chan_list = list(channels)
        n = len(chan_list)
        require_square_mu_alpha(n=n, mu=mu, alpha=alpha, where="MultivariateHawkesProcess.refit")
        alpha_rows = [[float(x) for x in row] for row in alpha]

        now = time.time()
        self.channels = chan_list
        self.mu = [float(m) for m in mu]
        self.alpha = alpha_rows
        self._states = [HawkesState(last_t=now) for _ in self.channels]

    # ------------------------------------------------------------------ schema

    def _ensure_channel(
        self, name: str, *, default_alpha: float = 0.0, default_self_excite: float = 0.6
    ) -> int:
        if name in self.channels:
            return self.channels.index(name)
        idx = len(self.channels)
        self.channels.append(name)
        self.mu.append(self.baseline)
        for row in self.alpha:
            row.append(float(default_alpha))
        new_row = [float(default_alpha)] * (idx + 1)
        new_row[idx] = float(default_self_excite)
        self.alpha.append(new_row)
        self._states.append(HawkesState(last_t=time.time()))
        logger.debug(
            "MultivariateHawkesProcess._ensure_channel: name=%r idx=%d total=%d",
            name,
            idx,
            len(self.channels),
        )
        return idx

    def couple(self, source: str, target: str, *, weight: float) -> None:
        """Set ``alpha[target][source] = weight`` so source events excite target."""

        s = self._ensure_channel(source)
        t = self._ensure_channel(target)
        self.alpha[t][s] = float(weight)

    # ------------------------------------------------------------------ runtime

    def _decay(self, idx: int, now: float) -> float:
        st = self._states[idx]
        dt = max(0.0, float(now) - float(st.last_t))
        factor = math.exp(-self.beta * dt) if dt > 0.0 else 1.0
        new_cache = sum(c * factor for c in st.cache)
        st.cache = [new_cache] if new_cache > 1e-12 else []
        st.last_t = float(now)
        return new_cache

    def _decay_all(self, when: float) -> None:
        for j in range(len(self.channels)):
            self._decay(j, float(when))

    def _intensity_no_decay(self, idx: int) -> float:
        excite = 0.0
        for j in range(len(self.channels)):
            cache = sum(self._states[j].cache)
            excite += float(self.alpha[idx][j]) * cache
        return float(self.mu[idx] + excite)

    def observe(self, channel: str, *, t: float | None = None) -> None:
        """Record an arrival on ``channel`` at time ``t`` (default: now)."""

        idx = self._ensure_channel(channel)
        when = float(t) if t is not None else time.time()
        last_t = self._states[idx].last_t
        if when < last_t:
            logger.warning(
                "MultivariateHawkesProcess.observe: out-of-order event for channel=%r when=%.6f last_t=%.6f; "
                "events out of chronological order may produce incorrect intensities",
                channel,
                when,
                last_t,
            )
        self._decay_all(when)
        self._states[idx].cache.append(1.0)
        logger.debug(
            "MultivariateHawkesProcess.observe: channel=%r at=%.4f cache=%.4f",
            channel,
            when,
            sum(self._states[idx].cache),
        )

    def intensity(self, channel: str, *, t: float | None = None) -> float:
        """Conditional intensity λ_i(t) given the recorded history."""

        idx = self._ensure_channel(channel)
        when = float(t) if t is not None else time.time()
        self._decay_all(when)
        return self._intensity_no_decay(idx)

    def intensity_vector(self, *, t: float | None = None) -> dict[str, float]:
        """All channel intensities at time ``t``."""

        when = float(t) if t is not None else time.time()
        self._decay_all(when)
        return {
            name: self._intensity_no_decay(i) for i, name in enumerate(self.channels)
        }

    # ------------------------------------------------------------------ learning

    def negative_log_likelihood(
        self, events: Sequence[tuple[str, float]], *, horizon: float | None = None
    ) -> float:
        """NLL of an exponential-kernel multivariate Hawkes process.

        Used by the DMN to spot-check whether ``alpha`` is consistent with the
        recorded sequence; growing NLL between ticks signals the substrate
        should re-fit the excitation matrix.
        """

        if not events:
            return 0.0
        sorted_events = sorted(events, key=lambda e: e[1])
        # Reset state for evaluation.
        local = MultivariateHawkesProcess(beta=self.beta, baseline=self.baseline)
        local.channels = list(self.channels)
        local.mu = list(self.mu)
        local.alpha = [row[:] for row in self.alpha]
        local._states = [HawkesState(last_t=sorted_events[0][1]) for _ in self.channels]

        log_intensity_sum = 0.0
        for ch, t in sorted_events:
            lam = local.intensity(ch, t=t)
            if lam <= 0.0:
                lam = 1e-12
            log_intensity_sum += math.log(lam)
            local.observe(ch, t=t)
        T = float(horizon if horizon is not None else sorted_events[-1][1])
        T0 = float(sorted_events[0][1])
        # Compensator must use the same channel set / parameters as ``local``
        # (which may auto-register channels during intensity/observe).
        compensator = sum(local.mu) * (T - T0)
        # Per-channel α_{ij} contributions to compensator.
        for j, name in enumerate(local.channels):
            arrivals = [t for c, t in sorted_events if c == name]
            for s in arrivals:
                tail = max(0.0, T - s)
                kernel_int = (1.0 - math.exp(-local.beta * tail)) / max(
                    local.beta, 1e-9,
                )
                for i in range(len(local.channels)):
                    compensator += float(local.alpha[i][j]) * kernel_int
        nll = compensator - log_intensity_sum
        logger.debug(
            "MultivariateHawkesProcess.NLL: events=%d horizon=%.3f nll=%.4f",
            len(sorted_events),
            T,
            nll,
        )
        return float(nll)


class PersistentHawkes:
    """SQLite-backed persistence wrapper for ``MultivariateHawkesProcess``."""

    def __init__(self, path: str | Path, *, namespace: str = "main"):
        self._repo = HawkesRepository(path, namespace=namespace)
        self._repo.init_schema()

    @property
    def path(self) -> Path:
        return self._repo.path

    @property
    def namespace(self) -> str:
        return self._repo.namespace

    def save(self, process: MultivariateHawkesProcess) -> None:
        self._repo.upsert_process(
            beta=process.beta,
            baseline=process.baseline,
            channels=list(process.channels),
            mu=list(process.mu),
            alpha=[list(row) for row in process.alpha],
            state_dicts=[
                {"last_t": s.last_t, "cache": s.cache}
                for s in process._states
            ],
        )

    def load(self) -> MultivariateHawkesProcess | None:
        snap = self._repo.fetch()
        if snap is None:
            return None
        n = len(snap.channels)
        require_square_mu_alpha(
            n=n,
            mu=snap.mu,
            alpha=snap.alpha,
            where="PersistentHawkes.load",
        )
        states = [
            HawkesState(last_t=lt, cache=c)
            for lt, c in normalized_state_entries(
                snap.states_raw, n, where="PersistentHawkes.load"
            )
        ]
        proc = MultivariateHawkesProcess(beta=snap.beta, baseline=snap.baseline)
        proc.channels = snap.channels
        proc.mu = [float(x) for x in snap.mu]
        proc.alpha = [[float(x) for x in row] for row in snap.alpha]
        proc._states = states
        return proc
