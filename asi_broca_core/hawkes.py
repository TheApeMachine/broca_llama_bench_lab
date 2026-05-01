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

import json
import logging
import math
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)


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

        now = time.time()
        self.channels = list(channels)
        self.mu = [float(m) for m in mu]
        self.alpha = [[float(x) for x in row] for row in alpha]
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
            name: self._intensity_no_decay(self.channels.index(name))
            for name in self.channels
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
        compensator = sum(self.mu) * (T - T0)
        # Per-channel α_{ij} contributions to compensator.
        for j, name in enumerate(self.channels):
            arrivals = [t for c, t in sorted_events if c == name]
            for s in arrivals:
                tail = max(0.0, T - s)
                kernel_int = (1.0 - math.exp(-self.beta * tail)) / max(self.beta, 1e-9)
                for i in range(len(self.channels)):
                    compensator += float(self.alpha[i][j]) * kernel_int
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
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.namespace = namespace
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path)
        con.execute("PRAGMA journal_mode=WAL")
        return con

    def _init_schema(self) -> None:
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS hawkes_state (
                    namespace TEXT PRIMARY KEY,
                    beta REAL NOT NULL,
                    baseline REAL NOT NULL,
                    channels_json TEXT NOT NULL,
                    mu_json TEXT NOT NULL,
                    alpha_json TEXT NOT NULL,
                    states_json TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )

    def save(self, process: MultivariateHawkesProcess) -> None:
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO hawkes_state(namespace, beta, baseline, channels_json, mu_json, alpha_json, states_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(namespace) DO UPDATE SET
                    beta=excluded.beta,
                    baseline=excluded.baseline,
                    channels_json=excluded.channels_json,
                    mu_json=excluded.mu_json,
                    alpha_json=excluded.alpha_json,
                    states_json=excluded.states_json,
                    updated_at=excluded.updated_at
                """,
                (
                    self.namespace,
                    float(process.beta),
                    float(process.baseline),
                    json.dumps(process.channels),
                    json.dumps(process.mu),
                    json.dumps(process.alpha),
                    json.dumps(
                        [
                            {"last_t": s.last_t, "cache": s.cache}
                            for s in process._states
                        ]
                    ),
                    time.time(),
                ),
            )

    def load(self) -> MultivariateHawkesProcess | None:
        with self._connect() as con:
            row = con.execute(
                "SELECT beta, baseline, channels_json, mu_json, alpha_json, states_json FROM hawkes_state WHERE namespace=?",
                (self.namespace,),
            ).fetchone()
        if row is None:
            return None
        proc = MultivariateHawkesProcess(beta=float(row[0]), baseline=float(row[1]))
        proc.channels = list(json.loads(row[2]))
        proc.mu = list(json.loads(row[3]))
        proc.alpha = [list(r) for r in json.loads(row[4])]
        proc._states = [
            HawkesState(last_t=float(s["last_t"]), cache=list(s["cache"]))
            for s in json.loads(row[5])
        ]
        return proc


def fit_excitation_em(
    events: Sequence[tuple[str, float]],
    channels: Sequence[str],
    *,
    beta: float,
    iterations: int = 25,
    smoothing: float = 1e-3,
) -> tuple[list[float], list[list[float]]]:
    """Maximum-likelihood EM for exponential-kernel Hawkes (Veen & Schoenberg 2008).

    Returns ``(mu, alpha)``. Branching probabilities ``p_{ij}`` (the probability
    that event i was triggered by event j) are computed in the E-step; the
    M-step then re-estimates ``mu`` from un-triggered events and ``alpha`` from
    triggered ones. Convergence is monotone in NLL.
    """

    sorted_events = sorted(events, key=lambda e: e[1])
    chans = list(channels)
    if not sorted_events or not chans:
        return [smoothing] * len(chans), [[smoothing] * len(chans) for _ in chans]
    n = len(sorted_events)
    K = len(chans)
    idx_of = {c: i for i, c in enumerate(chans)}
    types: list[int] = []
    for c, _evt_t in sorted_events:
        if c not in idx_of:
            raise ValueError(
                f"fit_excitation_em: unknown event channel {c!r}; expected one of {sorted(idx_of.keys())!r}",
            )
        types.append(idx_of[c])
    times = [t for _, t in sorted_events]
    # Horizon width for baseline intensity; avoid the legacy ``+ 1.0`` bias.
    # Degenerate timeline (single timestamp): use a tiny positive span so divisions stay finite.
    T = float(times[-1] - times[0])
    if T <= 0.0:
        T = 1e-12

    mu = [max(smoothing, n / (K * T))] * K
    alpha = [[max(smoothing, 1.0 / (K * K)) for _ in range(K)] for _ in range(K)]

    for _ in range(max(1, int(iterations))):
        triggered_counts = [[0.0] * K for _ in range(K)]
        baseline_counts = [0.0] * K
        for i in range(n):
            ti = times[i]
            ki = types[i]
            weights = [mu[ki]]
            sources: list[int | None] = [None]
            for j in range(i):
                kj = types[j]
                tj = times[j]
                w = alpha[ki][kj] * math.exp(-beta * (ti - tj))
                if w > 0:
                    weights.append(w)
                    sources.append(j)
            total = sum(weights)
            if total <= 0.0:
                logger.warning(
                    "fit_excitation_em: non-positive branching total at event_index=%d ki=%d mu[ki]=%s total=%s "
                    "weights=%s baseline_counts=%s triggered_counts[ki]=%s; attributing unit mass to baseline_counts[%d]",
                    i,
                    ki,
                    mu[ki],
                    total,
                    weights,
                    baseline_counts,
                    [triggered_counts[ki][j] for j in range(K)],
                    ki,
                )
                baseline_counts[ki] += 1.0
                continue
            for w, src in zip(weights, sources):
                p = w / total
                if src is None:
                    baseline_counts[ki] += p
                else:
                    triggered_counts[ki][types[src]] += p
        # M-step.
        new_mu = [max(smoothing, baseline_counts[k] / max(T, 1e-9)) for k in range(K)]
        new_alpha = [[smoothing] * K for _ in range(K)]
        # Per-source normalizer: 1 - exp(-β (T - t_j)) summed across arrivals of type j.
        for j in range(K):
            arrivals_j = [times[idx] for idx in range(n) if types[idx] == j]
            kernel_sum = sum(
                1.0 - math.exp(-beta * max(0.0, times[-1] - tt)) for tt in arrivals_j
            )
            denom = max(kernel_sum / max(beta, 1e-9), smoothing)
            for i in range(K):
                new_alpha[i][j] = max(smoothing, triggered_counts[i][j] / denom)
        mu = new_mu
        alpha = new_alpha
    logger.debug(
        "fit_excitation_em: iterations=%d events=%d K=%d mu=%s",
        int(iterations),
        n,
        K,
        [round(m, 5) for m in mu],
    )
    return mu, alpha
