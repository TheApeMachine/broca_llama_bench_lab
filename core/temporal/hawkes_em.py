"""EM fit for exponential-kernel multivariate Hawkes excitation (Veen & Schoenberg 2008).

Separated from :mod:`core.temporal.hawkes` so the online process + persistence stay lean.
"""

from __future__ import annotations

import logging
import math
from typing import Sequence

logger = logging.getLogger(__name__)


def _event_types(
    sorted_events: list[tuple[str, float]],
    channel_index: dict[str, int],
) -> list[int]:
    types: list[int] = []
    for c, _t in sorted_events:
        if c not in channel_index:
            raise ValueError(
                f"fit_excitation_em: unknown event channel {c!r}; "
                f"expected one of {sorted(channel_index.keys())!r}",
            )
        types.append(channel_index[c])
    return types


def _horizon_seconds(times: list[float]) -> float:
    """Width of the observation window for baseline rate (handles degenerate single timestamp)."""

    T = float(times[-1] - times[0])
    return 1e-12 if T <= 0.0 else T


def _initial_mu_alpha(
    *,
    n_events: int,
    K: int,
    T: float,
    smoothing: float,
) -> tuple[list[float], list[list[float]]]:
    mu = [max(smoothing, n_events / (K * T))] * K
    alpha = [
        [max(smoothing, 1.0 / (K * K)) for _ in range(K)]
        for _ in range(K)
    ]
    return mu, alpha


def _branching_weights(
    i: int,
    *,
    times: list[float],
    types: list[int],
    mu: list[float],
    alpha: list[list[float]],
    beta: float,
) -> tuple[int, list[float], list[int | None]]:
    """Weights for ``event i`` being background vs triggered by each prior event."""

    ki = types[i]
    ti = times[i]
    weights: list[float] = [float(mu[ki])]
    sources: list[int | None] = [None]
    for j in range(i):
        kj = types[j]
        tj = times[j]
        w = alpha[ki][kj] * math.exp(-beta * (ti - tj))
        weights.append(w)
        sources.append(j)
    return ki, weights, sources


def _apply_branching_posterior(
    *,
    event_index: int,
    ki: int,
    K: int,
    weights: list[float],
    sources: list[int | None],
    types: list[int],
    mu: list[float],
    baseline_counts: list[float],
    triggered_counts: list[list[float]],
) -> None:
    total = sum(weights)
    if total <= 0.0:
        logger.warning(
            "fit_excitation_em: non-positive branching total at event_index=%d ki=%d mu[ki]=%s total=%s "
            "weights=%s baseline_counts=%s triggered_counts[ki]=%s; attributing unit mass to baseline_counts[%d]",
            event_index,
            ki,
            mu[ki],
            total,
            weights,
            baseline_counts,
            [triggered_counts[ki][j] for j in range(K)],
            ki,
        )
        baseline_counts[ki] += 1.0
        return
    for w, src in zip(weights, sources, strict=True):
        p = w / total
        if src is None:
            baseline_counts[ki] += p
        else:
            triggered_counts[ki][types[src]] += p


def _e_step(
    *,
    n: int,
    K: int,
    times: list[float],
    types: list[int],
    mu: list[float],
    alpha: list[list[float]],
    beta: float,
) -> tuple[list[float], list[list[float]]]:
    triggered_counts = [[0.0] * K for _ in range(K)]
    baseline_counts = [0.0] * K
    for i in range(n):
        ki, weights, sources = _branching_weights(
            i, times=times, types=types, mu=mu, alpha=alpha, beta=beta
        )
        _apply_branching_posterior(
            event_index=i,
            ki=ki,
            K=K,
            weights=weights,
            sources=sources,
            types=types,
            mu=mu,
            baseline_counts=baseline_counts,
            triggered_counts=triggered_counts,
        )
    return baseline_counts, triggered_counts


def _m_step(
    *,
    n: int,
    K: int,
    times: list[float],
    types: list[int],
    baseline_counts: list[float],
    triggered_counts: list[list[float]],
    beta: float,
    smoothing: float,
    T: float,
) -> tuple[list[float], list[list[float]]]:
    new_mu = [max(smoothing, baseline_counts[k] / max(T, 1e-9)) for k in range(K)]
    new_alpha = [[smoothing] * K for _ in range(K)]
    for j in range(K):
        arrivals_j = [times[idx] for idx in range(n) if types[idx] == j]
        kernel_sum = sum(
            1.0 - math.exp(-beta * max(0.0, times[-1] - tt)) for tt in arrivals_j
        )
        denom = max(kernel_sum / max(beta, 1e-9), smoothing)
        for i in range(K):
            new_alpha[i][j] = max(smoothing, triggered_counts[i][j] / denom)
    return new_mu, new_alpha


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
    channel_index = {c: i for i, c in enumerate(chans)}
    types = _event_types(sorted_events, channel_index)
    times = [t for _, t in sorted_events]
    T = _horizon_seconds(times)

    mu, alpha = _initial_mu_alpha(n_events=n, K=K, T=T, smoothing=smoothing)

    for _ in range(max(1, int(iterations))):
        baseline_counts, triggered_counts = _e_step(
            n=n, K=K, times=times, types=types, mu=mu, alpha=alpha, beta=beta
        )
        mu, alpha = _m_step(
            n=n,
            K=K,
            times=times,
            types=types,
            baseline_counts=baseline_counts,
            triggered_counts=triggered_counts,
            beta=beta,
            smoothing=smoothing,
            T=T,
        )

    logger.debug(
        "fit_excitation_em: iterations=%d events=%d K=%d mu=%s",
        int(iterations),
        n,
        K,
        [round(m, 5) for m in mu],
    )
    return mu, alpha
