"""Shared validation for Hawkes persistence and refit payloads.

Keeps :class:`MultivariateHawkesProcess` / :class:`PersistentHawkes` readable by
centralizing shape and JSON-snapshot checks.
"""

from __future__ import annotations

from typing import Sequence


def require_square_mu_alpha(
    *,
    n: int,
    mu: Sequence[object],
    alpha: Sequence[Sequence[object]],
    where: str,
) -> None:
    """Ensure ``len(mu) == n`` and ``alpha`` is ``n × n``."""

    if len(mu) != n:
        raise ValueError(f"{where}: len(mu)={len(mu)} != len(channels)={n}")
    if len(alpha) != n:
        raise ValueError(f"{where}: alpha has {len(alpha)} rows; expected {n}")
    for ri, row in enumerate(alpha):
        if len(row) != n:
            raise ValueError(
                f"{where}: alpha row {ri} has length {len(row)}; expected {n}",
            )


def normalized_state_entries(
    states_raw: object,
    n: int,
    *,
    where: str,
) -> list[tuple[float, list[float]]]:
    """Validate persisted per-channel state list; return ``(last_t, cache)`` tuples."""

    if not isinstance(states_raw, list):
        raise ValueError(f"{where}: states must be a JSON array (list), got {type(states_raw).__name__}")

    if len(states_raw) != n:
        raise ValueError(f"{where}: len(states)={len(states_raw)} != n_channels={n}")

    out: list[tuple[float, list[float]]] = []
    for si, s in enumerate(states_raw):
        if not isinstance(s, dict):
            raise ValueError(f"{where}: states[{si}] must be an object/dict")
        if "last_t" not in s or "cache" not in s:
            raise ValueError(
                f"{where}: states[{si}] missing required keys 'last_t' and/or 'cache'",
            )
        if not isinstance(s["last_t"], (int, float)):
            raise ValueError(f"{where}: states[{si}]['last_t'] must be numeric")
        if not isinstance(s["cache"], list):
            raise ValueError(f"{where}: states[{si}]['cache'] must be a list")
        try:
            cache_list = [float(x) for x in s["cache"]]
        except (TypeError, ValueError) as e:
            raise ValueError(f"{where}: states[{si}]['cache'] must be a numeric list") from e
        out.append((float(s["last_t"]), cache_list))
    return out
