"""Per-peer reliability tracker for the swarm membrane.

Each peer's reliability is the posterior mean of a Beta(α, β) over the
correctness of its broadcasts:

    reliability(peer) = α(peer) / (α(peer) + β(peer))

with α(peer) = 1.0 and β(peer) = 1.0 as the maximum-entropy uniform prior.
After observing prediction errors ``e_i ∈ [0, 1]`` against this peer's
broadcasts, the conjugate posterior accumulates ``α += (1 - e_i)`` and
``β += e_i``. Peers with consistently low prediction error converge toward
reliability 1.0; peers whose broadcasts contradict local high-confidence
beliefs converge toward 0.0.

The Beta-conjugate update is the Bayesian analogue of the substrate's existing
Dirichlet preference machinery and avoids any tunable magic-number defaults:
``α=β=1`` is *the* uninformative prior, not a knob.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict

logger = logging.getLogger(__name__)

_PRIOR_ALPHA = 1.0
_PRIOR_BETA = 1.0


class PeerReliabilityRegistry:
    """Beta-conjugate reliability score per peer node id."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._alpha: dict[str, float] = defaultdict(lambda: _PRIOR_ALPHA)
        self._beta: dict[str, float] = defaultdict(lambda: _PRIOR_BETA)

    def reliability_for(self, peer_id: str) -> float:
        """Posterior mean reliability in (0, 1)."""

        with self._lock:
            a = self._alpha[peer_id]
            b = self._beta[peer_id]

        total = a + b

        if total <= 0.0:
            raise ValueError(
                f"PeerReliabilityRegistry.reliability_for: peer {peer_id!r} has zero concentration"
            )

        return float(a / total)

    def record_prediction_error(self, peer_id: str, error: float) -> None:
        """Update the Beta posterior with one observation.

        ``error`` is the prediction error of the peer's broadcast against the
        local node's verified state, on ``[0, 1]``. ``error=0`` is a perfect
        prediction (full credit); ``error=1`` is a complete contradiction.
        """

        e = float(error)

        if not 0.0 <= e <= 1.0:
            raise ValueError(
                f"PeerReliabilityRegistry.record_prediction_error: error {e} outside [0, 1]"
            )

        with self._lock:
            self._alpha[peer_id] = self._alpha[peer_id] + (1.0 - e)
            self._beta[peer_id] = self._beta[peer_id] + e
            a = self._alpha[peer_id]
            b = self._beta[peer_id]

        logger.debug(
            "PeerReliabilityRegistry.record: peer=%s error=%.3f alpha=%.3f beta=%.3f reliability=%.4f",
            peer_id,
            e,
            a,
            b,
            a / (a + b),
        )

    def snapshot(self) -> dict[str, float]:
        """Inspectable copy of current per-peer reliability."""

        with self._lock:
            peers = list(self._alpha.keys())
            scores = {p: self._alpha[p] / (self._alpha[p] + self._beta[p]) for p in peers}

        return scores
