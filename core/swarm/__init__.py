"""UDP multicast swarm — peers cross a conformal-quarantine membrane.

The membrane validates native-tool source against the local sandbox+conformal
calibration and tags every other payload with the broadcasting peer's
posterior reliability so downstream Bayesian machinery can decay a
hallucinating peer toward zero influence without crossing zero.
"""

from .peer import PeerInfo
from .peer_reliability import PeerReliabilityRegistry
from .quarantine import PeerQuarantine, PeerRejected
from .swarm import Swarm

__all__ = [
    "PeerInfo",
    "PeerQuarantine",
    "PeerRejected",
    "PeerReliabilityRegistry",
    "Swarm",
]
