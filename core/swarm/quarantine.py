"""PeerQuarantine — semipermeable membrane between the swarm and the local bus.

Every event that arrives from a peer must pass through this layer before it can
update the local epistemic state. Two enforcement paths:

1. **Native-tool source** (any payload carrying executable ``source`` plus
   ``sample_inputs`` and ``domain``): the local node treats it as fundamentally
   untrusted. The injected ``tool_validator`` is invoked in the local sandbox
   and must yield a singleton conformal prediction set; otherwise the payload
   is rejected outright. There is no "trusted peer" fast-path — the substrate
   only attaches code its own conformal calibration agrees with.

2. **All other event topics**: the payload is tagged with the peer's current
   posterior reliability (``_peer_reliability`` ∈ (0, 1)) and the peer id
   (``_peer_id``). Downstream consumers — Dirichlet preference updates, claim
   trust scoring, expected-free-energy calculations — must respect the tag so
   a hallucinating peer's contributions decay exponentially toward zero
   without ever crossing into negative mathematical bounds.

The quarantine never silently drops events. Rejection raises through the
caller, which is the swarm's receive loop; the loop logs and continues so a
single bad payload cannot kill the network thread.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Mapping

from .peer_reliability import PeerReliabilityRegistry

logger = logging.getLogger(__name__)


_NATIVE_TOOL_SOURCE_FIELDS = ("source", "sample_inputs", "domain")
_CONTROL_PLANE_TOPICS = frozenset({"swarm.peer.joined", "swarm.peer.left"})


class PeerRejected(Exception):
    """The peer payload failed local validation; the swarm must drop it."""


class PeerQuarantine:
    """Validate and tag swarm payloads before they reach the local event bus."""

    def __init__(
        self,
        *,
        reliability: PeerReliabilityRegistry,
        tool_validator: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        self._reliability = reliability
        self._tool_validator = tool_validator
        self._rejected = 0
        self._tagged = 0

    @property
    def reliability(self) -> PeerReliabilityRegistry:
        return self._reliability

    @property
    def stats(self) -> dict[str, int]:
        return {"rejected": int(self._rejected), "tagged": int(self._tagged)}

    def intercept(self, topic: str, payload: Any, peer_id: str) -> dict[str, Any]:
        """Return the enriched payload to publish, or raise :class:`PeerRejected`."""

        if topic in _CONTROL_PLANE_TOPICS:
            tagged = self._coerce_dict(payload)
            tagged["_peer_id"] = str(peer_id)
            self._tagged += 1
            return tagged

        tagged = self._coerce_dict(payload)

        if self._is_native_tool_source(topic, tagged):
            self._validate_native_tool(topic, tagged, peer_id)

        rel = self._reliability.reliability_for(peer_id)
        tagged["_peer_id"] = str(peer_id)
        tagged["_peer_reliability"] = float(rel)
        self._tagged += 1

        return tagged

    def record_prediction_error(self, peer_id: str, error: float) -> None:
        """Forward to the registry — convenience for downstream consumers."""

        self._reliability.record_prediction_error(peer_id, error)

    def _coerce_dict(self, payload: Any) -> dict[str, Any]:
        if isinstance(payload, dict):
            return dict(payload)

        return {"_raw": str(payload)}

    def _is_native_tool_source(self, topic: str, payload: Mapping[str, Any]) -> bool:
        if not topic.startswith("native_tool"):
            return False

        return all(field in payload for field in _NATIVE_TOOL_SOURCE_FIELDS)

    def _validate_native_tool(
        self, topic: str, payload: Mapping[str, Any], peer_id: str
    ) -> None:
        if self._tool_validator is None:
            self._rejected += 1
            raise PeerRejected(
                f"PeerQuarantine: peer {peer_id!r} broadcast native-tool source on topic "
                f"{topic!r} but no local tool_validator is wired; refusing SCM attachment"
            )

        try:
            self._tool_validator(payload)
        except Exception as exc:
            self._rejected += 1
            logger.warning(
                "PeerQuarantine: peer=%s topic=%s native-tool source rejected by local validator: %s",
                peer_id,
                topic,
                exc,
            )
            raise PeerRejected(str(exc)) from exc
