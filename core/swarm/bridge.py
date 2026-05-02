"""Bridge between the local EventBus and the LAN multicast swarm.

The bridge selectively forwards local events to the network and injects
received network events into the local bus. This keeps the swarm
communication non-invasive — only explicitly bridged topics cross the wire.

Usage:
    from core.swarm import SwarmNode
    from core.swarm.bridge import SwarmBridge
    from core.system.event_bus import get_default_bus

    bus = get_default_bus()
    node = SwarmNode(node_id="mosaic-alpha", capabilities=["llm", "affect"])
    bridge = SwarmBridge(node=node, bus=bus)
    bridge.start()

    # Now any local publish to "swarm.*" topics gets broadcast to peers,
    # and any received network event gets published to the local bus.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Sequence

logger = logging.getLogger(__name__)


class SwarmBridge:
    """Bidirectional bridge between local EventBus and SwarmNode.

    Outbound: subscribes to local EventBus topics matching the forward_patterns
    and broadcasts them to the swarm.

    Inbound: receives network events from the SwarmNode and publishes them
    to the local EventBus with a "net." prefix to distinguish from local events.

    Configuration:
        forward_patterns: Topics to forward to the network.
            Default: ["swarm.*", "organ.*.broadcast", "memory.update"]
        receive_prefix: Prefix added to received topics on local bus.
            Default: "net." (so "organ.memory.update" → "net.organ.memory.update")
    """

    def __init__(
        self,
        *,
        node: Any,  # SwarmNode
        bus: Any,   # EventBus
        forward_patterns: Sequence[str] | None = None,
        receive_prefix: str = "net.",
        max_forward_rate: float = 100.0,  # max messages/sec to network
    ):
        from . import SwarmNode
        assert isinstance(node, SwarmNode)

        self._node = node
        self._bus = bus
        self._forward_patterns = list(forward_patterns or [
            "swarm.*",
            "organ.*.broadcast",
            "memory.update",
            "frame.comprehend",
        ])
        self._receive_prefix = receive_prefix
        self._max_forward_rate = max_forward_rate
        self._last_forward_time = 0.0
        self._forwarded_count = 0
        self._received_count = 0
        self._subscriptions: list[Any] = []
        self._started = False

    def start(self) -> None:
        """Start the bridge: subscribe to local bus and wire up network reception."""
        if self._started:
            return

        # Wire inbound: network → local bus
        self._node._on_receive = self._handle_network_message
        self._node._on_peer_joined = self._handle_peer_joined
        self._node._on_peer_left = self._handle_peer_left

        # Wire outbound: local bus → network
        for pattern in self._forward_patterns:
            self._bus.subscribe(pattern, self._handle_local_event)

        # Start the swarm node if not already running
        if not self._node.is_running:
            self._node.start()

        self._started = True
        logger.info("SwarmBridge started: forwarding %s", self._forward_patterns)

    def stop(self) -> None:
        """Stop the bridge."""
        self._node.stop()
        self._started = False

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "started": self._started,
            "forwarded": self._forwarded_count,
            "received": self._received_count,
            "forward_patterns": self._forward_patterns,
            "node": self._node.stats,
        }

    def _handle_local_event(self, topic: str, payload: dict) -> None:
        """Forward a local event to the network (outbound)."""
        import time
        now = time.monotonic()
        # Rate limiting
        if self._max_forward_rate > 0:
            min_interval = 1.0 / self._max_forward_rate
            if now - self._last_forward_time < min_interval:
                return
        self._last_forward_time = now

        # Don't forward events that came FROM the network (avoid loops)
        if topic.startswith(self._receive_prefix):
            return

        self._node.broadcast(topic, payload)
        self._forwarded_count += 1

    def _handle_network_message(self, msg: dict) -> None:
        """Inject a received network event into the local bus (inbound)."""
        msg_type = msg.get("type")

        if msg_type == "event":
            topic = msg.get("topic", "")
            payload = msg.get("payload", {})
            # Add provenance
            payload["_from_node"] = msg.get("src", "unknown")
            payload["_network_ts"] = msg.get("ts", 0)
            # Publish with prefix to avoid forwarding loops
            local_topic = f"{self._receive_prefix}{topic}"
            self._bus.publish(local_topic, payload)
            self._received_count += 1

        elif msg_type == "state":
            state = msg.get("state", {})
            src = msg.get("src", "unknown")
            self._bus.publish(f"{self._receive_prefix}state.snapshot", {
                "from_node": src,
                "state": state,
            })
            self._received_count += 1

    def _handle_peer_joined(self, peer: Any) -> None:
        """Notify local bus of new peer discovery."""
        self._bus.publish("swarm.peer.joined", {
            "node_id": peer.node_id,
            "capabilities": peer.capabilities,
            "address": peer.address,
        })

    def _handle_peer_left(self, peer: Any) -> None:
        """Notify local bus of peer departure."""
        self._bus.publish("swarm.peer.left", {
            "node_id": peer.node_id,
        })
