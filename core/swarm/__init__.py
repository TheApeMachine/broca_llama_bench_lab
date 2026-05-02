"""UDP multicast swarm — every node sees everything, no filters.

Every Mosaic instance on the LAN is part of the swarm. The swarm is not
optional. All EventBus events flow to the network. All network events
flow to the local EventBus. There is no selective bridging, no filtering,
no "forward patterns." Nodes communicate freely and the substrate decides
what to do with what it hears.

The only deduplication is: don't process your own messages, and don't
process the same sequence number twice from the same sender.
"""

from __future__ import annotations

import logging
import socket
import struct
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import msgpack

logger = logging.getLogger(__name__)

MCAST_GROUP = "239.255.77.1"
MCAST_PORT = 50077
MCAST_TTL = 1
PROTOCOL_VERSION = 1
HEARTBEAT_INTERVAL_S = 2.0
PEER_TIMEOUT_S = 8.0


@dataclass
class PeerInfo:
    node_id: str
    address: tuple[str, int]
    capabilities: list[str] = field(default_factory=list)
    last_seen: float = field(default_factory=time.monotonic)
    last_seq: int = 0

    @property
    def age_seconds(self) -> float:
        return time.monotonic() - self.last_seen


class Swarm:
    """UDP multicast swarm node wired directly into the EventBus.

    On construction, the swarm joins the multicast group and starts
    listening. Every event published to the local EventBus is broadcast
    to the LAN. Every event received from the LAN is published to the
    local EventBus. Peers are discovered automatically via heartbeat.

    There are no configuration options. There is one swarm per process.
    """

    def __init__(self, event_bus: Any, *, capabilities: list[str]):
        """Wire up and start immediately.

        Args:
            event_bus: The local EventBus instance. Must have publish(topic, payload)
                and subscribe(pattern, callback) methods.
            capabilities: List of organ/subsystem names this node is running.
        """
        self._bus = event_bus
        self.node_id = f"mosaic-{uuid.uuid4().hex[:8]}"
        self.capabilities = capabilities
        self._peers: dict[str, PeerInfo] = {}
        self._peers_lock = threading.Lock()
        self._seen_seqs: dict[str, int] = defaultdict(lambda: -1)
        self._seq = 0
        self._seq_lock = threading.Lock()
        self._running = threading.Event()
        self._total_sent = 0
        self._total_received = 0

        # Sockets
        self._tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self._tx.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, MCAST_TTL)
        self._tx.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)

        self._rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self._rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self._rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except (AttributeError, OSError):
            pass
        self._rx.bind(("", MCAST_PORT))
        mreq = struct.pack("4sL", socket.inet_aton(MCAST_GROUP), socket.INADDR_ANY)
        self._rx.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        self._rx.settimeout(1.0)

        # Subscribe to ALL local events and forward them to the network
        self._bus.subscribe("*", self._on_local_event)

        # Start threads
        self._running.set()
        self._rx_thread = threading.Thread(target=self._receive_loop, daemon=True, name="swarm-rx")
        self._hb_thread = threading.Thread(target=self._heartbeat_loop, daemon=True, name="swarm-hb")
        self._rx_thread.start()
        self._hb_thread.start()

        logger.info("Swarm online: %s caps=%s", self.node_id, self.capabilities)

    @property
    def peers(self) -> list[PeerInfo]:
        with self._peers_lock:
            return [p for p in self._peers.values() if p.age_seconds <= PEER_TIMEOUT_S]

    @property
    def n_peers(self) -> int:
        return len(self.peers)

    def stop(self) -> None:
        """Send BYE and shut down."""
        if not self._running.is_set():
            return
        self._running.clear()
        self._send({"type": "bye", "src": self.node_id, "seq": self._next_seq(), "v": PROTOCOL_VERSION})
        try:
            self._tx.close()
        except OSError:
            pass
        try:
            mreq = struct.pack("4sL", socket.inet_aton(MCAST_GROUP), socket.INADDR_ANY)
            self._rx.setsockopt(socket.IPPROTO_IP, socket.IP_DROP_MEMBERSHIP, mreq)
            self._rx.close()
        except OSError:
            pass
        self._rx_thread.join(timeout=3.0)
        self._hb_thread.join(timeout=3.0)
        logger.info("Swarm offline: %s", self.node_id)

    def _next_seq(self) -> int:
        with self._seq_lock:
            self._seq += 1
            return self._seq

    def _send(self, msg: dict) -> None:
        data = msgpack.packb(msg, use_bin_type=True)
        if len(data) > 65507:
            return
        try:
            self._tx.sendto(data, (MCAST_GROUP, MCAST_PORT))
        except OSError:
            pass

    def _on_local_event(self, topic: str, payload: Any) -> None:
        """Every local EventBus event gets broadcast to the LAN."""
        # Don't re-broadcast events that came from the network
        if isinstance(payload, dict) and payload.get("_from_swarm"):
            return
        self._send({
            "v": PROTOCOL_VERSION,
            "type": "event",
            "src": self.node_id,
            "seq": self._next_seq(),
            "ts": time.time(),
            "topic": topic,
            "payload": payload if isinstance(payload, dict) else {"_raw": str(payload)},
        })
        self._total_sent += 1

    def _receive_loop(self) -> None:
        while self._running.is_set():
            try:
                data, addr = self._rx.recvfrom(65535)
            except socket.timeout:
                self._expire_peers()
                continue
            except OSError:
                break

            try:
                msg = msgpack.unpackb(data, raw=False)
            except Exception:
                continue

            if msg.get("v") != PROTOCOL_VERSION:
                continue

            src = msg.get("src", "")
            seq = msg.get("seq", 0)

            if src == self.node_id:
                continue

            if self._seen_seqs[src] >= seq:
                continue
            self._seen_seqs[src] = seq

            self._total_received += 1
            msg_type = msg.get("type")

            if msg_type == "announce":
                self._handle_announce(src, addr, msg)
            elif msg_type == "event":
                self._handle_event(msg)
            elif msg_type == "bye":
                self._handle_bye(src)

    def _handle_announce(self, src: str, addr: tuple, msg: dict) -> None:
        caps = msg.get("caps", [])
        with self._peers_lock:
            is_new = src not in self._peers
            if is_new:
                self._peers[src] = PeerInfo(node_id=src, address=addr, capabilities=caps, last_seq=msg.get("seq", 0))
                logger.info("Peer joined: %s @ %s caps=%s", src, addr, caps)
                self._bus.publish("swarm.peer.joined", {"node_id": src, "capabilities": caps, "address": addr, "_from_swarm": True})
            else:
                peer = self._peers[src]
                peer.last_seen = time.monotonic()
                peer.last_seq = msg.get("seq", 0)
                peer.capabilities = caps
                peer.address = addr

    def _handle_event(self, msg: dict) -> None:
        topic = msg.get("topic", "")
        payload = msg.get("payload", {})
        if isinstance(payload, dict):
            payload["_from_swarm"] = True
            payload["_from_node"] = msg.get("src", "")
        self._bus.publish(topic, payload)

    def _handle_bye(self, src: str) -> None:
        with self._peers_lock:
            removed = self._peers.pop(src, None)
        if removed:
            logger.info("Peer left: %s", src)
            self._bus.publish("swarm.peer.left", {"node_id": src, "_from_swarm": True})

    def _expire_peers(self) -> None:
        now = time.monotonic()
        with self._peers_lock:
            dead = [nid for nid, p in self._peers.items() if now - p.last_seen > PEER_TIMEOUT_S]
            for nid in dead:
                del self._peers[nid]
                logger.info("Peer expired: %s", nid)
                self._bus.publish("swarm.peer.left", {"node_id": nid, "_from_swarm": True})

    def _heartbeat_loop(self) -> None:
        while self._running.is_set():
            self._send({
                "v": PROTOCOL_VERSION,
                "type": "announce",
                "src": self.node_id,
                "seq": self._next_seq(),
                "ts": time.time(),
                "caps": self.capabilities,
            })
            self._running.wait(timeout=HEARTBEAT_INTERVAL_S)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.stop()
