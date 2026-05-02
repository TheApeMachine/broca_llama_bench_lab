"""The swarm node: UDP multicast wired into the EventBus.

Every local event goes to the network. Every network event comes to the
local bus. Peers discover each other via heartbeat. No filters, no options.
Errors are raised, not swallowed.
"""

from __future__ import annotations

import logging
import socket
import struct
import threading
import time
import uuid
from collections import defaultdict
from typing import Any

import msgpack

from .peer import PeerInfo, PEER_TIMEOUT_S

logger = logging.getLogger(__name__)

MCAST_GROUP = "239.255.77.1"
MCAST_PORT = 50077
MCAST_TTL = 1
PROTOCOL_VERSION = 1
HEARTBEAT_INTERVAL_S = 2.0


class Swarm:
    """UDP multicast swarm node wired directly into the EventBus.

    On construction, joins the multicast group and starts two daemon threads.
    Every EventBus event is broadcast. Every received event is published locally.
    """

    def __init__(self, event_bus: Any, *, capabilities: list[str]):
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

        self._tx = self._create_sender()
        self._rx = self._create_receiver()

        self._bus.subscribe("*", self._on_local_event)

        self._running.set()

        self._rx_thread = threading.Thread(target=self._receive_loop, daemon=True, name="swarm-rx")
        self._hb_thread = threading.Thread(target=self._heartbeat_loop, daemon=True, name="swarm-hb")
        self._rx_thread.start()
        self._hb_thread.start()

        logger.info("Swarm online: %s caps=%s", self.node_id, self.capabilities)

    @property
    def peers(self) -> list[PeerInfo]:
        with self._peers_lock:
            return [p for p in self._peers.values() if p.is_alive]

    @property
    def n_peers(self) -> int:
        return len(self.peers)

    def stop(self) -> None:
        """Send BYE and shut down. Raises on socket errors."""
        if not self._running.is_set():
            return

        self._running.clear()
        self._send({"type": "bye", "src": self.node_id, "seq": self._next_seq(), "v": PROTOCOL_VERSION})
        self._tx.close()

        mreq = struct.pack("4sL", socket.inet_aton(MCAST_GROUP), socket.INADDR_ANY)
        self._rx.setsockopt(socket.IPPROTO_IP, socket.IP_DROP_MEMBERSHIP, mreq)
        self._rx.close()

        self._rx_thread.join(timeout=3.0)
        self._hb_thread.join(timeout=3.0)
        logger.info("Swarm offline: %s", self.node_id)

    def _create_sender(self) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, MCAST_TTL)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)
        return sock

    def _create_receiver(self) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        sock.bind(("", MCAST_PORT))

        mreq = struct.pack("4sL", socket.inet_aton(MCAST_GROUP), socket.INADDR_ANY)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        sock.settimeout(1.0)
        return sock

    def _next_seq(self) -> int:
        with self._seq_lock:
            self._seq += 1
            return self._seq

    def _send(self, msg: dict) -> None:
        data = msgpack.packb(msg, use_bin_type=True)
        assert len(data) <= 65507, f"Message too large: {len(data)} bytes"
        self._tx.sendto(data, (MCAST_GROUP, MCAST_PORT))

    def _on_local_event(self, topic: str, payload: Any) -> None:
        """Forward every local event to the network."""
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
                if not self._running.is_set():
                    break
                raise

            msg = msgpack.unpackb(data, raw=False)

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
                self._peers[src] = PeerInfo(
                    node_id=src, address=addr,
                    capabilities=caps, last_seq=msg.get("seq", 0),
                )
                logger.info("Peer joined: %s @ %s caps=%s", src, addr, caps)
                self._bus.publish("swarm.peer.joined", {
                    "node_id": src, "capabilities": caps,
                    "address": addr, "_from_swarm": True,
                })
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
