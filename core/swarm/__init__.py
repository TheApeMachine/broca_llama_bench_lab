"""UDP multicast swarm communication for Mosaic substrate instances.

Enables multiple Mosaic nodes on a LAN to discover each other and exchange
events without any central orchestrator. Each node announces its presence
via periodic heartbeats and selectively bridges local EventBus events to
the network.

Design principles:
- No orchestration: nodes are peers, not leader/follower
- Gossip-style: eventual consistency, not strong consistency
- Fire-and-forget: UDP is unreliable by design; messages are idempotent
- LAN-only: TTL=1 means packets never leave the local subnet
- Zero infrastructure: no broker, no registry, no DNS

Architecture:
    Local EventBus ←→ [Selective Bridge] ←→ MulticastBus ←→ LAN (239.255.77.1:50077)
                                                             ↕
                                                    Other Mosaic Nodes

Usage:
    from core.swarm import SwarmNode

    node = SwarmNode(
        node_id="mosaic-alpha",
        capabilities=["llm", "visual_cortex", "auditory_cortex"],
        on_receive=lambda msg: event_bus.publish(msg["topic"], msg["payload"]),
    )
    node.start()

    # Publish to all peers
    node.broadcast("organ.memory.state_update", {"key": "ada.location", "value": "rome"})

    # Check who's online
    peers = node.peers()
    # [PeerInfo(node_id="mosaic-beta", capabilities=["llm", "affect"], ...)]

    node.stop()
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import socket
import struct
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

logger = logging.getLogger(__name__)

try:
    import msgpack
    HAS_MSGPACK = True
except ImportError:
    HAS_MSGPACK = False

# --- Protocol constants ---

MCAST_GROUP = "239.255.77.1"   # Administratively scoped, site-local
MCAST_PORT = 50077             # Mosaic-specific port
MCAST_TTL = 1                  # LAN-only: never crosses a router
PROTOCOL_VERSION = 1

HEARTBEAT_INTERVAL_S = 2.0     # Announce every 2 seconds
PEER_TIMEOUT_S = 8.0           # 4× heartbeat = dead
MAX_PACKET_SIZE = 65507        # UDP maximum payload

# Message types
MSG_ANNOUNCE = "announce"
MSG_EVENT = "event"
MSG_BYE = "bye"
MSG_STATE = "state"            # Full state snapshot (periodic)


# --- Peer tracking ---

@dataclass
class PeerInfo:
    """Information about a discovered peer node."""
    node_id: str
    address: tuple[str, int]       # (ip, port) from recvfrom
    capabilities: list[str] = field(default_factory=list)
    last_seen: float = field(default_factory=time.monotonic)
    last_seq: int = 0
    join_time: float = field(default_factory=time.time)

    @property
    def age_seconds(self) -> float:
        return time.monotonic() - self.last_seen

    @property
    def is_stale(self) -> bool:
        return self.age_seconds > PEER_TIMEOUT_S


class PeerTable:
    """Thread-safe table of discovered peers with automatic expiry."""

    def __init__(self):
        self._peers: dict[str, PeerInfo] = {}
        self._lock = threading.Lock()

    def update(self, node_id: str, address: tuple, capabilities: list[str], seq: int) -> bool:
        """Update peer info. Returns True if this is a newly discovered peer."""
        with self._lock:
            is_new = node_id not in self._peers
            if is_new:
                self._peers[node_id] = PeerInfo(
                    node_id=node_id,
                    address=address,
                    capabilities=capabilities,
                    last_seq=seq,
                )
            else:
                peer = self._peers[node_id]
                peer.last_seen = time.monotonic()
                peer.last_seq = seq
                peer.capabilities = capabilities
                peer.address = address
            return is_new

    def remove(self, node_id: str) -> PeerInfo | None:
        """Explicitly remove a peer (e.g., on BYE message)."""
        with self._lock:
            return self._peers.pop(node_id, None)

    def expire(self) -> list[PeerInfo]:
        """Remove stale peers. Returns list of removed peers."""
        now = time.monotonic()
        with self._lock:
            dead = [p for p in self._peers.values() if now - p.last_seen > PEER_TIMEOUT_S]
            for p in dead:
                del self._peers[p.node_id]
            return dead

    def live(self) -> list[PeerInfo]:
        """Return all currently live peers."""
        with self._lock:
            return [p for p in self._peers.values() if not p.is_stale]

    def __len__(self) -> int:
        with self._lock:
            return len(self._peers)

    def __contains__(self, node_id: str) -> bool:
        with self._lock:
            return node_id in self._peers


class SequenceTracker:
    """Per-sender deduplication via monotonic sequence numbers."""

    def __init__(self):
        self._seen: dict[str, int] = defaultdict(lambda: -1)

    def is_duplicate(self, sender_id: str, seq: int) -> bool:
        """Returns True if we've already processed this or a later seq from sender."""
        last = self._seen[sender_id]
        if seq <= last:
            return True
        self._seen[sender_id] = seq
        return False


# --- Multicast socket helpers ---

def _make_sender_socket() -> socket.socket:
    """Create a UDP socket for sending to the multicast group."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, MCAST_TTL)
    # Enable loopback so we can test multiple nodes on one machine
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)
    return sock


def _make_receiver_socket() -> socket.socket:
    """Create a UDP socket for receiving from the multicast group."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    except (AttributeError, OSError):
        pass  # Windows / some Linux kernels
    sock.bind(("", MCAST_PORT))
    # Join the multicast group
    group = socket.inet_aton(MCAST_GROUP)
    mreq = struct.pack("4sL", group, socket.INADDR_ANY)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    sock.settimeout(1.0)  # 1s timeout for clean shutdown polling
    return sock


# --- Serialization ---

def _pack(msg: dict) -> bytes:
    """Serialize message to bytes via msgpack."""
    if not HAS_MSGPACK:
        import json
        return json.dumps(msg, separators=(",", ":"), default=str).encode("utf-8")
    return msgpack.packb(msg, use_bin_type=True)


def _unpack(data: bytes) -> dict | None:
    """Deserialize bytes to message dict. Returns None on failure."""
    try:
        if HAS_MSGPACK:
            return msgpack.unpackb(data, raw=False)
        else:
            import json
            return json.loads(data.decode("utf-8"))
    except Exception:
        return None


# --- The SwarmNode ---

class SwarmNode:
    """A single Mosaic node participating in the LAN swarm.

    Manages:
    - Periodic heartbeat announcements
    - Peer discovery and timeout
    - Event broadcasting to all peers
    - Event reception and deduplication
    - Optional HMAC signing for untrusted LANs

    The node runs two daemon threads (receiver + heartbeat) and exposes
    a simple publish/subscribe interface that bridges to the local EventBus.
    """

    def __init__(
        self,
        *,
        node_id: str | None = None,
        capabilities: Sequence[str] | None = None,
        on_receive: Callable[[dict], None] | None = None,
        on_peer_joined: Callable[[PeerInfo], None] | None = None,
        on_peer_left: Callable[[PeerInfo], None] | None = None,
        shared_secret: bytes | None = None,
        multicast_group: str = MCAST_GROUP,
        multicast_port: int = MCAST_PORT,
    ):
        if not HAS_MSGPACK:
            logger.warning("SwarmNode: msgpack not installed, falling back to JSON (slower)")

        self.node_id = node_id or f"mosaic-{uuid.uuid4().hex[:8]}"
        self.capabilities = list(capabilities or [])
        self._on_receive = on_receive
        self._on_peer_joined = on_peer_joined
        self._on_peer_left = on_peer_left
        self._shared_secret = shared_secret
        self._mcast_group = multicast_group
        self._mcast_port = multicast_port

        self._peer_table = PeerTable()
        self._seq_tracker = SequenceTracker()
        self._seq = 0
        self._seq_lock = threading.Lock()
        self._running = threading.Event()

        self._tx_sock: socket.socket | None = None
        self._rx_sock: socket.socket | None = None
        self._rx_thread: threading.Thread | None = None
        self._hb_thread: threading.Thread | None = None

        self._total_sent = 0
        self._total_received = 0
        self._total_dropped = 0

    @property
    def peers(self) -> list[PeerInfo]:
        """Currently live peers."""
        return self._peer_table.live()

    @property
    def n_peers(self) -> int:
        return len(self._peer_table)

    @property
    def is_running(self) -> bool:
        return self._running.is_set()

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "running": self.is_running,
            "n_peers": self.n_peers,
            "total_sent": self._total_sent,
            "total_received": self._total_received,
            "total_dropped": self._total_dropped,
            "capabilities": self.capabilities,
            "peers": [{"node_id": p.node_id, "caps": p.capabilities, "age_s": round(p.age_seconds, 1)} for p in self.peers],
        }

    def start(self) -> None:
        """Start the swarm node (heartbeat + receiver threads)."""
        if self._running.is_set():
            return

        self._tx_sock = _make_sender_socket()
        self._rx_sock = _make_receiver_socket()
        self._running.set()

        self._rx_thread = threading.Thread(
            target=self._receiver_loop, daemon=True, name=f"swarm-rx-{self.node_id[:8]}"
        )
        self._hb_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True, name=f"swarm-hb-{self.node_id[:8]}"
        )
        self._rx_thread.start()
        self._hb_thread.start()

        logger.info("SwarmNode started: %s (caps=%s, group=%s:%d)",
                    self.node_id, self.capabilities, self._mcast_group, self._mcast_port)

    def stop(self) -> None:
        """Graceful shutdown: send BYE, close sockets, join threads."""
        if not self._running.is_set():
            return

        self._running.clear()

        # Best-effort BYE
        self._send_raw({"v": PROTOCOL_VERSION, "type": MSG_BYE,
                        "src": self.node_id, "seq": self._next_seq()})

        # Close sockets
        if self._tx_sock:
            try:
                self._tx_sock.close()
            except OSError:
                pass
        if self._rx_sock:
            try:
                group = socket.inet_aton(self._mcast_group)
                mreq = struct.pack("4sL", group, socket.INADDR_ANY)
                self._rx_sock.setsockopt(socket.IPPROTO_IP, socket.IP_DROP_MEMBERSHIP, mreq)
                self._rx_sock.close()
            except OSError:
                pass

        # Join threads
        if self._rx_thread:
            self._rx_thread.join(timeout=3.0)
        if self._hb_thread:
            self._hb_thread.join(timeout=3.0)

        logger.info("SwarmNode stopped: %s", self.node_id)

    def broadcast(self, topic: str, payload: dict) -> None:
        """Broadcast an event to all peers on the LAN.

        Args:
            topic: Event topic string (e.g., "organ.memory.update")
            payload: Arbitrary dict payload (must be msgpack-serializable)
        """
        if not self._running.is_set():
            return

        msg = {
            "v": PROTOCOL_VERSION,
            "type": MSG_EVENT,
            "src": self.node_id,
            "seq": self._next_seq(),
            "ts": time.time(),
            "topic": topic,
            "payload": payload,
        }
        self._send_raw(msg)
        self._total_sent += 1

    def broadcast_state(self, state: dict) -> None:
        """Broadcast a full state snapshot (periodic, for catch-up)."""
        msg = {
            "v": PROTOCOL_VERSION,
            "type": MSG_STATE,
            "src": self.node_id,
            "seq": self._next_seq(),
            "ts": time.time(),
            "state": state,
        }
        self._send_raw(msg)
        self._total_sent += 1

    # --- Internal ---

    def _next_seq(self) -> int:
        with self._seq_lock:
            self._seq += 1
            return self._seq

    def _send_raw(self, msg: dict) -> None:
        """Serialize and send a message to the multicast group."""
        data = _pack(msg)
        if self._shared_secret:
            mac = hmac.new(self._shared_secret, data, hashlib.sha256).digest()[:16]
            data = data + mac
        if len(data) > MAX_PACKET_SIZE:
            logger.warning("Message too large (%d bytes), dropping", len(data))
            return
        try:
            if self._tx_sock:
                self._tx_sock.sendto(data, (self._mcast_group, self._mcast_port))
        except OSError as exc:
            logger.debug("Send failed: %s", exc)

    def _verify_and_unpack(self, data: bytes) -> dict | None:
        """Verify HMAC (if secret set) and deserialize."""
        if self._shared_secret:
            if len(data) < 16:
                return None
            body, mac = data[:-16], data[-16:]
            expected = hmac.new(self._shared_secret, body, hashlib.sha256).digest()[:16]
            if not hmac.compare_digest(mac, expected):
                self._total_dropped += 1
                return None
            data = body
        return _unpack(data)

    def _receiver_loop(self) -> None:
        """Main receive loop — runs in daemon thread."""
        while self._running.is_set():
            try:
                data, addr = self._rx_sock.recvfrom(65535)
            except socket.timeout:
                # Expire dead peers on each timeout
                dead = self._peer_table.expire()
                for peer in dead:
                    logger.debug("Peer expired: %s", peer.node_id)
                    if self._on_peer_left:
                        self._on_peer_left(peer)
                continue
            except OSError:
                break

            msg = self._verify_and_unpack(data)
            if msg is None:
                continue

            # Protocol version check
            if msg.get("v") != PROTOCOL_VERSION:
                continue

            src = msg.get("src", "")
            seq = msg.get("seq", 0)

            # Skip our own messages
            if src == self.node_id:
                continue

            # Deduplication
            if self._seq_tracker.is_duplicate(src, seq):
                self._total_dropped += 1
                continue

            self._total_received += 1
            msg_type = msg.get("type")

            if msg_type == MSG_ANNOUNCE:
                caps = msg.get("caps", [])
                is_new = self._peer_table.update(src, addr, caps, seq)
                if is_new:
                    logger.info("Peer discovered: %s @ %s (caps=%s)", src, addr, caps)
                    if self._on_peer_joined:
                        peer = self._peer_table._peers.get(src)
                        if peer:
                            self._on_peer_joined(peer)

            elif msg_type == MSG_EVENT:
                if self._on_receive:
                    self._on_receive(msg)

            elif msg_type == MSG_STATE:
                if self._on_receive:
                    self._on_receive(msg)

            elif msg_type == MSG_BYE:
                removed = self._peer_table.remove(src)
                if removed:
                    logger.info("Peer departed: %s (graceful)", src)
                    if self._on_peer_left:
                        self._on_peer_left(removed)

    def _heartbeat_loop(self) -> None:
        """Periodic announce loop — runs in daemon thread."""
        while self._running.is_set():
            msg = {
                "v": PROTOCOL_VERSION,
                "type": MSG_ANNOUNCE,
                "src": self.node_id,
                "seq": self._next_seq(),
                "ts": time.time(),
                "caps": self.capabilities,
            }
            self._send_raw(msg)
            # Interruptible sleep
            self._running.wait(timeout=HEARTBEAT_INTERVAL_S)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass
