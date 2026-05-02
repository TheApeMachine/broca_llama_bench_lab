"""A discovered peer on the LAN."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


PEER_TIMEOUT_S = 8.0


@dataclass
class PeerInfo:
    """A discovered peer node on the multicast group."""

    node_id: str
    address: tuple[str, int]
    capabilities: list[str] = field(default_factory=list)
    last_seen: float = field(default_factory=time.monotonic)
    last_seq: int = 0

    @property
    def age_seconds(self) -> float:
        return time.monotonic() - self.last_seen

    @property
    def is_alive(self) -> bool:
        return self.age_seconds <= PEER_TIMEOUT_S
