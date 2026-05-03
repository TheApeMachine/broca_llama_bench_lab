"""Process-wide registry of closed-form alignment matrices.

Each alignment is constructed once at substrate startup from a pair of frozen
pretrained matrices and then cached forever. Callers look up alignments by
name (e.g. ``"gliner2->gliclass"`` or ``"swm->llama"``) and apply them via the
:class:`BaseAlignment` interface.

The registry is the single source of truth for "which matrix translates which
representation into which target space"; nothing in the substrate constructs
an alignment ad-hoc.
"""

from __future__ import annotations

import threading
from typing import Iterator

from .base import BaseAlignment


class AlignmentRegistry:
    """Name-keyed registry of closed-form alignment operators."""

    def __init__(self) -> None:
        self._alignments: dict[str, BaseAlignment] = {}
        self._lock = threading.Lock()

    def register(self, alignment: BaseAlignment) -> None:
        with self._lock:
            if alignment.name in self._alignments:
                raise ValueError(
                    f"AlignmentRegistry.register: name {alignment.name!r} already registered"
                )
            self._alignments[alignment.name] = alignment

    def get(self, name: str) -> BaseAlignment:
        with self._lock:
            alignment = self._alignments.get(str(name))

        if alignment is None:
            raise KeyError(f"AlignmentRegistry.get: no alignment named {name!r}")

        return alignment

    def has(self, name: str) -> bool:
        with self._lock:
            return str(name) in self._alignments

    def names(self) -> list[str]:
        with self._lock:
            return list(self._alignments.keys())

    def __len__(self) -> int:
        with self._lock:
            return len(self._alignments)

    def __iter__(self) -> Iterator[BaseAlignment]:
        with self._lock:
            return iter(list(self._alignments.values()))
