"""The Substrate Working Memory itself.

A name-keyed registry of :class:`SWMSlot` plus the algebraic combinators that
let the substrate's higher-level reasoning (active inference, SCM,
ComprehensionPipeline) compose new slots from existing ones via the VSA
primitives. All operations are closed-form; nothing here is trainable.
"""

from __future__ import annotations

import threading
from typing import Iterable, Iterator, Sequence

import torch

from ..symbolic import DEFAULT_VSA_DIM, bind, bundle, cleanup, cosine, permute, unbind
from ..workspace import WorkspacePublisher
from .source import SWMSource
from .swm_slot import SWMSlot


class SubstrateWorkingMemory:
    """Continuous global workspace at ``DEFAULT_VSA_DIM`` width."""

    def __init__(self, *, dim: int = DEFAULT_VSA_DIM) -> None:
        if int(dim) <= 0:
            raise ValueError(f"SubstrateWorkingMemory.dim must be positive, got {dim}")

        self.dim = int(dim)
        self._slots: dict[str, SWMSlot] = {}
        self._tick: int = 0
        self._lock = threading.Lock()

    # -- single-slot API ------------------------------------------------------

    def write(self, name: str, vector: torch.Tensor, *, source: SWMSource) -> SWMSlot:
        if vector.shape[-1] != self.dim:
            raise ValueError(
                f"SubstrateWorkingMemory.write: vector last dim must be {self.dim}, got {vector.shape[-1]}"
            )

        # SWM slots participate in VSA bind/bundle with :class:`VSACodebook` atoms,
        # which are materialized on CPU. Keeping workspace vectors on CPU avoids
        # mps:0 vs cpu mixed-device fft/matmul when encoders run on Metal.
        flat = vector.detach().to(dtype=torch.float32).cpu().view(-1).contiguous()

        with self._lock:
            self._tick += 1
            slot = SWMSlot(
                name=str(name),
                vector=flat,
                source=source,
                written_at_tick=self._tick,
            )
            self._slots[slot.name] = slot
            tick_snapshot = self._tick
            slot_count = len(self._slots)

        WorkspacePublisher.emit(
            "swm.write",
            {
                "slot": slot.name,
                "source": source.value,
                "tick": tick_snapshot,
                "slot_count": slot_count,
                "norm": float(flat.norm().item()),
            },
        )

        return slot

    def read(self, name: str) -> SWMSlot:
        with self._lock:
            slot = self._slots.get(str(name))

        if slot is None:
            raise KeyError(f"SubstrateWorkingMemory.read: no slot named {name!r}")

        return slot

    def has(self, name: str) -> bool:
        with self._lock:
            return str(name) in self._slots

    def remove(self, name: str) -> SWMSlot:
        with self._lock:
            slot = self._slots.pop(str(name), None)

        if slot is None:
            raise KeyError(f"SubstrateWorkingMemory.remove: no slot named {name!r}")

        return slot

    def names(self) -> list[str]:
        with self._lock:
            return list(self._slots.keys())

    def __len__(self) -> int:
        with self._lock:
            return len(self._slots)

    def __iter__(self) -> Iterator[SWMSlot]:
        with self._lock:
            return iter(list(self._slots.values()))

    # -- algebraic combinators ------------------------------------------------

    def bind_slots(self, name_a: str, name_b: str, *, into: str) -> SWMSlot:
        a = self.read(name_a)
        b = self.read(name_b)
        return self.write(into, bind(a.vector, b.vector), source=SWMSource.SUBSTRATE_ALGEBRA)

    def unbind_slots(self, encoded: str, role: str, *, into: str) -> SWMSlot:
        c = self.read(encoded)
        r = self.read(role)
        return self.write(into, unbind(c.vector, r.vector), source=SWMSource.SUBSTRATE_ALGEBRA)

    def bundle_slots(self, names: Sequence[str], *, into: str) -> SWMSlot:
        slots = [self.read(n) for n in names]

        if not slots:
            raise ValueError("SubstrateWorkingMemory.bundle_slots requires at least one slot")

        return self.write(
            into,
            bundle([s.vector for s in slots]),
            source=SWMSource.SUBSTRATE_ALGEBRA,
        )

    def permute_slot(self, name: str, *, shift: int, into: str) -> SWMSlot:
        s = self.read(name)
        return self.write(into, permute(s.vector, shift=int(shift)), source=SWMSource.SUBSTRATE_ALGEBRA)

    # -- queries --------------------------------------------------------------

    def cosine_to(self, name: str, vector: torch.Tensor) -> float:
        slot = self.read(name)
        return cosine(slot.vector, vector)

    def cleanup_to_slot(self, query: torch.Tensor, *, candidates: Iterable[str] | None = None) -> tuple[str, float]:
        with self._lock:
            if candidates is None:
                book = {name: slot.vector for name, slot in self._slots.items()}
            else:
                names = list(candidates)
                book = {n: self._slots[n].vector for n in names if n in self._slots}
                missing = sorted(set(names) - set(book))

                if missing:
                    raise KeyError(
                        f"SubstrateWorkingMemory.cleanup_to_slot: unknown candidates {missing}"
                    )

        if not book:
            raise ValueError("SubstrateWorkingMemory.cleanup_to_slot: no candidates to compare against")

        return cleanup(query, book)
