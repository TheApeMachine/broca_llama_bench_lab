"""Multi-modal prediction error vector for joint EFE minimisation.

Each organ that publishes into the substrate working memory also publishes a
scalar prediction error in ``[0, 1]`` — typically ``1 - confidence`` for
encoders that report a confidence score, or the lexical-surprise gap for
language paths. Active inference operates on the vector across organs, not on
any single channel, so the agent's posterior over policies weights actions
that reduce the *highest-error* organ next, not just the noisiest single
signal.

The vector is the closed-form analogue of Friston's hierarchical predictive
coding: prediction errors at every level of the generative model are
integrated into a single free-energy objective. We compute the integration
explicitly and expose it as a tensor that the existing
:class:`core.agent.active_inference.CoupledEFEAgent` can consume.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Iterable

import torch

from ..swm.source import SWMSource
from ..workspace import WorkspacePublisher


@dataclass(frozen=True)
class OrganError:
    """One organ's most-recent prediction-error reading."""

    source: SWMSource
    error: float
    written_at_tick: int


class PredictionErrorVector:
    """Per-organ prediction-error registry; exposes the joint as a tensor."""

    def __init__(self) -> None:
        self._errors: dict[str, OrganError] = {}
        self._tick: int = 0
        self._lock = threading.Lock()

    def record(self, *, source: SWMSource, error: float) -> OrganError:
        if not (0.0 <= float(error) <= 1.0):
            raise ValueError(
                f"PredictionErrorVector.record: error must be in [0, 1], got {error}"
            )

        with self._lock:
            self._tick += 1
            entry = OrganError(source=source, error=float(error), written_at_tick=self._tick)
            self._errors[source.value] = entry
            joint = sum(e.error for e in self._errors.values())
            organ_count = len(self._errors)

        WorkspacePublisher.emit(
            "prediction_error.record",
            {
                "source": source.value,
                "error": float(error),
                "tick": entry.written_at_tick,
                "joint_free_energy": float(joint),
                "organ_count": int(organ_count),
            },
        )

        return entry

    def get(self, source: SWMSource) -> OrganError:
        with self._lock:
            entry = self._errors.get(source.value)

        if entry is None:
            raise KeyError(
                f"PredictionErrorVector.get: no error recorded for organ {source.value!r}"
            )

        return entry

    def has(self, source: SWMSource) -> bool:
        with self._lock:
            return source.value in self._errors

    def __len__(self) -> int:
        with self._lock:
            return len(self._errors)

    def sources(self) -> list[SWMSource]:
        with self._lock:
            entries = list(self._errors.values())

        return [e.source for e in entries]

    def as_tensor(self, *, sources: Iterable[SWMSource] | None = None) -> torch.Tensor:
        """Return a 1-D tensor of errors in the requested order.

        When ``sources`` is omitted the vector is laid out in the order organs
        were first registered. Missing organs raise — silent zero-fill would
        let the joint EFE happily ignore an absent modality, exactly the kind
        of fallback the substrate forbids.
        """

        with self._lock:
            entries = dict(self._errors)

        if sources is None:
            ordered = [e.error for e in entries.values()]
        else:
            ordered = []

            for s in sources:
                entry = entries.get(s.value)

                if entry is None:
                    raise KeyError(
                        f"PredictionErrorVector.as_tensor: requested source {s.value!r} has no recorded error"
                    )

                ordered.append(entry.error)

        return torch.tensor(ordered, dtype=torch.float32)

    def joint_free_energy(self, *, sources: Iterable[SWMSource] | None = None) -> float:
        """Sum-of-errors approximation of joint free energy across modalities.

        For uncorrelated channels this is the closed-form upper bound on the
        joint surprise. Correlation handling (covariance-weighted sum) is a
        future extension when an organ-pair covariance estimator exists.
        """

        v = self.as_tensor(sources=sources)
        return float(v.sum().item())

    def reset(self) -> None:
        with self._lock:
            self._errors.clear()
            self._tick = 0
