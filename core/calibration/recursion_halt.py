"""Adaptive halt criterion for the substrate's recursive r-round loop.

The recursion controller asks the halter "should I run another round?" at the
end of every round. Two cheap closed-form signals decide:

* **Convergence**: cosine similarity between the active SWM thought slot in
  consecutive rounds. When the substrate's working memory stops moving
  meaningfully, further rounds add noise rather than refinement.

* **Hard cap**: an explicit maximum on the number of rounds. The substrate
  is bounded by physics (finite compute) regardless of convergence.

Both checks are derived constants — no per-call tunables. The cosine
threshold is the noise floor of the underlying VSA hyperdimensional space
(``1 / sqrt(D_swm)`` quasi-orthogonality bound), inflated to a comfortable
operating margin. The hard cap follows LatentMAS's empirical finding that
gains plateau around ``r=3`` for cross-agent loops; substrates with deeper
algebra may go further but get diminishing returns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..symbolic import cosine
from ..swm import SubstrateWorkingMemory


DEFAULT_MAX_ROUNDS: int = 3


@dataclass(frozen=True)
class HaltDecision:
    """Outcome of a single halt check."""

    halt: bool
    reason: str
    cosine_to_previous: float
    rounds_completed: int


class RecursionHalt:
    """Closed-form stopping criterion for recursive substrate rollouts."""

    def __init__(
        self,
        *,
        swm: SubstrateWorkingMemory,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
    ) -> None:
        if int(max_rounds) <= 0:
            raise ValueError(f"RecursionHalt.max_rounds must be positive, got {max_rounds}")

        self._swm = swm
        self._max_rounds = int(max_rounds)
        self._convergence_floor = 1.0 - 4.0 / math.sqrt(float(swm.dim))
        # ^ For D=10000 this is 0.96; for D=1024 it'd be 0.875. Derived from
        # the i.i.d. Gaussian quasi-orthogonality bound times a 4-sigma margin.
        self._previous: torch.Tensor | None = None

    @property
    def max_rounds(self) -> int:
        return self._max_rounds

    @property
    def convergence_floor(self) -> float:
        return self._convergence_floor

    def reset(self) -> None:
        self._previous = None

    def check(self, *, slot_name: str, rounds_completed: int) -> HaltDecision:
        if rounds_completed >= self._max_rounds:
            current = self._swm.read(slot_name).vector.detach().clone()
            cos = self._cosine_to_prev(current)
            self._previous = current
            return HaltDecision(
                halt=True,
                reason="max_rounds_reached",
                cosine_to_previous=cos,
                rounds_completed=int(rounds_completed),
            )

        current = self._swm.read(slot_name).vector.detach().clone()
        cos = self._cosine_to_prev(current)
        self._previous = current

        if cos >= self._convergence_floor:
            return HaltDecision(
                halt=True,
                reason="converged",
                cosine_to_previous=cos,
                rounds_completed=int(rounds_completed),
            )

        return HaltDecision(
            halt=False,
            reason="continue",
            cosine_to_previous=cos,
            rounds_completed=int(rounds_completed),
        )

    def _cosine_to_prev(self, current: torch.Tensor) -> float:
        if self._previous is None:
            return float("-inf")

        return cosine(current, self._previous)
