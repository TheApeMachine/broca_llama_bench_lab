"""GW-Dreamer-style imagination loop over the substrate working memory.

The imaginer is the multi-modal analogue of DreamerV3's policy-evaluation
loop, fitted to the substrate's continuous global workspace:

1. **Predict K imagined futures.** Given the current observation feature
   vector ``f0 ∈ R^{d_organ}``, generate ``K`` candidate trajectories
   ``[f0, f1, ..., f_{T-1}]`` of length ``T`` each. The predictor is
   pluggable; the default is a deterministic JL-perturbed unfold from ``f0``
   that exercises the whole loop without depending on V-JEPA's predictor
   weights (which the public HF checkpoint does not expose).

2. **Run the substrate over each imagined timestep.** For every step ``t``
   of every trajectory ``k`` we publish ``f_{k,t}`` into the SWM as a
   V-JEPA hidden contribution and let the
   :class:`RecursionController` refine the workspace. Joint free energy is
   accumulated from the per-organ
   :class:`PredictionErrorVector` after each round.

3. **Pick the lowest-EFE trajectory.** The selected trajectory and its
   accumulated joint free energy are returned in :class:`ImaginedPlan`.
   Active-inference action selection downstream pulls the action label
   off the best trajectory and feeds it back into the world.

The whole loop is closed-form (JL projections, ridge alignment, VSA bind).
No predictor or actor is trained.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import torch

from ..substrate.prediction_error import PredictionErrorVector
from ..substrate.recursion_controller import RecursionController
from ..swm import EncoderSWMPublisher, SubstrateWorkingMemory, SWMSource
from ..workspace import WorkspacePublisher


DEFAULT_K_TRAJECTORIES: int = 3
DEFAULT_T_HORIZON: int = 4


@dataclass(frozen=True)
class ImaginedTrajectory:
    """One candidate sequence of future feature vectors plus its joint EFE."""

    trajectory_index: int
    features: list[torch.Tensor]
    joint_free_energy: float


@dataclass(frozen=True)
class ImaginedPlan:
    """Outcome of one imagination cycle."""

    chosen: ImaginedTrajectory
    candidates: list[ImaginedTrajectory] = field(default_factory=list)


FuturePredictor = Callable[[torch.Tensor, int, int], torch.Tensor]


def jl_unfold_predictor(
    *,
    seed: int,
    perturbation_scale: float = 0.1,
) -> FuturePredictor:
    """Deterministic perturbation predictor — placeholder for a trained world model.

    Returns ``f_{t+1} = f_t + ε`` where ``ε ~ N(0, σ² I)`` with σ derived from
    ``perturbation_scale / sqrt(d)``. Each ``(trajectory_index, step_index)``
    pair is sampled from a deterministic seed so two calls with the same
    arguments return identical futures.
    """

    def _predict(current: torch.Tensor, trajectory_index: int, step_index: int) -> torch.Tensor:
        d = int(current.shape[-1])
        g = torch.Generator(device="cpu")
        g.manual_seed(((int(seed) << 16) ^ (int(trajectory_index) << 8) ^ int(step_index)) & 0x7FFFFFFFFFFFFFFF)
        eps = torch.empty(d, dtype=torch.float32)
        eps.normal_(mean=0.0, std=float(perturbation_scale) / math.sqrt(float(d)), generator=g)
        return current.detach().to(torch.float32).view(-1) + eps

    return _predict


class VJEPAImaginer:
    """Run K-trajectory T-horizon imagination over the substrate."""

    def __init__(
        self,
        *,
        swm: SubstrateWorkingMemory,
        publisher: EncoderSWMPublisher,
        controller: RecursionController,
        prediction_errors: PredictionErrorVector,
        predictor: FuturePredictor | None = None,
        k_trajectories: int = DEFAULT_K_TRAJECTORIES,
        t_horizon: int = DEFAULT_T_HORIZON,
        seed: int = 0,
    ) -> None:
        if int(k_trajectories) <= 0:
            raise ValueError(
                f"VJEPAImaginer.k_trajectories must be positive, got {k_trajectories}"
            )

        if int(t_horizon) <= 0:
            raise ValueError(
                f"VJEPAImaginer.t_horizon must be positive, got {t_horizon}"
            )

        self._swm = swm
        self._publisher = publisher
        self._controller = controller
        self._errors = prediction_errors
        self._k = int(k_trajectories)
        self._t = int(t_horizon)
        self._predictor = predictor if predictor is not None else jl_unfold_predictor(seed=int(seed))

    @property
    def k_trajectories(self) -> int:
        return self._k

    @property
    def t_horizon(self) -> int:
        return self._t

    @torch.no_grad()
    def imagine(
        self,
        *,
        current_features: torch.Tensor,
        prompt_input_ids: torch.Tensor,
    ) -> ImaginedPlan:
        """Generate K trajectories of T futures and return the lowest-EFE plan.

        ``current_features`` is the V-JEPA feature vector for the current
        observation — shape ``[d_organ]`` or ``[1, d_organ]``.
        ``prompt_input_ids`` is the token prompt the substrate uses to drive
        each per-step latent rollout (typically the same prompt across all
        candidates so the substrate's contribution is the only varying
        signal).
        """

        if current_features.ndim not in (1, 2):
            raise ValueError(
                f"VJEPAImaginer.imagine: current_features must be 1-D or 2-D, got shape {tuple(current_features.shape)}"
            )

        f0 = current_features.detach().to(torch.float32).view(-1)
        candidates: list[ImaginedTrajectory] = []

        for k in range(self._k):
            features: list[torch.Tensor] = []
            current = f0.clone()
            joint_efe = 0.0

            for t in range(self._t):
                next_f = self._predictor(current, k, t)
                features.append(next_f)
                current = next_f

                self._publisher.publish_hidden(
                    source=SWMSource.VJEPA,
                    hidden=next_f.view(1, 1, -1),
                    confidence=1.0,
                )
                self._controller.run(input_ids=prompt_input_ids)
                joint_efe += self._errors.joint_free_energy()

            candidates.append(
                ImaginedTrajectory(
                    trajectory_index=k,
                    features=features,
                    joint_free_energy=float(joint_efe),
                )
            )

        chosen = min(candidates, key=lambda c: c.joint_free_energy)

        WorkspacePublisher.emit(
            "imagination.cycle",
            {
                "k_trajectories": self._k,
                "t_horizon": self._t,
                "chosen_index": chosen.trajectory_index,
                "chosen_efe": chosen.joint_free_energy,
                "all_efes": [c.joint_free_energy for c in candidates],
            },
        )

        return ImaginedPlan(chosen=chosen, candidates=candidates)
