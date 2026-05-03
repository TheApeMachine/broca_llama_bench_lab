"""SubstrateConceptGraft — continuous concept attraction / repulsion.

Replaces token-level logit biasing. The substrate supplies token IDs grouped by
concept; the graft fetches the corresponding ``lm_head`` rows, averages each
group into a unit direction, bundles the per-concept directions, and injects
the result into the residual stream at the last position:

* ``broca_concept_token_ids: Mapping[str, Sequence[int]]`` — concepts to
  attract toward. The bundled direction is added to the residual.
* ``broca_repulsion_token_ids: Mapping[str, Sequence[int]]`` — concepts to
  suppress. The bundled direction is subtracted, geometrically pushing
  subsequent attention queries (and the final lm_head projection) away from
  the concept's vocabulary subspace.

This operates entirely in the host's continuous geometry — no per-token logit
edits at the vocab layer — so a concept the substrate has just orthogonalized
out of the OntologicalRegistry composes naturally with the frozen LLM's
auto-regressive composition rather than colliding with a single discrete ID.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from ..grafting.grafts import (
    BaseGraft,
    DEFAULT_GRAFT_TARGET_SNR,
    snr_magnitude,
    state_confidence,
    state_inertia,
    state_target_snr_scale,
)


class SubstrateConceptGraft(BaseGraft):
    """Inject continuous concept directions into the residual stream."""

    def __init__(self, *, target_snr: float = DEFAULT_GRAFT_TARGET_SNR):
        super().__init__()
        self.target_snr = float(target_snr)
        self.mixer_priority = 0.5
        self.last_debug: dict = {}

    def _bundled_direction(
        self,
        lm_weight: torch.Tensor,
        groups: Mapping[str, Sequence[int]],
    ) -> torch.Tensor | None:
        if not groups:
            return None

        directions: list[torch.Tensor] = []
        vocab_size = int(lm_weight.shape[0])

        for token_ids in groups.values():
            ids = [int(t) for t in token_ids if 0 <= int(t) < vocab_size]

            if not ids:
                continue

            idx = torch.as_tensor(ids, device=lm_weight.device, dtype=torch.long)
            mean = lm_weight.index_select(0, idx).mean(dim=0)
            directions.append(F.normalize(mean, dim=-1))

        if not directions:
            return None

        bundled = torch.stack(directions, dim=0).mean(dim=0)

        return F.normalize(bundled, dim=-1)

    def _last_position(self, x: torch.Tensor, state: dict) -> torch.Tensor:
        last = state.get("last_indices")

        if last is None:
            raise ValueError(
                "SubstrateConceptGraft.forward: missing required state key 'last_indices'"
            )

        return last.to(x.device)

    def forward(self, x: torch.Tensor, state: dict) -> torch.Tensor:
        if not self.enabled:
            return x

        attract = state.get("broca_concept_token_ids") or {}
        repel = state.get("broca_repulsion_token_ids") or {}

        if not attract and not repel:
            return x

        model = state.get("model")

        if model is None:
            raise ValueError(
                "SubstrateConceptGraft.forward: state['model'] required to read lm_head rows"
            )

        lm_weight = model.lm_head.weight.detach().to(device=x.device, dtype=x.dtype)
        attract_dir = self._bundled_direction(lm_weight, attract)
        repel_dir = self._bundled_direction(lm_weight, repel)

        if attract_dir is None and repel_dir is None:
            return x

        last = self._last_position(x, state)
        rows = torch.arange(x.shape[0], device=x.device)
        host_at_last = x[rows, last]

        magnitude = snr_magnitude(
            host_at_last,
            target_snr=self.target_snr,
            confidence=state_confidence(state),
            inertia=state_inertia(state),
            substrate_scale=state_target_snr_scale(state),
        )

        delta = torch.zeros_like(host_at_last)

        if attract_dir is not None:
            delta = delta + attract_dir.unsqueeze(0).expand_as(host_at_last) * magnitude

        if repel_dir is not None:
            delta = delta - repel_dir.unsqueeze(0).expand_as(host_at_last) * magnitude

        out = x.clone()
        out[rows, last] = out[rows, last] + delta

        self.last_debug = {
            "attract_concepts": list(attract.keys()) if attract else [],
            "repel_concepts": list(repel.keys()) if repel else [],
            "magnitude_max": float(magnitude.detach().max().item()) if magnitude.numel() else 0.0,
        }

        return out
