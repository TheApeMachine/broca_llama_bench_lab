"""Token sampling algorithms."""

from __future__ import annotations

import torch


class Sampling:
    """Named token selection algorithms for logits."""

    def next_token(
        self,
        logits: torch.Tensor,
        *,
        do_sample: bool,
        temperature: float,
        top_p: float,
    ) -> int:
        if not do_sample:
            return self.greedy(logits)
        return self.nucleus(logits, temperature=temperature, top_p=top_p)

    def greedy(self, logits: torch.Tensor) -> int:
        return int(logits.argmax().item())

    def nucleus(self, logits: torch.Tensor, *, temperature: float, top_p: float) -> int:
        scaled = logits / max(float(temperature), 1e-5)
        probabilities = torch.softmax(scaled, dim=-1)
        sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True)
        cumulative = torch.cumsum(sorted_probabilities, dim=-1)
        over_threshold = (cumulative > float(top_p)).nonzero(as_tuple=False)
        keep = (
            int(over_threshold[0, 0].item()) + 1
            if over_threshold.numel() > 0
            else int(probabilities.numel())
        )
        kept_probabilities = sorted_probabilities[: max(1, keep)]
        kept_indices = sorted_indices[: max(1, keep)]
        kept_probabilities = kept_probabilities / kept_probabilities.sum().clamp_min(1e-12)
        pick = int(torch.multinomial(kept_probabilities, num_samples=1).item())
        return int(kept_indices[pick].item())
