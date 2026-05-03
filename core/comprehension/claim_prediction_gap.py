"""ClaimPredictionGap — measure how surprised the host is by a contradicting claim.

When the substrate is about to consolidate or refuse a claim, it compares
the host's per-token surprise on that claim with grafts active vs grafts
disabled. A larger positive ``gap`` means the LLM finds the claim less
plausible; consolidation uses this as a trust attenuator so a low-prior
statement repeated by an attacker requires more corroboration to flip a
belief than a low-surprise statement.

Returns ``None`` when the host cannot run a graft-aware forward pass (e.g.
test fakes that don't implement ``return_cache``).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..cognition.predictive_coding import lexical_surprise_gap
from ..frame import ParsedClaim


if TYPE_CHECKING:
    from ..cognition.substrate import SubstrateController


logger = logging.getLogger(__name__)


class ClaimPredictionGap:
    """Stateless wrapper around the lexical-surprise probe."""

    @classmethod
    def measure(
        cls,
        mind: "SubstrateController",
        utterance: str,
        claim: ParsedClaim,
    ) -> float | None:
        try:
            plan_words = [claim.subject, claim.predicate, claim.obj, "."]
            broca_features = mind.frame_packer.broca(
                "memory_write",
                claim.subject,
                claim.obj,
                float(claim.confidence),
                claim.evidence,
                vsa_bundle=mind.encode_triple_vsa(claim.subject, claim.predicate, claim.obj),
                vsa_projection_seed=int(mind.seed),
            )
            _ce_g, _ce_p, gap = lexical_surprise_gap(
                mind.host,
                mind.tokenizer,
                utterance=utterance,
                plan_words=plan_words,
                broca_features=broca_features,
            )
            return float(gap)
        except (AttributeError, RuntimeError, TypeError, ValueError, StopIteration, IndexError):
            logger.debug(
                "ClaimPredictionGap.measure: unavailable host path utterance=%r",
                utterance[:200],
            )
            return None
