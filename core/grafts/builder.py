"""Attach lexical, continuous-feature, concept, and KV-memory grafts to the Broca host."""

from __future__ import annotations

from typing import Any

from ..frame import FrameDimensions
from ..grafting.grafts import DEFAULT_GRAFT_TARGET_SNR, KVMemoryGraft
from .concept_graft import SubstrateConceptGraft
from .lexical_plan import LexicalPlanGraft
from .trainable_feature import TrainableFeatureGraft


class HostGraftsBuilder:
    """Constructs substrate-owned grafts and registers them on ``mind.host``.

    All grafts attach at the ``final_hidden`` slot — concepts cross the frozen-LLM
    boundary as continuous directions in the residual stream rather than as
    final-layer logit edits, so the host's auto-regressive composition mediates
    expression and suppression of substrate-named concepts. The KV-memory graft
    receives activation modes captured during chat by the dynamic-graft
    synthesizer; the grafts builder only constructs and attaches it — the
    capture-and-replay loop is plumbed by the chat-time pipeline.
    """

    @classmethod
    def populate(cls, mind: Any, *, lexical_target_snr: float | None) -> None:
        snr = lexical_target_snr if lexical_target_snr is not None else DEFAULT_GRAFT_TARGET_SNR
        mind.lexical_graft = LexicalPlanGraft(target_snr=snr)
        mind.host.add_graft("final_hidden", mind.lexical_graft)
        mind.feature_graft = TrainableFeatureGraft(
            FrameDimensions.broca_feature_dim(),
            int(getattr(mind.host.cfg, "d_model", 96)),
            target_snr=snr,
        )
        host_param = None
        params = getattr(mind.host, "parameters", None)
        if callable(params):
            host_param = next(iter(params()), None)
            if host_param is not None:
                mind.feature_graft.to(host_param.device)
        mind.host.add_graft("final_hidden", mind.feature_graft)
        mind.concept_graft = SubstrateConceptGraft(target_snr=snr)
        mind.host.add_graft("final_hidden", mind.concept_graft)
        mind.kv_memory_graft = KVMemoryGraft(
            d_model=int(getattr(mind.host.cfg, "d_model", 96)),
            target_snr=snr,
        )
        if host_param is not None:
            mind.kv_memory_graft.to(host_param.device)
        mind.host.add_graft("final_hidden", mind.kv_memory_graft)
        mind._host_param = host_param
