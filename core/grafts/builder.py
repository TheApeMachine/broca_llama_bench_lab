"""Attach lexical, continuous-feature, and logit grafts to the Broca host."""

from __future__ import annotations

from typing import Any

from ..frame import FrameDimensions
from ..grafting.grafts import DEFAULT_GRAFT_TARGET_SNR
from .lexical_plan import LexicalPlanGraft
from .logit_bias import SubstrateLogitBiasGraft
from .trainable_feature import TrainableFeatureGraft


class HostGraftsBuilder:
    """Constructs substrate-owned grafts and registers them on ``mind.host``."""

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
        mind.logit_bias_graft = SubstrateLogitBiasGraft()
        mind.host.add_graft("logits", mind.logit_bias_graft)
        mind._host_param = host_param
