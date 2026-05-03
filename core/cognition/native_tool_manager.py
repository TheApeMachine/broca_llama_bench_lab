"""NativeToolManager — substrate-side façade over native tool synthesis.

The substrate controller used to inline four methods that wrapped
:class:`NativeToolRegistry` and :class:`ToolForagingAgent`. They cluster
under one concern: deciding whether the substrate's confusion warrants
synthesizing a new SCM equation, performing the synthesis, attaching it,
and propagating drift back into intrinsic cues. That cluster lives here.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from ..agent.active_inference import ToolForagingAgent, entropy as belief_entropy
from ..natives.native_tools import NativeTool
from ..workspace import IntrinsicCue


if TYPE_CHECKING:
    from .substrate import SubstrateController


logger = logging.getLogger(__name__)


class NativeToolManager:
    """Thin façade exposing the native-tool surface the controller used to own."""

    def __init__(self, mind: "SubstrateController") -> None:
        self._mind = mind

    def handle_drift(self, tool: NativeTool, evidence: Mapping[str, Any]) -> None:
        """Turn native-tool exchangeability drift into an active-inference cue."""

        mind = self._mind
        cue = IntrinsicCue(
            urgency=1.0,
            faculty="tool_resynthesis",
            evidence={
                "tool": tool.name,
                "parents": list(tool.parents),
                "domain": [repr(v) for v in tool.domain],
                **dict(evidence),
            },
            source="native_tool_martingale",
        )
        mind.workspace.intrinsic_cues.append(cue)
        mind.tool_foraging_agent = ToolForagingAgent.build(
            n_existing_tools=mind.tool_registry.count(),
            insufficient_prior=1.0 - 1e-6,
        )
        mind.event_bus.publish(
            "native_tool.drift",
            {"tool": tool.name, "urgency": cue.urgency, "evidence": dict(cue.evidence)},
        )

    def synthesize(
        self,
        name: str,
        source: str,
        *,
        function_name: str | None = None,
        parents: Sequence[str],
        domain: Sequence[Any],
        sample_inputs: Sequence[dict],
        description: str = "",
        attach: bool = True,
        overwrite: bool = False,
    ) -> NativeTool:
        mind = self._mind
        tool = mind.tool_registry.synthesize(
            name,
            source,
            function_name=function_name,
            parents=parents,
            domain=domain,
            sample_inputs=sample_inputs,
            description=description,
            overwrite=overwrite,
            conformal_predictor=mind.native_tool_conformal,
        )
        if attach:
            try:
                mind.tool_registry.attach_to_scm(
                    mind.scm,
                    topology_lock=mind._cognitive_state_lock,
                    on_tool_drift=mind._handle_native_tool_drift,
                )
            except Exception:
                logger.exception("NativeToolManager.synthesize: SCM re-attach failed")
        mind.tool_foraging_agent = ToolForagingAgent.build(
            n_existing_tools=mind.tool_registry.count(),
            insufficient_prior=0.5,
        )
        return tool

    def attach_to_scm(self) -> int:
        """Re-attach every persisted native tool onto the SCM. Returns count attached."""

        mind = self._mind
        return mind.tool_registry.attach_to_scm(
            mind.scm,
            topology_lock=mind._cognitive_state_lock,
            on_tool_drift=mind._handle_native_tool_drift,
        )

    def should_synthesize(self) -> bool:
        """Run the tool foraging agent against the current substrate state."""

        mind = self._mind
        try:
            coupled = mind.unified_agent.decide()
        except Exception:
            return False
        if coupled.faculty == "spatial":
            posterior = list(coupled.spatial_decision.posterior_over_policies)
        else:
            posterior = list(coupled.causal_decision.posterior_over_policies)
        n = len(posterior)
        if n < 2:
            insufficient_prior = 0.5
        else:
            h = belief_entropy(posterior)
            h_max = math.log(n)
            insufficient_prior = max(1e-6, min(1 - 1e-6, h / max(h_max, 1e-9)))
        mind.tool_foraging_agent.update_belief(
            insufficient_prior=float(insufficient_prior)
        )
        return mind.tool_foraging_agent.should_synthesize()
