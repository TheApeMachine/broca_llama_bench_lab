"""NativeToolManager — façade over native tool synthesis (registry + foraging belief)."""

from __future__ import annotations

import logging
import math
from typing import Any, Mapping, Sequence

from ..agent.active_inference import ToolForagingAgent, entropy as belief_entropy
from ..workspace import IntrinsicCue
from .hypothesis_synthesizer import HypothesisSynthesizer
from .native_tools import NativeTool, ToolSynthesisError
from .tool_foraging_slot import ToolForagingSlot


logger = logging.getLogger(__name__)


class NativeToolManager:
    """Thin façade over :class:`NativeToolRegistry` and foraging belief updates."""

    def __init__(
        self,
        *,
        tool_registry: Any,
        scm: Any,
        workspace: Any,
        event_bus: Any,
        slot: ToolForagingSlot,
        unified_agent: Any,
        native_tool_conformal: Any,
        session: Any,
    ) -> None:
        self._registry = tool_registry
        self._scm = scm
        self._workspace = workspace
        self._event_bus = event_bus
        self._slot = slot
        self._unified = unified_agent
        self._tool_conformal = native_tool_conformal
        self._session = session
        self._hypothesis_synthesizer = HypothesisSynthesizer(scm=scm, tool_registry=tool_registry)

    def handle_drift(self, tool: NativeTool, evidence: Mapping[str, Any]) -> None:
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
        self._workspace.intrinsic_cues.append(cue)
        self._slot.agent = ToolForagingAgent.build(
            n_existing_tools=self._registry.count(),
            insufficient_prior=1.0 - 1e-6,
        )
        self._event_bus.publish(
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
        tool = self._registry.synthesize(
            name,
            source,
            function_name=function_name,
            parents=parents,
            domain=domain,
            sample_inputs=sample_inputs,
            description=description,
            overwrite=overwrite,
            conformal_predictor=self._tool_conformal,
        )
        if attach:
            try:
                self._registry.attach_to_scm(
                    self._scm,
                    topology_lock=self._session.cognitive_state_lock,
                    on_tool_drift=self.handle_drift,
                )
            except Exception:
                logger.exception("NativeToolManager.synthesize: SCM re-attach failed")
        self._slot.agent = ToolForagingAgent.build(
            n_existing_tools=self._registry.count(),
            insufficient_prior=0.5,
        )
        return tool

    def attach_to_scm(self) -> int:
        return self._registry.attach_to_scm(
            self._scm,
            topology_lock=self._session.cognitive_state_lock,
            on_tool_drift=self.handle_drift,
        )

    def should_synthesize(self) -> bool:
        try:
            coupled = self._unified.decide()
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
        self._slot.agent.update_belief(insufficient_prior=float(insufficient_prior))
        return bool(self._slot.agent.should_synthesize())

    def synthesize_recommended(self) -> NativeTool | None:
        """Author and persist the next conjunction hypothesis when foraging recommends it.

        Closes the loop between the EFE math (which decides *that* a new tool is
        needed) and the registry (which holds *what* tools exist). The hypothesis
        synthesizer picks the next uncovered binary endogenous pair, compiles +
        verifies the conjunction through the existing sandbox / conformal gate,
        and re-attaches the registry to the SCM so the new node is queryable.
        """

        if not self.should_synthesize():
            return None

        try:
            tool = self._hypothesis_synthesizer.attempt_one()
        except ToolSynthesisError:
            logger.exception("NativeToolManager.synthesize_recommended: hypothesis synthesis failed")
            return None

        if tool is None:
            return None

        try:
            self._registry.attach_to_scm(
                self._scm,
                topology_lock=self._session.cognitive_state_lock,
                on_tool_drift=self.handle_drift,
            )
        except Exception:
            logger.exception("NativeToolManager.synthesize_recommended: SCM re-attach failed")

        self._slot.agent = ToolForagingAgent.build(
            n_existing_tools=self._registry.count(),
            insufficient_prior=0.5,
        )
        self._event_bus.publish(
            "native_tool.synthesized",
            {"tool": tool.name, "parents": list(tool.parents), "domain": list(tool.domain)},
        )

        return tool
