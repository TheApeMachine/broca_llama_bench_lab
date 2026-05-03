"""OrchestrationLinker — attaches orchestration façades after faculties exist."""

from __future__ import annotations

from typing import Any

from ..chat.orchestrator import ChatOrchestrator
from ..chat.plan_speaker import PlanSpeaker
from ..comprehension.claim_refiner import ClaimRefiner
from ..comprehension.deferred_queue import DeferredRelationQueue
from ..comprehension.pipeline import ComprehensionPipeline
from ..dmn.worker_supervisor import WorkerSupervisor
from ..grafts.feature import FrameGraftProjection
from ..idletime.macro_adapter import MacroAdapter
from ..learning.preference_adapter import PreferenceAdapter
from ..memory.algebraic_adapter import AlgebraicMemoryAdapter
from ..natives.native_tool_manager import NativeToolManager
from .inspector import SubstrateInspector


class OrchestrationLinker:
    """Final wiring step after :class:`SubstrateBuilder` constructs faculties."""

    @classmethod
    def wire(cls, mind: Any) -> None:
        mind.preference = PreferenceAdapter(
            spatial_preference=mind.spatial_preference,
            causal_preference=mind.causal_preference,
            hawkes=mind.hawkes,
            pomdp=mind.pomdp,
            causal_pomdp=mind.causal_pomdp,
            preference_persistence=mind.preference_persistence,
        )
        mind.preference.sync_to_pomdp()

        mind.algebra = AlgebraicMemoryAdapter(
            vsa=mind.vsa,
            hopfield_memory=mind.hopfield_memory,
            ontology=mind.ontology,
        )
        mind.claims = ClaimRefiner(
            vsa=mind.vsa,
            memory=mind.memory,
            hopfield_memory=mind.hopfield_memory,
            algebra=mind.algebra,
        )
        mind.macros = MacroAdapter(
            journal=mind.journal,
            macro_registry=mind.macro_registry,
            chunking_compiler=mind.chunking_compiler,
        )
        mind.graft_frame = FrameGraftProjection(mind)
        mind.native_tools = NativeToolManager(
            tool_registry=mind.tool_registry,
            scm=mind.scm,
            workspace=mind.workspace,
            event_bus=mind.event_bus,
            slot=mind.tool_foraging,
            unified_agent=mind.unified_agent,
            native_tool_conformal=mind.native_tool_conformal,
            session=mind.session,
        )
        mind.tool_registry.attach_to_scm(
            mind.scm,
            topology_lock=mind.session.cognitive_state_lock,
            on_tool_drift=mind.native_tools.handle_drift,
        )

        mind.deferred_relations = DeferredRelationQueue(
            router=mind.router,
            event_bus=mind.event_bus,
            hawkes=mind.hawkes,
            claims=mind.claims,
            substrate=mind,
            session=mind.session,
        )
        mind.comprehension = ComprehensionPipeline(mind)
        mind.deferred_relations.bind_comprehension(mind.comprehension)

        mind.chat = ChatOrchestrator(mind)
        mind.speaker = PlanSpeaker(mind)
        mind.inspector = SubstrateInspector(mind)
        mind.workers = WorkerSupervisor(mind)
