"""SubstrateBuilder — lifts the substrate's 25-faculty construction out of the controller.

The previous controller had a 170-line ``__init__`` that built a host, three
graft instances, a multimodal perception pipeline, a workspace, six
perception encoders, an intent gate, a router, four POMDP / active inference
agents, an SCM, three SQLite-backed persistence layers, two Dirichlet
preference stores, an ontology registry, a Hopfield memory, a VSA codebook,
a motor trainer, a macro registry, a native-tool registry, an activation-
memory store, a dynamic-graft synthesizer, and a tool-foraging agent —
all inline in the controller class.

This builder owns that construction. The controller's ``__init__`` reduces
to a single ``SubstrateBuilder.populate(self, …)`` call.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from pathlib import Path
from typing import Any

from ..agent.active_inference import (
    ActiveInferenceAgent,
    CoupledEFEAgent,
    ToolForagingAgent,
    build_causal_epistemic_pomdp,
    build_tiger_pomdp,
)
from ..calibration.conformal import ConformalPredictor, PersistentConformalCalibration
from ..causal import build_simpson_scm
from ..cognition.affect_trace import PersistentAffectTrace
from ..cognition.intent_gate import IntentGate
from ..cognition.semantic_cascade import SemanticCascade
from ..cognition.encoder_relation_extractor import EncoderRelationExtractor
from ..cognition.multimodal_perception import MultimodalPerceptionPipeline
from ..comprehension import CognitiveRouter, DeferredRelationIngest
from ..dmn import CognitiveBackgroundWorker
from ..encoders.affect import AffectEncoder
from ..encoders.classification import SemanticClassificationEncoder
from ..encoders.extraction import ExtractionEncoder
from ..frame import EmbeddingProjector, FrameDimensions, FramePacker
from ..grafting.dynamic_grafts import DynamicGraftSynthesizer
from ..grafts import LexicalPlanGraft, SubstrateLogitBiasGraft, TrainableFeatureGraft
from ..host.llama_broca_host import LlamaBrocaHost
from ..host.hf_tokenizer_compat import HuggingFaceBrocaTokenizer
from ..idletime.chunking import DMNChunkingCompiler, MacroChunkRegistry
from ..idletime.ontological_expansion import PersistentOntologicalRegistry
from ..learning.motor_learning import GraftMotorTrainer
from ..learning.preference_learning import DirichletPreference, PersistentPreference
from ..memory import (
    HopfieldAssociativeMemory,
    SQLiteActivationMemory,
    SymbolicMemory,
    WorkspaceJournal,
)
from ..natives.native_tools import NativeTool, NativeToolRegistry
from ..substrate.graph import EpisodeAssociationGraph
from ..substrate.runtime import default_substrate_sqlite_path, ensure_parent_dir
from ..symbolic.vsa import VSACodebook
from ..system.device import pick_torch_device
from ..temporal.hawkes import MultivariateHawkesProcess, PersistentHawkes
from ..workspace import BaseWorkspace, GlobalWorkspace, WorkspaceBuilder
from .constants import DEFAULT_CHAT_MODEL_ID


logger = logging.getLogger(__name__)


class SubstrateBuilder:
    """Constructs every faculty the controller needs and assigns to ``mind``."""

    @classmethod
    def populate(
        cls,
        mind: Any,
        *,
        seed: int = 0,
        db_path: str | Path | None = None,
        namespace: str = "main",
        llama_model_id: str | None = None,
        device: Any = None,
        hf_token: Any = None,
        lexical_target_snr: float | None = None,
        preload_host_tokenizer: tuple[LlamaBrocaHost, HuggingFaceBrocaTokenizer] | None = None,
    ) -> None:
        from ..grafts.lexical_plan import LexicalPlanGraft  # noqa: F811  (avoid circular at import time)

        mind.seed = seed
        rp = Path(db_path) if db_path is not None else default_substrate_sqlite_path()
        ensure_parent_dir(rp)
        mid = llama_model_id or DEFAULT_CHAT_MODEL_ID

        cls._init_state(mind, rp, namespace, mid)
        cls._build_persistence_layer(mind, rp, namespace)
        cls._build_host(mind, mid, device, hf_token, preload_host_tokenizer)
        cls._build_grafts(mind, lexical_target_snr)
        cls._build_perception(mind, device)
        cls._build_comprehension(mind)
        cls._build_reasoning(mind, rp, namespace, seed)
        cls._build_motor(mind)
        cls._build_chunking(mind, rp, namespace)
        cls._build_native_tools(mind, rp, namespace)
        cls._build_dynamic_grafts(mind, rp, namespace)
        cls._build_tool_foraging(mind)
        cls._build_workspace_handle(mind)

    # -- per-concern construction helpers -------------------------------------

    @classmethod
    def _build_persistence_layer(cls, mind: Any, rp: Path, namespace: str) -> None:
        mind.memory = SymbolicMemory(rp, namespace=namespace)
        mind.journal = WorkspaceJournal(rp, shared_memory=mind.memory)
        mind.episode_graph = EpisodeAssociationGraph(rp)

    @classmethod
    def _build_host(
        cls,
        mind: Any,
        model_id: str,
        device: Any,
        hf_token: Any,
        preload: tuple[Any, Any] | None,
    ) -> None:
        if preload is None:
            import torch

            from . import substrate as substrate_mod

            resolved_device = (
                device if isinstance(device, torch.device) else pick_torch_device(device)
            )
            mind.host, mind.tokenizer = substrate_mod.load_llama_broca_host(
                model_id, device=resolved_device, token=hf_token
            )
        else:
            mind.host, mind.tokenizer = preload
        mind.text_encoder = EmbeddingProjector.from_host(mind.host, mind.tokenizer)
        mind.frame_packer = FramePacker(mind.text_encoder)

    @classmethod
    def _build_grafts(cls, mind: Any, lexical_target_snr: float | None) -> None:
        from ..grafting.grafts import DEFAULT_GRAFT_TARGET_SNR

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

    @classmethod
    def _build_perception(cls, mind: Any, device: Any) -> None:
        import torch

        host_param = getattr(mind, "_host_param", None)
        encoder_device = (
            host_param.device
            if host_param is not None
            else device
            if isinstance(device, torch.device)
            else pick_torch_device(device)
        )
        mind.multimodal_perception = MultimodalPerceptionPipeline(device=encoder_device)
        mind.workspace = GlobalWorkspace()

    @classmethod
    def _build_comprehension(cls, mind: Any) -> None:
        mind.extraction_encoder = ExtractionEncoder()
        mind.classification_encoder = SemanticClassificationEncoder()
        mind.semantic_cascade = SemanticCascade(classifier=mind.classification_encoder)
        mind.affect_encoder = AffectEncoder()
        mind.intent_gate = IntentGate(mind.semantic_cascade)
        mind.router = CognitiveRouter(
            extractor=EncoderRelationExtractor(
                intent_gate=mind.intent_gate,
                extraction=mind.extraction_encoder,
            )
        )

    @classmethod
    def _build_reasoning(cls, mind: Any, rp: Path, namespace: str, seed: int) -> None:
        d_model = int(getattr(mind.host.cfg, "d_model", 96))
        mind.pomdp = build_tiger_pomdp()
        mind.active_agent = ActiveInferenceAgent(mind.pomdp, horizon=1, learn=False)
        mind.scm = build_simpson_scm()
        mind.causal_pomdp = build_causal_epistemic_pomdp(mind.scm)
        mind.causal_agent = ActiveInferenceAgent(mind.causal_pomdp, horizon=1, learn=False)
        mind.unified_agent = CoupledEFEAgent(mind.active_agent, mind.causal_agent)
        mind.affect_trace = PersistentAffectTrace(rp, namespace=f"{namespace}__affect")
        mind.vsa = VSACodebook(dim=10_000, base_seed=int(seed))
        mind.hopfield_memory = HopfieldAssociativeMemory(d_model=d_model, max_items=65_536)
        mind.conformal_calibration = PersistentConformalCalibration(
            rp, namespace=f"{namespace}__conformal"
        )
        mind.relation_conformal = ConformalPredictor(alpha=0.1, method="lac", min_calibration=8)
        mind.conformal_calibration.hydrate(mind.relation_conformal, channel="relation_extraction")
        mind.native_tool_conformal = ConformalPredictor(alpha=0.1, method="lac", min_calibration=8)
        mind.conformal_calibration.hydrate(mind.native_tool_conformal, channel="native_tool_output")
        mind.hawkes_persistence = PersistentHawkes(rp, namespace=f"{namespace}__hawkes")
        loaded = mind.hawkes_persistence.load()
        mind.hawkes = (
            loaded if loaded is not None else MultivariateHawkesProcess(beta=0.5, baseline=0.05)
        )
        mind.preference_persistence = PersistentPreference(rp, namespace=f"{namespace}__pref")
        mind.spatial_preference = mind.preference_persistence.load("spatial") or DirichletPreference(
            len(mind.pomdp.observation_names),
            initial_C=list(mind.pomdp.C),
            prior_strength=4.0,
        )
        mind.causal_preference = mind.preference_persistence.load("causal") or DirichletPreference(
            len(mind.causal_pomdp.observation_names),
            initial_C=list(mind.causal_pomdp.C),
            prior_strength=4.0,
        )
        mind._sync_preference_to_pomdp()
        mind.ontology_persistence = PersistentOntologicalRegistry(
            rp, namespace=f"{namespace}__ontology"
        )
        mind.ontology = mind.ontology_persistence.load(
            dim=FrameDimensions.SKETCH_DIM, frequency_threshold=8
        )
        mind.discovered_scm = None
        mind.motor_replay = []

    @classmethod
    def _build_motor(cls, mind: Any) -> None:
        mind.motor_trainer = GraftMotorTrainer(mind.host, mind.tokenizer, (mind.feature_graft,))

    @classmethod
    def _build_chunking(cls, mind: Any, rp: Path, namespace: str) -> None:
        mind.macro_registry = MacroChunkRegistry(rp, namespace=f"{namespace}__macros")
        mind.chunking_compiler = DMNChunkingCompiler(mind, registry=mind.macro_registry)

    @classmethod
    def _build_native_tools(cls, mind: Any, rp: Path, namespace: str) -> None:
        mind.tool_registry = NativeToolRegistry(rp, namespace=f"{namespace}__tools")
        try:
            mind.tool_registry.attach_to_scm(
                mind.scm,
                topology_lock=mind._cognitive_state_lock,
                on_tool_drift=mind._handle_native_tool_drift,
            )
        except Exception:
            logger.exception("SubstrateBuilder: initial tool attachment failed")

    @classmethod
    def _build_dynamic_grafts(cls, mind: Any, rp: Path, namespace: str) -> None:
        mind.activation_memory = SQLiteActivationMemory(
            rp, default_namespace=f"{namespace}__activation"
        )
        mind.dynamic_graft_synth = DynamicGraftSynthesizer(
            mind.activation_memory, namespace=f"{namespace}__activation"
        )

    @classmethod
    def _build_tool_foraging(cls, mind: Any) -> None:
        mind.tool_foraging_agent = ToolForagingAgent.build(
            n_existing_tools=mind.tool_registry.count(),
            insufficient_prior=0.5,
        )

    @classmethod
    def _build_workspace_handle(cls, mind: Any) -> None:
        mind.event_bus: BaseWorkspace = WorkspaceBuilder().process_default()

    @classmethod
    def _init_state(cls, mind: Any, rp: Path, namespace: str, model_id: str) -> None:
        mind._last_intent = None
        mind._last_affect = None
        mind._last_user_affect_trace_id = None
        mind._last_journal_id = None
        mind._background_worker: CognitiveBackgroundWorker | None = None
        mind._self_improve_worker: Any | None = None
        mind._cognitive_state_lock = threading.RLock()
        mind._deferred_relation_jobs: deque[DeferredRelationIngest] = deque()
        mind._next_deferred_relation_job_id = 1
        mind._last_chat_meta = {}
        mind._db_path = rp
        mind._namespace = namespace
        mind._llama_model_id = model_id
