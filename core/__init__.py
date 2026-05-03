__version__ = "0.6.1-mosaic"

import os
from typing import Any

from .infra.logging_setup import configure_lab_logging

from .agent.active_inference import (
    ActiveInferenceAgent,
    CategoricalPOMDP,
    CoupledDecision,
    CoupledEFEAgent,
    ToolForagingAgent,
    build_causal_epistemic_pomdp,
    build_tiger_pomdp,
    build_tool_foraging_pomdp,
    derived_listen_channel_reliability,
    extend_pomdp_with_synthesize_tool,
)
from .substrate.controller import SubstrateController
from .dmn import CognitiveBackgroundWorker, DMNConfig
from .grafts import TrainableFeatureGraft
from .memory import SymbolicMemory, WorkspaceJournal
from .workspace import IntrinsicCue
from .frame import CognitiveFrame
from .causal import FiniteSCM, build_frontdoor_scm, build_simpson_scm
from .system.device import pick_torch_device
from .grafting.grafts import (
    ActiveInferenceTokenGraft,
    CoupledActiveInferenceTokenGraft,
    CausalEffectTokenGraft,
    FeatureVectorGraft,
    KVMemoryGraft,
)
from .host.hf_tokenizer_compat import HuggingFaceBrocaTokenizer
from .host.llama_broca_host import LlamaBrocaHost, load_llama_broca_host
from .memory import SQLiteActivationMemory
from .substrate.graph import EpisodeAssociationGraph, merge_epistemic_evidence_dict
from .host.tokenizer import SPEECH_BRIDGE_PREFIX, speech_seed_ids, utterance_words
from .frame import (
    EmbeddingProjector,
    FrameDimensions,
    FramePacker,
    HypervectorProjector,
    NumericTail,
    SubwordProjector,
)
from .symbolic.vsa import bind, unbind, bundle, permute, hypervector, cleanup

# Optional / heavier subsystems: imported on first attribute access via __getattr__ (PEP 562).

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "ChunkingDetectionConfig": (".idletime.chunking", "ChunkingDetectionConfig"),
    "CompiledMacro": (".idletime.chunking", "CompiledMacro"),
    "MacroChunkRegistry": (".idletime.chunking", "MacroChunkRegistry"),
    "DMNChunkingCompiler": (".idletime.chunking", "DMNChunkingCompiler"),
    "macro_frame_features": (".idletime.chunking", "macro_frame_features"),
    "NativeTool": (".natives.native_tools", "NativeTool"),
    "NativeToolRegistry": (".natives.native_tools", "NativeToolRegistry"),
    "SandboxResult": (".natives.native_tools", "SandboxResult"),
    "ToolSandbox": (".natives.native_tools", "ToolSandbox"),
    "tool_sandbox_from_env": (".natives.native_tools", "tool_sandbox_from_env"),
    "ToolSynthesisError": (".natives.native_tools", "ToolSynthesisError"),
    "ACTIVATION_MODE_KIND": (".grafting.dynamic_grafts", "ACTIVATION_MODE_KIND"),
    "CapturedActivationMode": (".grafting.dynamic_grafts", "CapturedActivationMode"),
    "DynamicGraftSynthesizer": (".grafting.dynamic_grafts", "DynamicGraftSynthesizer"),
    "capture_activation_mode": (".grafting.dynamic_grafts", "capture_activation_mode"),
    "CausalConstraint": (".cognition.top_down_control", "CausalConstraint"),
    "CausalConstraintGraft": (".cognition.top_down_control", "CausalConstraintGraft"),
    "EpistemicInterruptionMonitor": (".cognition.top_down_control", "EpistemicInterruptionMonitor"),
    "EpistemicInterruptionResult": (".cognition.top_down_control", "EpistemicInterruptionResult"),
    "HypothesisAttempt": (".cognition.top_down_control", "HypothesisAttempt"),
    "HypothesisMaskingGraft": (".cognition.top_down_control", "HypothesisMaskingGraft"),
    "HypothesisSearchResult": (".cognition.top_down_control", "HypothesisSearchResult"),
    "HypothesisVerdict": (".cognition.top_down_control", "HypothesisVerdict"),
    "InterruptionEvent": (".cognition.top_down_control", "InterruptionEvent"),
    "InterruptionVerdict": (".cognition.top_down_control", "InterruptionVerdict"),
    "IterativeHypothesisSearch": (".cognition.top_down_control", "IterativeHypothesisSearch"),
    "ModalityShiftGraft": (".cognition.top_down_control", "ModalityShiftGraft"),
    "VSACodebook": (".symbolic.vsa", "VSACodebook"),
    "vsa_cosine": (".symbolic.vsa", "cosine"),
    "HopfieldAssociativeMemory": (".memory.hopfield", "HopfieldAssociativeMemory"),
    "hopfield_update": (".memory.hopfield", "hopfield_update"),
    "derived_inverse_temperature": (".memory.hopfield", "derived_inverse_temperature"),
    "VisionEncoder": (".vision.vision", "VisionEncoder"),
    "ConformalPredictor": (".calibration.conformal", "ConformalPredictor"),
    "ConformalSet": (".calibration.conformal", "ConformalSet"),
    "OnlineConformalMartingale": (".calibration.conformal", "OnlineConformalMartingale"),
    "PersistentConformalCalibration": (".calibration.conformal", "PersistentConformalCalibration"),
    "empirical_coverage": (".calibration.conformal", "empirical_coverage"),
    "MultivariateHawkesProcess": (".temporal.hawkes", "MultivariateHawkesProcess"),
    "PersistentHawkes": (".temporal.hawkes", "PersistentHawkes"),
    "fit_excitation_em": (".temporal.hawkes", "fit_excitation_em"),
    "GraftMotorTrainer": (".learning.motor_learning", "GraftMotorTrainer"),
    "MotorLearningConfig": (".learning.motor_learning", "MotorLearningConfig"),
    "DirichletPreference": (".learning.preference_learning", "DirichletPreference"),
    "PersistentPreference": (".learning.preference_learning", "PersistentPreference"),
    "feedback_polarity_from_text": (".learning.preference_learning", "feedback_polarity_from_text"),
    "OntologicalRegistry": (".idletime.ontological_expansion", "OntologicalRegistry"),
    "PersistentOntologicalRegistry": (".idletime.ontological_expansion", "PersistentOntologicalRegistry"),
    "gram_schmidt_orthogonalize": (".idletime.ontological_expansion", "gram_schmidt_orthogonalize"),
    "pc_algorithm": (".causal.causal_discovery", "pc_algorithm"),
    "build_scm_from_skeleton": (".causal.causal_discovery", "build_scm_from_skeleton"),
    "DiscoveredGraph": (".causal.causal_discovery", "DiscoveredGraph"),
    "local_predicate_cluster": (".causal.causal_discovery", "local_predicate_cluster"),
    "orient_temporal_edges": (".causal.causal_discovery", "orient_temporal_edges"),
    "project_rows_to_variables": (".causal.causal_discovery", "project_rows_to_variables"),
    "TemporalCausalTraceBuilder": (".causal.temporal", "TemporalCausalTraceBuilder"),
    "DockerToolSandbox": (".system.sandbox", "DockerToolSandbox"),
    "SelfImproveConfig": (".workers.docker_self_improve_worker", "SelfImproveConfig"),
    "SelfImproveDockerWorker": (".workers.docker_self_improve_worker", "SelfImproveDockerWorker"),
}


def __getattr__(name: str) -> Any:
    spec = _LAZY_EXPORTS.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = spec
    import importlib

    mod = importlib.import_module(module_name, __package__)
    val = getattr(mod, attr)
    globals()[name] = val
    return val


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "configure_lab_logging",
    "ActiveInferenceAgent",
    "CategoricalPOMDP",
    "CoupledDecision",
    "CoupledEFEAgent",
    "ToolForagingAgent",
    "build_causal_epistemic_pomdp",
    "build_tiger_pomdp",
    "build_tool_foraging_pomdp",
    "extend_pomdp_with_synthesize_tool",
    "derived_listen_channel_reliability",
    "ChunkingDetectionConfig",
    "CompiledMacro",
    "DMNChunkingCompiler",
    "MacroChunkRegistry",
    "macro_frame_features",
    "NativeTool",
    "NativeToolRegistry",
    "SandboxResult",
    "ToolSandbox",
    "tool_sandbox_from_env",
    "ToolSynthesisError",
    "ACTIVATION_MODE_KIND",
    "CapturedActivationMode",
    "DynamicGraftSynthesizer",
    "capture_activation_mode",
    "CausalConstraint",
    "CausalConstraintGraft",
    "EpistemicInterruptionMonitor",
    "EpistemicInterruptionResult",
    "HypothesisAttempt",
    "HypothesisMaskingGraft",
    "HypothesisSearchResult",
    "HypothesisVerdict",
    "InterruptionEvent",
    "InterruptionVerdict",
    "IterativeHypothesisSearch",
    "ModalityShiftGraft",
    "SubstrateController",
    "CognitiveBackgroundWorker",
    "CognitiveFrame",
    "DMNConfig",
    "IntrinsicCue",
    "SymbolicMemory",
    "WorkspaceJournal",
    "TrainableFeatureGraft",
    "EpisodeAssociationGraph",
    "merge_epistemic_evidence_dict",
    "FiniteSCM",
    "build_frontdoor_scm",
    "build_simpson_scm",
    "KVMemoryGraft",
    "ActiveInferenceTokenGraft",
    "CoupledActiveInferenceTokenGraft",
    "CausalEffectTokenGraft",
    "FeatureVectorGraft",
    "HuggingFaceBrocaTokenizer",
    "SQLiteActivationMemory",
    "LlamaBrocaHost",
    "load_llama_broca_host",
    "pick_torch_device",
    "SPEECH_BRIDGE_PREFIX",
    "speech_seed_ids",
    "utterance_words",
    "EmbeddingProjector",
    "FrameDimensions",
    "FramePacker",
    "HypervectorProjector",
    "NumericTail",
    "SubwordProjector",
    "VSACodebook",
    "bind",
    "unbind",
    "bundle",
    "permute",
    "hypervector",
    "vsa_cosine",
    "cleanup",
    "HopfieldAssociativeMemory",
    "hopfield_update",
    "derived_inverse_temperature",
    "VisionEncoder",
    "ConformalPredictor",
    "ConformalSet",
    "OnlineConformalMartingale",
    "PersistentConformalCalibration",
    "empirical_coverage",
    "MultivariateHawkesProcess",
    "PersistentHawkes",
    "fit_excitation_em",
    "GraftMotorTrainer",
    "MotorLearningConfig",
    "DirichletPreference",
    "PersistentPreference",
    "feedback_polarity_from_text",
    "OntologicalRegistry",
    "PersistentOntologicalRegistry",
    "gram_schmidt_orthogonalize",
    "pc_algorithm",
    "build_scm_from_skeleton",
    "DiscoveredGraph",
    "local_predicate_cluster",
    "orient_temporal_edges",
    "project_rows_to_variables",
    "TemporalCausalTraceBuilder",
    "DockerToolSandbox",
    "SelfImproveConfig",
    "SelfImproveDockerWorker",
]

_auto_log = str(os.environ.get("AUTO_CONFIGURE_LAB_LOGGING", "")).strip().lower()
if _auto_log in {"1", "true"}:
    configure_lab_logging()
