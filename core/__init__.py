__version__ = "0.5.0-llama-bench"

import os
from typing import Any

from .logging_setup import configure_lab_logging

from .active_inference import (
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
from .broca import (
    BrocaMind,
    CognitiveBackgroundWorker,
    CognitiveFrame,
    DMNConfig,
    IntrinsicCue,
    PersistentSemanticMemory,
    TrainableBrocaGraft,
    WorkspaceJournal,
    cognitive_frame_from_episode_row,
)
from .causal import FiniteSCM, build_frontdoor_scm, build_simpson_scm
from .device_utils import pick_torch_device
from .grafts import ActiveInferenceTokenGraft, CoupledActiveInferenceTokenGraft, CausalEffectTokenGraft, FeatureVectorGraft, KVMemoryGraft
from .hf_tokenizer_compat import HuggingFaceBrocaTokenizer
from .llama_broca_host import LlamaBrocaHost, load_llama_broca_host
from .memory import SQLiteActivationMemory
from .substrate_graph import EpisodeAssociationGraph, merge_epistemic_evidence_dict
from .tokenizer import SPEECH_BRIDGE_PREFIX, speech_seed_ids, utterance_words
from .continuous_frame import (
    COGNITIVE_FRAME_DIM,
    BROCA_FEATURE_DIM,
    FrozenSubwordProjector,
    frozen_subword_projector_from_model,
    pack_broca_features,
    pack_cognitive_frame,
    semantic_subword_sketch,
    stable_sketch,
)
from .vsa import bind, unbind, bundle, permute, hypervector, cleanup

# Optional / heavier subsystems: imported on first attribute access via __getattr__ (PEP 562).

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "ChunkingDetectionConfig": (".chunking", "ChunkingDetectionConfig"),
    "CompiledMacro": (".chunking", "CompiledMacro"),
    "DMNChunkingCompiler": (".chunking", "DMNChunkingCompiler"),
    "MacroChunkRegistry": (".chunking", "MacroChunkRegistry"),
    "macro_frame_features": (".chunking", "macro_frame_features"),
    "NativeTool": (".native_tools", "NativeTool"),
    "NativeToolRegistry": (".native_tools", "NativeToolRegistry"),
    "SandboxResult": (".native_tools", "SandboxResult"),
    "ToolSandbox": (".native_tools", "ToolSandbox"),
    "tool_sandbox_from_env": (".native_tools", "tool_sandbox_from_env"),
    "ToolSynthesisError": (".native_tools", "ToolSynthesisError"),
    "ACTIVATION_MODE_KIND": (".dynamic_grafts", "ACTIVATION_MODE_KIND"),
    "CapturedActivationMode": (".dynamic_grafts", "CapturedActivationMode"),
    "DynamicGraftSynthesizer": (".dynamic_grafts", "DynamicGraftSynthesizer"),
    "capture_activation_mode": (".dynamic_grafts", "capture_activation_mode"),
    "CausalConstraint": (".top_down_control", "CausalConstraint"),
    "CausalConstraintGraft": (".top_down_control", "CausalConstraintGraft"),
    "EpistemicInterruptionMonitor": (".top_down_control", "EpistemicInterruptionMonitor"),
    "EpistemicInterruptionResult": (".top_down_control", "EpistemicInterruptionResult"),
    "HypothesisAttempt": (".top_down_control", "HypothesisAttempt"),
    "HypothesisMaskingGraft": (".top_down_control", "HypothesisMaskingGraft"),
    "HypothesisSearchResult": (".top_down_control", "HypothesisSearchResult"),
    "HypothesisVerdict": (".top_down_control", "HypothesisVerdict"),
    "InterruptionEvent": (".top_down_control", "InterruptionEvent"),
    "InterruptionVerdict": (".top_down_control", "InterruptionVerdict"),
    "IterativeHypothesisSearch": (".top_down_control", "IterativeHypothesisSearch"),
    "ModalityShiftGraft": (".top_down_control", "ModalityShiftGraft"),
    "VSACodebook": (".vsa", "VSACodebook"),
    "vsa_cosine": (".vsa", "cosine"),
    "HopfieldAssociativeMemory": (".hopfield", "HopfieldAssociativeMemory"),
    "hopfield_update": (".hopfield", "hopfield_update"),
    "derived_inverse_temperature": (".hopfield", "derived_inverse_temperature"),
    "VisionEncoder": (".vision", "VisionEncoder"),
    "ConformalPredictor": (".conformal", "ConformalPredictor"),
    "ConformalSet": (".conformal", "ConformalSet"),
    "PersistentConformalCalibration": (".conformal", "PersistentConformalCalibration"),
    "empirical_coverage": (".conformal", "empirical_coverage"),
    "MultivariateHawkesProcess": (".hawkes", "MultivariateHawkesProcess"),
    "PersistentHawkes": (".hawkes", "PersistentHawkes"),
    "fit_excitation_em": (".hawkes", "fit_excitation_em"),
    "GraftMotorTrainer": (".motor_learning", "GraftMotorTrainer"),
    "MotorLearningConfig": (".motor_learning", "MotorLearningConfig"),
    "DirichletPreference": (".preference_learning", "DirichletPreference"),
    "PersistentPreference": (".preference_learning", "PersistentPreference"),
    "feedback_polarity_from_text": (".preference_learning", "feedback_polarity_from_text"),
    "OntologicalRegistry": (".ontological_expansion", "OntologicalRegistry"),
    "PersistentOntologicalRegistry": (".ontological_expansion", "PersistentOntologicalRegistry"),
    "gram_schmidt_orthogonalize": (".ontological_expansion", "gram_schmidt_orthogonalize"),
    "pc_algorithm": (".causal_discovery", "pc_algorithm"),
    "build_scm_from_skeleton": (".causal_discovery", "build_scm_from_skeleton"),
    "DiscoveredGraph": (".causal_discovery", "DiscoveredGraph"),
    "local_predicate_cluster": (".causal_discovery", "local_predicate_cluster"),
    "project_rows_to_variables": (".causal_discovery", "project_rows_to_variables"),
    "DockerToolSandbox": (".docker_sandbox", "DockerToolSandbox"),
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
    "BrocaMind",
    "CognitiveBackgroundWorker",
    "CognitiveFrame",
    "DMNConfig",
    "IntrinsicCue",
    "PersistentSemanticMemory",
    "WorkspaceJournal",
    "TrainableBrocaGraft",
    "cognitive_frame_from_episode_row",
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
    "COGNITIVE_FRAME_DIM",
    "BROCA_FEATURE_DIM",
    "FrozenSubwordProjector",
    "frozen_subword_projector_from_model",
    "pack_cognitive_frame",
    "pack_broca_features",
    "semantic_subword_sketch",
    "stable_sketch",
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
    "project_rows_to_variables",
    "DockerToolSandbox",
]

_auto_log = str(os.environ.get("AUTO_CONFIGURE_LAB_LOGGING", "")).strip().lower()
if _auto_log in {"1", "true"}:
    configure_lab_logging()
