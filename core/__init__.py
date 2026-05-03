"""Mosaic public API.

The package root is intentionally a thin lazy-export surface.  Importing
``core`` no longer imports the live substrate, SQL-backed memory, host models,
or background workers; those concerns are resolved only when their public name
is requested.
"""

from __future__ import annotations

from typing import Any

from .infra.lazy_exports import LazyExportRegistry

__version__ = "0.6.1-mosaic"

_EXPORTS: dict[str, tuple[str, str]] = {
    "configure_lab_logging": (".infra.logging_setup", "configure_lab_logging"),
    "ActiveInferenceAgent": (".agent.active_inference", "ActiveInferenceAgent"),
    "CategoricalPOMDP": (".agent.active_inference", "CategoricalPOMDP"),
    "CoupledDecision": (".agent.active_inference", "CoupledDecision"),
    "CoupledEFEAgent": (".agent.active_inference", "CoupledEFEAgent"),
    "ToolForagingAgent": (".agent.active_inference", "ToolForagingAgent"),
    "build_causal_epistemic_pomdp": (".agent.active_inference", "build_causal_epistemic_pomdp"),
    "build_tiger_pomdp": (".agent.active_inference", "build_tiger_pomdp"),
    "build_tool_foraging_pomdp": (".agent.active_inference", "build_tool_foraging_pomdp"),
    "derived_listen_channel_reliability": (".agent.active_inference", "derived_listen_channel_reliability"),
    "extend_pomdp_with_synthesize_tool": (".agent.active_inference", "extend_pomdp_with_synthesize_tool"),
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
    "SubstrateController": (".substrate.controller", "SubstrateController"),
    "CognitiveBackgroundWorker": (".dmn", "CognitiveBackgroundWorker"),
    "CognitiveFrame": (".frame", "CognitiveFrame"),
    "DMNConfig": (".dmn", "DMNConfig"),
    "IntrinsicCue": (".workspace", "IntrinsicCue"),
    "SymbolicMemory": (".memory", "SymbolicMemory"),
    "WorkspaceJournal": (".memory", "WorkspaceJournal"),
    "TrainableFeatureGraft": (".grafts", "TrainableFeatureGraft"),
    "EpisodeAssociationGraph": (".substrate.graph", "EpisodeAssociationGraph"),
    "merge_epistemic_evidence_dict": (".substrate.graph", "merge_epistemic_evidence_dict"),
    "FiniteSCM": (".causal", "FiniteSCM"),
    "build_frontdoor_scm": (".causal", "build_frontdoor_scm"),
    "build_simpson_scm": (".causal", "build_simpson_scm"),
    "KVMemoryGraft": (".grafting.grafts", "KVMemoryGraft"),
    "ActiveInferenceTokenGraft": (".grafting.grafts", "ActiveInferenceTokenGraft"),
    "CoupledActiveInferenceTokenGraft": (".grafting.grafts", "CoupledActiveInferenceTokenGraft"),
    "CausalEffectTokenGraft": (".grafting.grafts", "CausalEffectTokenGraft"),
    "FeatureVectorGraft": (".grafting.grafts", "FeatureVectorGraft"),
    "HuggingFaceBrocaTokenizer": (".host.hf_tokenizer_compat", "HuggingFaceBrocaTokenizer"),
    "SQLiteActivationMemory": (".memory", "SQLiteActivationMemory"),
    "LlamaBrocaHost": (".host.llama_broca_host", "LlamaBrocaHost"),
    "load_llama_broca_host": (".host.llama_broca_host", "load_llama_broca_host"),
    "pick_torch_device": (".system.device", "pick_torch_device"),
    "SPEECH_BRIDGE_PREFIX": (".host.tokenizer", "SPEECH_BRIDGE_PREFIX"),
    "speech_seed_ids": (".host.tokenizer", "speech_seed_ids"),
    "utterance_words": (".host.tokenizer", "utterance_words"),
    "EmbeddingProjector": (".frame", "EmbeddingProjector"),
    "FrameDimensions": (".frame", "FrameDimensions"),
    "FramePacker": (".frame", "FramePacker"),
    "HypervectorProjector": (".frame", "HypervectorProjector"),
    "NumericTail": (".frame", "NumericTail"),
    "SubwordProjector": (".frame", "SubwordProjector"),
    "VSACodebook": (".symbolic.vsa", "VSACodebook"),
    "bind": (".symbolic.vsa", "bind"),
    "unbind": (".symbolic.vsa", "unbind"),
    "bundle": (".symbolic.vsa", "bundle"),
    "permute": (".symbolic.vsa", "permute"),
    "hypervector": (".symbolic.vsa", "hypervector"),
    "vsa_cosine": (".symbolic.vsa", "cosine"),
    "cleanup": (".symbolic.vsa", "cleanup"),
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
    "fit_excitation_em": (".temporal.hawkes_em", "fit_excitation_em"),
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

_registry = LazyExportRegistry(package=__package__ or __name__, exports=_EXPORTS)
__all__ = _registry.names()


def __getattr__(name: str) -> Any:
    return _registry.resolve(globals(), name)


def __dir__() -> list[str]:
    return _registry.dir_entries(globals())


_registry.auto_configure_logging(__import__(__name__), env_var="AUTO_CONFIGURE_LAB_LOGGING")
