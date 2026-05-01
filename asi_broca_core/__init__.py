__version__ = "0.5.0-llama-bench"

from typing import Any

from .logging_setup import configure_lab_logging

from .active_inference import (
    ActiveInferenceAgent,
    CategoricalPOMDP,
    CoupledDecision,
    CoupledEFEAgent,
    build_causal_epistemic_pomdp,
    build_tiger_pomdp,
    derived_listen_channel_reliability,
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
    FrozenSubwordProjector,
    frozen_subword_projector_from_model,
    pack_cognitive_frame,
    semantic_subword_sketch,
    stable_sketch,
)
from .vsa import bind, unbind, bundle, permute, hypervector, cleanup

# Heavy / optional: load on first attribute access (PEP 562).
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
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
}


def __getattr__(name: str) -> Any:
    spec = _LAZY_EXPORTS.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = spec
    import importlib

    mod = importlib.import_module(module_name, __package__)
    val = getattr(mod, attr)
    if name == "vsa_cosine":
        globals()["vsa_cosine"] = val
    else:
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
    "build_causal_epistemic_pomdp",
    "build_tiger_pomdp",
    "derived_listen_channel_reliability",
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
    "FrozenSubwordProjector",
    "frozen_subword_projector_from_model",
    "pack_cognitive_frame",
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
]

configure_lab_logging()
