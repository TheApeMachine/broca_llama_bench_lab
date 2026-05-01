__version__ = "0.5.0-llama-bench"

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
from .vsa import VSACodebook, bind, unbind, bundle, permute, hypervector, cosine as vsa_cosine, cleanup
from .hopfield import HopfieldAssociativeMemory, hopfield_update, derived_inverse_temperature
from .vision import VisionEncoder
from .conformal import ConformalPredictor, ConformalSet, PersistentConformalCalibration, empirical_coverage
from .hawkes import MultivariateHawkesProcess, PersistentHawkes, fit_excitation_em
from .motor_learning import GraftMotorTrainer, MotorLearningConfig
from .preference_learning import DirichletPreference, PersistentPreference, feedback_polarity_from_text
from .ontological_expansion import OntologicalRegistry, PersistentOntologicalRegistry, gram_schmidt_orthogonalize
from .causal_discovery import pc_algorithm, build_scm_from_skeleton, DiscoveredGraph

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
