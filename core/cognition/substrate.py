"""Cognitive substrate orchestration for a frozen Llama host.

`SymbolicMemory` is SQLite-backed factual storage (WAL, one shared
connection per instance, guarded by a lock for thread-safe reuse).
`GlobalWorkspace` blackboards frames and `IntrinsicCue` signals from language
and background workers. `CognitiveBackgroundWorker` / DMN phases run offline
consolidation and emit cues (tagged with `source="dmn"` where applicable).

`SubstrateController` wires `LlamaBrocaHost` to `BaseGraft` / lexical and logit grafts,
`DynamicGraftSynthesizer` modes (`DYNAMIC_GRAFT*` in ``dynamic_grafts``),
active inference + SCM faculties (`build_simpson_scm`, tools, Hawkes, conformal,
etc.), and routes utterances through `CognitiveRouter`. Grafts read
``extra_state`` (e.g. ``broca_features``, ``broca_logit_bias``) during
`LlamaBrocaHost.forward`; background threads must use workspace locks where the
host is shared.

**Public knobs (non-exhaustive):** `DEFAULT_CHAT_MODEL_ID`, `SEMANTIC_CONFIDENCE_FLOOR`,
`BELIEF_REVISION_LOG_ODDS_THRESHOLD`, `BELIEF_REVISION_MIN_CLAIMS`, plus the main
types `SubstrateController`, `SymbolicMemory`, `GlobalWorkspace`, `CognitiveFrame`,
`CognitiveRouter`, `IntrinsicCue`, `LexicalPlanGraft`, `TrainableFeatureGraft`.
"""

from __future__ import annotations

import json
import hashlib
import logging
import math
import os
import random
import sqlite3
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..agent.active_inference import (
    ActiveInferenceAgent,
    CoupledEFEAgent,
    ToolForagingAgent,
    build_causal_epistemic_pomdp,
    build_tiger_pomdp,
    entropy as belief_entropy,
)
from ..causal import build_simpson_scm
from ..idletime.chunking import (
    ChunkingDetectionConfig,
    CompiledMacro,
    DMNChunkingCompiler,
    MacroChunkRegistry,
    macro_frame_features,
)
from ..frame import (
    EmbeddingProjector,
    FrameDimensions,
    FramePacker,
    SubwordProjector,
    TextEncoder,
)
from ..system.device import pick_torch_device

_SUBWORD = SubwordProjector()
from ..grafting.grafts import (
    BaseGraft,
    DEFAULT_GRAFT_TARGET_SNR,
    snr_magnitude,
    state_confidence,
    state_inertia,
    state_target_snr_scale,
)
from ..host.hf_tokenizer_compat import HuggingFaceBrocaTokenizer
from ..substrate.runtime import default_substrate_sqlite_path, ensure_parent_dir
from ..host.llama_broca_host import LlamaBrocaHost, load_llama_broca_host
from .predictive_coding import lexical_surprise_gap
from ..substrate.graph import EpisodeAssociationGraph, merge_epistemic_evidence_dict
from ..host.tokenizer import speech_seed_ids, utterance_words
from ..symbolic.vsa import VSACodebook, bundle, cosine as vsa_cosine
from ..memory.hopfield import HopfieldAssociativeMemory
from ..calibration.conformal import ConformalPredictor, PersistentConformalCalibration
from ..temporal.hawkes import MultivariateHawkesProcess, PersistentHawkes, fit_excitation_em
from ..learning.preference_learning import DirichletPreference, PersistentPreference, feedback_polarity_from_text
from ..learning.motor_learning import GraftMotorTrainer
from ..idletime.ontological_expansion import OntologicalRegistry, PersistentOntologicalRegistry
from ..causal.causal_discovery import (
    build_scm_from_skeleton,
    local_predicate_cluster,
    orient_temporal_edges,
    pc_algorithm,
    project_rows_to_variables,
)
from ..causal.temporal import TemporalCausalTraceBuilder
from ..natives.native_tools import NativeTool, NativeToolRegistry, ToolSandbox, ToolSynthesisError
from ..grafting.dynamic_grafts import DynamicGraftSynthesizer, CapturedActivationMode, ACTIVATION_MODE_KIND
from ..workspace import BaseWorkspace, GlobalWorkspace, IntrinsicCue, WorkspaceBuilder
from ..memory import ClaimTrust, SQLiteActivationMemory, SymbolicMemory, WorkspaceJournal
from ..grafts import LexicalPlanGraft, SubstrateLogitBiasGraft, TrainableFeatureGraft
from ..dmn import CognitiveBackgroundWorker, DMNConfig
from ..encoders.classification import SemanticClassificationEncoder
from ..encoders.extraction import ExtractionEncoder
from ..encoders.affect import AffectEncoder, AffectState

from .constants import (
    DEFAULT_CHAT_MODEL_ID,
    SEMANTIC_CONFIDENCE_FLOOR,
    BELIEF_REVISION_LOG_ODDS_THRESHOLD,
    BELIEF_REVISION_MIN_CLAIMS,
)
from .intent_gate import IntentGate, UtteranceIntent
from .semantic_cascade import SemanticCascade
from .encoder_relation_extractor import EncoderRelationExtractor
from .derived_strength import DerivedStrength, StrengthInputs
from .multimodal_perception import MultimodalPerceptionPipeline
from .observation import CognitiveObservation
from .affect_trace import PersistentAffectTrace

logger = logging.getLogger(__name__)


from ..frame import CognitiveFrame, ParsedClaim, ParsedQuery
from ..comprehension import (
    ClaimPredictionGap,
    CognitiveRouter,
    DeferredRelationIngest,
    LexicalTokens,
    MemoryQueryParser,
    SCMTargetPicker,
    TextRelevance,
)


# Backwards-compat function aliases — substrate.py's controller still calls
# these; once Layer 7 dissolves the controller they go away with it.
def _word_tokens(toks):
    return LexicalTokens.words(toks)


def _lexical_tokens(value):
    return LexicalTokens.lexical(value)


def _frame_relevance(utterance, toks, frame, text_encoder):
    return TextRelevance.frame(utterance, toks, frame, text_encoder)



def _affect_evidence(affect: AffectState) -> dict[str, Any]:
    """Compact, JSON-friendly summary of an :class:`AffectState`.

    Stored on every frame so derived graft strength, preference learning,
    and intrinsic cues all consume the same numbers — there is no second
    affect call that could disagree with this one.
    """

    return {
        "dominant_emotion": str(affect.dominant_emotion),
        "dominant_score": float(affect.dominant_score),
        "confidences": [
            {"label": item.label, "score": float(item.score), "signal": item.signal}
            for item in affect.confidences
        ],
        "valence": float(affect.valence),
        "arousal": float(affect.arousal),
        "entropy": float(affect.entropy),
        "certainty": float(affect.certainty),
        "preference_signal": str(affect.preference_signal),
        "preference_strength": float(affect.preference_strength),
        "cognitive_states": dict(affect.cognitive_states),
    }


def affect_certainty(affect: AffectState | None) -> float:
    """Affect-driven certainty in ``[0, 1]`` for derived graft strength.

    Uses normalized entropy of the full GoEmotions vector when available:
    a peaked affective response means the user's emotional signal is
    unambiguous; a flat distribution means the user is hard to read and the
    substrate should nudge, not hammer.
    """

    if affect is None:
        return 1.0
    if affect.confidences:
        return max(0.0, min(1.0, float(affect.certainty)))
    return max(0.0, min(1.0, float(affect.dominant_score)))


def default_lexical_target_snr(model: nn.Module) -> float:
    """Target SNR for the lexical Broca graft.

    Geometry-independent: the graft injects ``target_snr`` × host RMS energy
    along the planned token direction, so the same fraction works regardless of
    ``d_model``. The argument is accepted for API compatibility with callers
    that still want to inspect the host's configuration.
    """

    _ = model
    return DEFAULT_GRAFT_TARGET_SNR


def _motor_replay_messages_plan_forced(frame: CognitiveFrame, plan_words: Sequence[str]) -> list[dict[str, str]]:
    """One user turn synthesizing lexical-plan context for REM chat-template supervision."""

    chunks = (
        f"intent={frame.intent}",
        f"subject={frame.subject or ''}",
        f"answer={frame.answer or ''}",
        f"plan={' '.join(plan_words)}",
    )
    return [{"role": "user", "content": " | ".join(chunks)}]


from ..generation import PlanForcedGenerator, TokenDecoder

# Substrate.py's controller still calls these names internally; they collapse
# into the canonical classes and disappear with the controller in Layer 7.
def decode_generation(tokenizer, generated):
    return TokenDecoder.decode(tokenizer, generated)


def generate_from_plan(model, tokenizer, plan_tokens, *, prefix=None, max_new_tokens=None, broca_features=None):
    return PlanForcedGenerator.generate(
        model,
        tokenizer,
        plan_tokens,
        prefix=prefix,
        max_new_tokens=max_new_tokens,
        broca_features=broca_features,
    )



class SubstrateController:
    """Cognitive substrate with the language model demoted to speech interface."""

    host: LlamaBrocaHost
    tokenizer: HuggingFaceBrocaTokenizer

    def __init__(
        self,
        *,
        seed: int = 0,
        db_path: str | Path | None = None,
        namespace: str = "main",
        llama_model_id: str | None = None,
        device: torch.device | str | None = None,
        hf_token: str | bool | None = None,
        lexical_target_snr: float | None = None,
        preload_host_tokenizer: tuple[LlamaBrocaHost, HuggingFaceBrocaTokenizer] | None = None,
    ):
        self.seed = seed
        rp = Path(db_path) if db_path is not None else default_substrate_sqlite_path()
        ensure_parent_dir(rp)
        mid = llama_model_id or DEFAULT_CHAT_MODEL_ID
        self.memory = SymbolicMemory(rp, namespace=namespace)
        self.journal = WorkspaceJournal(rp, shared_memory=self.memory)
        self.episode_graph = EpisodeAssociationGraph(rp)
        self._last_journal_id: int | None = None
        if preload_host_tokenizer is None:
            resolved_device = device if isinstance(device, torch.device) else pick_torch_device(device)
            self.host, self.tokenizer = load_llama_broca_host(mid, device=resolved_device, token=hf_token)
        else:
            self.host, self.tokenizer = preload_host_tokenizer
        self.text_encoder = EmbeddingProjector.from_host(self.host, self.tokenizer)
        self.frame_packer = FramePacker(self.text_encoder)
        snr = lexical_target_snr if lexical_target_snr is not None else default_lexical_target_snr(self.host)
        self.lexical_graft = LexicalPlanGraft(target_snr=snr)
        self.host.add_graft("final_hidden", self.lexical_graft)
        self.feature_graft = TrainableFeatureGraft(
            FrameDimensions.broca_feature_dim(),
            int(getattr(self.host.cfg, "d_model", 96)),
            target_snr=snr,
        )
        host_param = None
        params = getattr(self.host, "parameters", None)
        if callable(params):
            host_param = next(iter(params()), None)
            if host_param is not None:
                self.feature_graft.to(host_param.device)
        self.host.add_graft("final_hidden", self.feature_graft)
        self.logit_bias_graft = SubstrateLogitBiasGraft()
        self.host.add_graft("logits", self.logit_bias_graft)
        encoder_device = (
            host_param.device
            if host_param is not None
            else device
            if isinstance(device, torch.device)
            else pick_torch_device(device)
        )
        self.multimodal_perception = MultimodalPerceptionPipeline(device=encoder_device)
        self.workspace = GlobalWorkspace()
        self.extraction_encoder = ExtractionEncoder()
        self.classification_encoder = SemanticClassificationEncoder()
        self.semantic_cascade = SemanticCascade(
            classifier=self.classification_encoder,
        )
        self.affect_encoder = AffectEncoder()
        self.affect_trace = PersistentAffectTrace(rp, namespace=f"{namespace}__affect")
        self.intent_gate = IntentGate(self.semantic_cascade)
        self._last_intent: UtteranceIntent | None = None
        self._last_affect: AffectState | None = None
        self._last_user_affect_trace_id: int | None = None
        self.router = CognitiveRouter(
            extractor=EncoderRelationExtractor(
                intent_gate=self.intent_gate,
                extraction=self.extraction_encoder,
            )
        )
        self.pomdp = build_tiger_pomdp()
        self.active_agent = ActiveInferenceAgent(self.pomdp, horizon=1, learn=False)
        self.scm = build_simpson_scm()
        self.causal_pomdp = build_causal_epistemic_pomdp(self.scm)
        self.causal_agent = ActiveInferenceAgent(self.causal_pomdp, horizon=1, learn=False)
        self.unified_agent = CoupledEFEAgent(self.active_agent, self.causal_agent)
        self._background_worker: CognitiveBackgroundWorker | None = None
        self._self_improve_worker: Any | None = None
        self._cognitive_state_lock = threading.RLock()
        self._deferred_relation_jobs: deque[DeferredRelationIngest] = deque()
        self._next_deferred_relation_job_id = 1

        # New substrates ----------------------------------------------------
        d_model = int(getattr(self.host.cfg, "d_model", 96))
        self.vsa = VSACodebook(dim=10_000, base_seed=int(seed))
        self.hopfield_memory = HopfieldAssociativeMemory(d_model=d_model, max_items=65_536)
        self.conformal_calibration = PersistentConformalCalibration(rp, namespace=f"{namespace}__conformal")
        self.relation_conformal = ConformalPredictor(alpha=0.1, method="lac", min_calibration=8)
        self.conformal_calibration.hydrate(self.relation_conformal, channel="relation_extraction")
        self.native_tool_conformal = ConformalPredictor(alpha=0.1, method="lac", min_calibration=8)
        self.conformal_calibration.hydrate(self.native_tool_conformal, channel="native_tool_output")
        # Hawkes channels are populated lazily by ``observe_event`` so the
        # excitation matrix grows with the user's vocabulary instead of being
        # hardcoded.
        self.hawkes_persistence = PersistentHawkes(rp, namespace=f"{namespace}__hawkes")
        loaded = self.hawkes_persistence.load()
        self.hawkes = loaded if loaded is not None else MultivariateHawkesProcess(beta=0.5, baseline=0.05)
        # One Dirichlet preference per active-inference faculty.
        self.preference_persistence = PersistentPreference(rp, namespace=f"{namespace}__pref")
        self.spatial_preference = self.preference_persistence.load("spatial") or DirichletPreference(
            len(self.pomdp.observation_names),
            initial_C=list(self.pomdp.C),
            prior_strength=4.0,
        )
        self.causal_preference = self.preference_persistence.load("causal") or DirichletPreference(
            len(self.causal_pomdp.observation_names),
            initial_C=list(self.causal_pomdp.C),
            prior_strength=4.0,
        )
        self._sync_preference_to_pomdp()
        # Hebbian-promoted ontology axes share the sketch dimension.
        self.ontology_persistence = PersistentOntologicalRegistry(rp, namespace=f"{namespace}__ontology")
        self.ontology = self.ontology_persistence.load(dim=FrameDimensions.SKETCH_DIM, frequency_threshold=8)
        # Causal-discovery learns a fresh SCM from observation data when DMN
        # decides the user has accumulated enough coherent variables to
        # justify rebuilding the model. The learned SCM is kept separate from
        # the bootstrap Simpson model so it is easy to A/B in benchmarks.
        self.discovered_scm: Any = None
        # Replay buffer for motor learning. Each item is one chat turn the
        # substrate produced; the trainer pulls items from here at REM time.
        self.motor_replay: list[dict] = []

        self.motor_trainer = GraftMotorTrainer(self.host, self.tokenizer, (self.feature_graft,))

        # Proceduralization (System 2 → System 1). The macro registry persists
        # compiled motifs across processes; the compiler runs on every DMN tick
        # and grows the registry as repeated reasoning patterns are detected.
        self.macro_registry = MacroChunkRegistry(rp, namespace=f"{namespace}__macros")
        self.chunking_compiler = DMNChunkingCompiler(self, registry=self.macro_registry)

        # Native tool synthesis. Tools live in the same SQLite file but in their
        # own namespace; ``attach_tools_to_scm`` rehydrates every persisted tool
        # into the live SCM as an endogenous equation.
        self.tool_registry = NativeToolRegistry(rp, namespace=f"{namespace}__tools")
        try:
            self.tool_registry.attach_to_scm(
                self.scm,
                topology_lock=self._cognitive_state_lock,
                on_tool_drift=self._handle_native_tool_drift,
            )
        except Exception:
            logger.exception("SubstrateController: initial tool attachment failed")

        # Activation-memory-backed dynamic graft synthesizer. The same SQLite
        # file backs the activation memory; modes are stored under their own
        # kind so they don't collide with other activation rows.
        self.activation_memory = SQLiteActivationMemory(
            rp, default_namespace=f"{namespace}__activation"
        )
        self.dynamic_graft_synth = DynamicGraftSynthesizer(
            self.activation_memory, namespace=f"{namespace}__activation"
        )

        # Tool foraging agent. The number of existing tools and the unified
        # agent's posterior entropy together drive when ``synthesize_tool``
        # wins on Expected Free Energy.
        self.tool_foraging_agent = ToolForagingAgent.build(
            n_existing_tools=self.tool_registry.count(),
            insufficient_prior=0.5,
        )

        # Workspace for live UI / debugger feeds. Defaults to the process-wide
        # one so the TUI sees publishes from this mind without explicit wiring.
        self.event_bus: BaseWorkspace = WorkspaceBuilder().process_default()
        self._last_chat_meta: dict[str, Any] = {}
        self._db_path = rp
        self._namespace = namespace
        self._llama_model_id = mid

    @property
    def llama_model_id(self) -> str:
        return self._llama_model_id

    @property
    def db_path(self) -> Path:
        return self._db_path

    @property
    def namespace(self) -> str:
        return self._namespace

    @property
    def background_worker(self) -> CognitiveBackgroundWorker | None:
        return self._background_worker

    def deferred_relation_ingest_online(self) -> bool:
        worker = self._background_worker
        return worker is not None and worker.running

    def deferred_relation_ingest_count(self) -> int:
        return len(self._deferred_relation_jobs)

    def _enqueue_deferred_relation_ingest(
        self,
        utterance: str,
        toks: Sequence[str],
        intent: UtteranceIntent,
        *,
        journal_id: int,
    ) -> DeferredRelationIngest:
        if not intent.allows_storage:
            raise ValueError(f"cannot defer non-storable intent: {intent.label}")

        job = DeferredRelationIngest(
            job_id=int(self._next_deferred_relation_job_id),
            utterance=str(utterance),
            tokens=tuple(str(t) for t in toks),
            intent=intent,
            journal_id=int(journal_id),
            queued_at=time.time(),
        )
        self._next_deferred_relation_job_id += 1
        self._deferred_relation_jobs.append(job)

        payload = {
            "job_id": job.job_id,
            "journal_id": job.journal_id,
            "intent_label": intent.label,
            "intent_confidence": float(intent.confidence),
            "pending": len(self._deferred_relation_jobs),
            "utterance": job.utterance[:200],
        }
        self.event_bus.publish("deferred_relation_ingest.queued", payload)

        worker = self._background_worker
        if worker is not None:
            worker.notify_work()

        return job

    def process_deferred_relation_ingest(self) -> list[dict[str, Any]]:
        with self._cognitive_state_lock:
            reflections: list[dict[str, Any]] = []
            while self._deferred_relation_jobs:
                job = self._deferred_relation_jobs.popleft()
                reflections.append(self._process_deferred_relation_job(job))
            return reflections

    def _process_deferred_relation_job(self, job: DeferredRelationIngest) -> dict[str, Any]:
        claim = self.router.extractor.extract_claim(
            job.utterance,
            job.tokens,
            utterance_intent=job.intent,
        )
        if claim is None:
            reflection = {
                "kind": "deferred_relation_ingest",
                "status": "no_relation",
                "job_id": job.job_id,
                "journal_id": job.journal_id,
                "utterance": job.utterance[:200],
                "intent_label": job.intent.label,
                "pending": len(self._deferred_relation_jobs),
            }
            self.event_bus.publish("deferred_relation_ingest.processed", reflection)
            return reflection

        refined = self.refine_extracted_claim(job.utterance, job.tokens, claim)
        frame = self.router._memory_write(self, job.utterance, refined)
        frame.evidence = {
            **dict(frame.evidence or {}),
            "deferred_relation_job_id": job.job_id,
            "source_journal_id": job.journal_id,
            "queued_at": job.queued_at,
            "processed_at": time.time(),
        }
        self.workspace.post_frame(frame)
        self._after_deferred_relation_commit(frame, job)

        reflection = {
            "kind": "deferred_relation_ingest",
            "status": frame.intent,
            "job_id": job.job_id,
            "journal_id": job.journal_id,
            "subject": frame.subject,
            "answer": frame.answer,
            "confidence": float(frame.confidence),
            "evidence": dict(frame.evidence),
            "pending": len(self._deferred_relation_jobs),
        }
        self.event_bus.publish("deferred_relation_ingest.processed", reflection)
        return reflection

    def _after_deferred_relation_commit(
        self,
        frame: CognitiveFrame,
        job: DeferredRelationIngest,
    ) -> None:
        try:
            self.hawkes.observe(str(frame.intent or "unknown"))
        except Exception:
            logger.exception("_after_deferred_relation_commit: hawkes observe failed")

        self._observe_frame_concepts(frame)
        self._remember_declarative_binding(frame, job.utterance)

    def consolidate_once(self) -> list[dict]:
        out = self.memory.consolidate_claims_once()
        logger.debug("SubstrateController.consolidate_once: reflections=%d", len(out))
        try:
            self.event_bus.publish("consolidation", {"reflections": len(out)})
        except Exception:
            logger.exception("SubstrateController.consolidate_once: event publish failed")
        return out

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-friendly snapshot of substrate state for live UIs.

        Designed to be cheap (read-only attribute access, no SQL writes) and
        safe (each subsystem is wrapped so a partial failure cannot break the
        UI). Callers may invoke this on a tick (the TUI polls at ~5Hz) without
        bothering with locks; the returned dict is a fresh copy.
        """

        snap: dict[str, Any] = {"ts": time.time()}

        try:
            device = next(self.host.parameters()).device
            device_str = str(device)
        except (StopIteration, AttributeError):
            device_str = "unknown"
        snap["model"] = {
            "id": self._llama_model_id,
            "device": device_str,
            "namespace": self._namespace,
            "db_path": str(self._db_path),
        }

        try:
            recent_claims = self.memory.claims()[-8:]
            mean_conf = self.memory.mean_confidence()
            snap["memory"] = {
                "count": int(self.memory.count()),
                "subjects": len(self.memory.subjects()),
                "mean_confidence": (float(mean_conf) if mean_conf is not None else None),
                "recent_claims": [
                    {
                        "subject": c.get("subject"),
                        "predicate": c.get("predicate"),
                        "object": c.get("object"),
                        "confidence": float(c.get("confidence", 0.0)),
                        "status": c.get("status"),
                    }
                    for c in recent_claims
                ],
            }
        except Exception:
            logger.exception("snapshot.memory failed")
            snap["memory"] = {"error": True}

        try:
            recent_journal = self.journal.recent(8)
            snap["journal"] = {
                "count": int(self.journal.count()),
                "recent": [
                    {
                        "id": int(r.get("id", 0)),
                        "intent": r.get("intent"),
                        "subject": r.get("subject"),
                        "answer": r.get("answer"),
                        "confidence": float(r.get("confidence", 0.0)),
                        "utterance": (r.get("utterance") or "")[:200],
                    }
                    for r in recent_journal
                ],
            }
        except Exception:
            logger.exception("snapshot.journal failed")
            snap["journal"] = {"error": True}

        try:
            latest = self.workspace.latest
            snap["workspace"] = {
                "frames_total": len(self.workspace.frames),
                "working_window": len(self.workspace.working),
                "intrinsic_cues": [
                    {
                        "urgency": float(c.urgency),
                        "faculty": c.faculty,
                        "source": c.source,
                        "evidence": dict(c.evidence) if isinstance(c.evidence, dict) else {},
                    }
                    for c in self.workspace.intrinsic_cues
                ],
                "latest_frame": (
                    {
                        "intent": latest.intent,
                        "subject": latest.subject,
                        "answer": latest.answer,
                        "confidence": float(latest.confidence),
                    }
                    if latest is not None
                    else None
                ),
            }
        except Exception:
            logger.exception("snapshot.workspace failed")
            snap["workspace"] = {"error": True}

        try:
            bg = self._background_worker
            snap["background"] = bg.state_snapshot() if bg is not None else {"running": False}
        except Exception:
            logger.exception("snapshot.background failed")
            snap["background"] = {"error": True}

        try:
            sw = self._self_improve_worker
            if sw is None:
                snap["self_improve"] = {"running": False, "enabled": False}
            else:
                snap["self_improve"] = {
                    "running": bool(sw.running),
                    "enabled": bool(getattr(sw.config, "enabled", False)),
                    "iterations": sw.get_iterations(),
                    "interval_s": float(getattr(sw.config, "interval_s", 0.0)),
                    "last_summary": sw.last_summary,
                    "last_error": sw.last_error,
                }
        except Exception:
            logger.exception("snapshot.self_improve failed")
            snap["self_improve"] = {"error": True}

        try:
            snap["substrate"] = {
                "vsa_atoms": len(self.vsa),
                "hopfield_stored": len(self.hopfield_memory),
                "hopfield_max_items": int(self.hopfield_memory.max_items),
                "hawkes_channels": len(self.hawkes.channels),
                "hawkes_intensity": dict(self.hawkes.intensity_vector()),
                "tools": int(self.tool_registry.count()),
                "macros": int(self.macro_registry.count()),
                "deferred_relation_ingest_pending": self.deferred_relation_ingest_count(),
                "ontology_axes": len(self.ontology),
                "discovered_scm": self.discovered_scm is not None,
            }
        except Exception:
            logger.exception("snapshot.substrate failed")
            snap["substrate"] = {"error": True}

        try:
            snap["encoders"] = self.multimodal_perception.stats()
        except Exception:
            logger.exception("snapshot.encoders failed")
            snap["encoders"] = {"error": True}

        try:
            snap["affect"] = self.affect_trace.summary()
        except Exception:
            logger.exception("snapshot.affect failed")
            snap["affect"] = {"error": True}

        try:
            snap["preferences"] = {
                "spatial_C": [float(x) for x in self.spatial_preference.expected_C()],
                "causal_C": [float(x) for x in self.causal_preference.expected_C()],
            }
        except Exception:
            logger.exception("snapshot.preferences failed")
            snap["preferences"] = {"error": True}

        try:
            snap["last_chat"] = dict(self._last_chat_meta) if self._last_chat_meta else None
        except Exception:
            snap["last_chat"] = None

        return snap

    # -- New substrate plumbing -----------------------------------------------

    def _sync_preference_to_pomdp(self) -> None:
        """Push the Dirichlet means into the live POMDPs' C vectors."""

        try:
            self.pomdp.C = list(self.spatial_preference.expected_C())
        except (AttributeError, TypeError):
            logger.exception("SubstrateController._sync_preference_to_pomdp: spatial sync failed")
        try:
            self.causal_pomdp.C = list(self.causal_preference.expected_C())
        except (AttributeError, TypeError):
            logger.exception("SubstrateController._sync_preference_to_pomdp: causal sync failed")

    def observe_user_feedback(
        self,
        *,
        faculty: str,
        observation_index: int,
        polarity: float,
        weight: float = 1.0,
        reason: str = "",
        conformal_set_size: int | None = None,
        epistemic_ambiguity_floor_strength: float = 0.18,
    ) -> None:
        """Forward user feedback into the right Dirichlet preference and sync.

        When ``conformal_set_size`` is strictly greater than one the substrate
        is in a demonstrably ambiguous regime; negative preference updates
        then respect an irreducible concentration floor so ``C`` cannot collapse
        toward silence simply because the user vented frustration.
        """

        if faculty == "spatial":
            target = self.spatial_preference
        elif faculty == "causal":
            target = self.causal_preference
        else:
            raise ValueError(f"SubstrateController.observe_user_feedback: unsupported faculty {faculty!r}; expected 'spatial' or 'causal'")
        floor: float | None = None
        if polarity < 0 and conformal_set_size is not None and int(conformal_set_size) > 1:
            floor = float(target.prior_strength * epistemic_ambiguity_floor_strength)
        target.update(
            observation_index,
            polarity=polarity,
            weight=weight,
            reason=reason,
            epistemic_alpha_floor=floor,
        )
        self._sync_preference_to_pomdp()
        try:
            self.preference_persistence.save(faculty, target)
        except (sqlite3.Error, OSError):
            logger.exception("SubstrateController.observe_user_feedback: preference save failed")

    def observe_event(self, channel: str, *, t: float | None = None) -> None:
        """Record an event on the Hawkes layer (used by the conversational loop)."""

        self.hawkes.observe(channel, t=t)

    def encode_triple_vsa(self, subject: str, predicate: str, obj: str) -> torch.Tensor:
        """Compose a hypervector representation of (subject, predicate, object).

        The VSA bundle is independent of the LLM's tokenizer and lets the
        substrate do role-filler algebra on facts without round-tripping
        through subwords.
        """

        return self.vsa.encode_triple(subject, predicate, obj)

    def _padded_hopfield_sketch(self, sketch: torch.Tensor) -> torch.Tensor:
        """Embed a lexical sketch in the Hopfield model width (zeros outside the sketch prefix)."""

        d = self.hopfield_memory.d_model
        out = torch.zeros(d, dtype=torch.float32)
        s = sketch.detach().float().view(-1)
        n = min(int(s.numel()), d)
        if n > 0:
            out[:n] = s[:n]
        return out

    def remember_hopfield(
        self,
        a_sketch: torch.Tensor,
        b_sketch: torch.Tensor,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Associate two padded sketches in Hopfield memory (public entry for tooling)."""

        self.hopfield_memory.remember(
            self._padded_hopfield_sketch(a_sketch),
            self._padded_hopfield_sketch(b_sketch),
            metadata=dict(metadata or {}),
        )

    def _after_frame_commit(
        self,
        out: CognitiveFrame,
        utterance: str,
        *,
        event_topic: str,
    ) -> None:
        """Run shared post-commit substrate side effects for a published frame."""

        try:
            self.hawkes.observe(str(out.intent or "unknown"))
        except Exception:
            logger.exception("_after_frame_commit: hawkes observe failed")

        if self._background_worker is not None:
            self._background_worker.mark_user_active()

        self._observe_frame_concepts(out)
        self._remember_declarative_binding(out, utterance)

        logger.debug(
            "_after_frame_commit: intent=%s confidence=%s journal_id=%s",
            out.intent,
            out.confidence,
            (out.evidence or {}).get("journal_id"),
        )

        try:
            payload = {
                "intent": out.intent,
                "subject": out.subject,
                "answer": out.answer,
                "confidence": float(out.confidence),
                "journal_id": (out.evidence or {}).get("journal_id"),
                "utterance": utterance[:200],
            }
            if event_topic == "frame.perception":
                payload.update(
                    {
                        "modality": (out.evidence or {}).get("modality"),
                        "source": (out.evidence or {}).get("source"),
                        "feature_dim": (out.evidence or {}).get("feature_dim"),
                    }
                )
            self.event_bus.publish(event_topic, payload)
        except Exception:
            logger.exception("_after_frame_commit: event publish failed")

    def _observe_frame_concepts(self, out: CognitiveFrame) -> None:
        for concept in (out.subject, out.answer):
            if isinstance(concept, str) and concept and concept != "unknown":
                self.ontology.observe(concept)
                base = _SUBWORD.encode(concept)
                self.ontology.maybe_promote(concept, base)

    def _remember_declarative_binding(self, out: CognitiveFrame, utterance: str) -> None:
        if out.subject and out.answer and out.intent in {"memory_write", "memory_lookup"}:
            try:
                pr_bind = str((out.evidence or {}).get("predicate", out.intent))
                self.vsa.encode_triple(out.subject, pr_bind, out.answer)
                ut_sk = _SUBWORD.encode(utterance[:512])
                trip_sk = _SUBWORD.encode(f"{out.subject}|{pr_bind}|{out.answer}")
                self.remember_hopfield(
                    ut_sk,
                    trip_sk,
                    metadata={"kind": "declarative_binding", "intent": out.intent},
                )
            except Exception:
                logger.exception("_after_frame_commit: vsa/hopfield binding failed")

    def _frame_from_observation(self, observation: CognitiveObservation) -> CognitiveFrame:
        """Convert a strict multimodal observation to a workspace frame."""

        return CognitiveFrame(
            f"perception_{observation.modality}",
            subject=observation.subject,
            answer=observation.answer,
            confidence=float(observation.confidence),
            evidence={
                **observation.frame_evidence(),
                "is_actionable": True,
                "allows_storage": False,
                "intent_label": f"perception_{observation.modality}",
                "intent_confidence": float(observation.confidence),
            },
        )

    def _commit_observation(self, observation: CognitiveObservation) -> CognitiveFrame:
        """Publish a multimodal observation into journal, workspace, VSA, and Hopfield memory."""

        source_text = f"[{observation.modality}:{observation.source}] {observation.answer}"
        frame = self._frame_from_observation(observation)
        with self._cognitive_state_lock:
            out = self._commit_frame(source_text, utterance_words(source_text), frame)
            self.vsa.encode_triple(observation.modality, "observed_as", observation.answer)
            self.remember_hopfield(
                _SUBWORD.encode(source_text[:512]),
                observation.features,
                metadata={
                    "kind": "multimodal_observation",
                    "modality": observation.modality,
                    "source": observation.source,
                    "intent": out.intent,
                    "journal_id": (out.evidence or {}).get("journal_id"),
                },
            )
        self._after_frame_commit(out, source_text, event_topic="frame.perception")
        return out

    def perceive_image(self, image: Any, *, source: str = "image") -> CognitiveFrame:
        """Run the vision encoders and commit their fused observation."""

        return self._commit_observation(
            self.multimodal_perception.perceive_image(image, source=source)
        )

    def perceive_video(self, frames: Any, *, source: str = "video") -> CognitiveFrame:
        """Run temporal + vision encoders and commit their fused observation."""

        return self._commit_observation(
            self.multimodal_perception.perceive_video(frames, source=source)
        )

    def perceive_audio(
        self,
        audio: Any,
        *,
        sampling_rate: int = 16000,
        source: str = "audio",
        language: str | None = None,
    ) -> CognitiveFrame:
        """Run Whisper/ImageBind audio encoders, then route transcripts through language memory."""

        observation = self.multimodal_perception.perceive_audio(
            audio,
            sampling_rate=int(sampling_rate),
            source=source,
            language=language,
        )
        out = self._commit_observation(observation)
        transcription = str((observation.evidence or {}).get("transcription") or "").strip()
        if transcription:
            transcription_frame = self.comprehend(transcription)
            try:
                self.event_bus.publish(
                    "frame.perception.transcription",
                    {
                        "audio_journal_id": (out.evidence or {}).get("journal_id"),
                        "transcription_journal_id": (transcription_frame.evidence or {}).get("journal_id"),
                        "transcription": transcription[:200],
                    },
                )
            except Exception:
                logger.exception("perceive_audio: transcription event publish failed")
        return out

    def broca_features_from_frame(self, frame: CognitiveFrame) -> torch.Tensor:
        """Sketch frame + numeric tail + sparse VSA injection for :class:`TrainableFeatureGraft`."""

        vsa_vec: torch.Tensor | None = None
        if frame.subject and frame.answer and str(frame.answer).lower() not in {"", "unknown"}:
            pr = str((frame.evidence or {}).get("predicate", frame.intent))
            try:
                vsa_vec = self.encode_triple_vsa(str(frame.subject), pr, str(frame.answer))
            except (RuntimeError, ValueError, TypeError):
                logger.debug("broca_features_from_frame: VSA encode skipped", exc_info=True)
        return self.frame_packer.broca(
            frame.intent,
            frame.subject,
            frame.answer,
            float(frame.confidence),
            frame.evidence,
            vsa_bundle=vsa_vec,
            vsa_projection_seed=int(self.seed),
        )

    def content_logit_bias_from_frame(self, frame: CognitiveFrame) -> dict[int, float]:
        """Token-ID bonuses derived from frame content for scripted host scoring."""

        return self._content_logit_bias(frame)

    def refine_extracted_claim(
        self, utterance: str, toks: Sequence[str], claim: ParsedClaim
    ) -> ParsedClaim:
        """Contextual cleanup of LLM-parsed triples using VSA similarity + optional Hopfield memory."""

        words = [w.lower() for w in _word_tokens(toks)]
        ctx_words = [w for w in words if len(w) > 1][:28]
        if len(ctx_words) < 2:
            return claim
        try:
            ctx_bundle = bundle([self.vsa.atom(w) for w in ctx_words])
        except (RuntimeError, ValueError, TypeError):
            logger.debug("refine_extracted_claim: context bundle failed", exc_info=True)
            return claim

        pred = claim.predicate.lower()
        candidates_obj: set[str] = {claim.obj.lower()}
        try:
            candidates_obj |= set(self.memory.distinct_objects_for_predicate(pred))
        except (sqlite3.Error, OSError, TypeError):
            logger.debug("refine_extracted_claim: predicate object lookup failed", exc_info=True)
        try:
            for _s, _p, o, _c, _e in self.memory.all_facts():
                ol = str(o).lower()
                if claim.obj.lower() in ol or ol in claim.obj.lower() or ol in words:
                    candidates_obj.add(ol)
        except (sqlite3.Error, OSError, TypeError):
            logger.debug("refine_extracted_claim: all_facts scan failed", exc_info=True)

        candidates_obj = {c for c in candidates_obj if c}
        best_obj = claim.obj.lower()
        try:
            base_trip = self.vsa.encode_triple(claim.subject.lower(), pred, best_obj)
            base_sim = vsa_cosine(ctx_bundle, base_trip)
        except (RuntimeError, ValueError, TypeError):
            return claim

        for cand in candidates_obj:
            if cand == best_obj:
                continue
            try:
                trip = self.vsa.encode_triple(claim.subject.lower(), pred, cand)
                sc = vsa_cosine(ctx_bundle, trip)
                if sc > base_sim + 0.03:
                    base_sim = sc
                    best_obj = cand
            except (RuntimeError, ValueError, TypeError):
                continue

        try:
            q = self._padded_hopfield_sketch(_SUBWORD.encode(utterance[:512]))
            if len(self.hopfield_memory) > 0:
                ret, w = self.hopfield_memory.retrieve(q)
                if w.numel() and float(w.max().item()) > 0.2:
                    hf_best: str | None = None
                    hf_score = -1.0
                    u = ret[:FrameDimensions.SKETCH_DIM]
                    for cand in candidates_obj:
                        cc = float(
                            F.cosine_similarity(
                                u.view(1, -1),
                                _SUBWORD.encode(cand).view(1, -1),
                            ).item()
                        )
                        if cc > hf_score:
                            hf_score = cc
                            hf_best = cand
                    if hf_best is not None and hf_score > 0.38 and hf_best != best_obj:
                        trip_h = self.vsa.encode_triple(claim.subject.lower(), pred, hf_best)
                        if vsa_cosine(ctx_bundle, trip_h) >= base_sim - 0.02:
                            best_obj = hf_best
        except (RuntimeError, ValueError, TypeError):
            logger.debug("refine_extracted_claim: Hopfield assist failed", exc_info=True)

        if best_obj == claim.obj.lower():
            return claim
        ev = dict(claim.evidence)
        ev["wernicke_refine"] = "vsa_hopfield_object"
        ev["object_before_refine"] = claim.obj
        return ParsedClaim(
            subject=claim.subject,
            predicate=claim.predicate,
            obj=best_obj,
            confidence=min(1.0, float(claim.confidence) * 0.95),
            evidence=ev,
        )

    # -- Native tool synthesis -------------------------------------------------

    def _handle_native_tool_drift(self, tool: NativeTool, evidence: Mapping[str, Any]) -> None:
        """Turn native-tool exchangeability drift into an active-inference cue."""

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
        self.workspace.intrinsic_cues.append(cue)
        self.tool_foraging_agent = ToolForagingAgent.build(
            n_existing_tools=self.tool_registry.count(),
            insufficient_prior=1.0 - 1e-6,
        )
        self.event_bus.publish(
            "native_tool.drift",
            {"tool": tool.name, "urgency": cue.urgency, "evidence": dict(cue.evidence)},
        )

    def synthesize_native_tool(
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
        """Compile, sandbox, verify, persist, and (optionally) attach a synthesized tool.

        After synthesis the tool foraging agent's belief is updated to reflect
        the larger toolbox, so the next ``synthesize_tool`` decision factors in
        the additional coverage.
        """

        tool = self.tool_registry.synthesize(
            name,
            source,
            function_name=function_name,
            parents=parents,
            domain=domain,
            sample_inputs=sample_inputs,
            description=description,
            overwrite=overwrite,
            conformal_predictor=self.native_tool_conformal,
        )
        if attach:
            try:
                self.tool_registry.attach_to_scm(
                    self.scm,
                    topology_lock=self._cognitive_state_lock,
                    on_tool_drift=self._handle_native_tool_drift,
                )
            except Exception:
                logger.exception("SubstrateController.synthesize_native_tool: SCM re-attach failed")
        # Rebuild the tool foraging agent so its likelihoods reflect the new tool count.
        self.tool_foraging_agent = ToolForagingAgent.build(
            n_existing_tools=self.tool_registry.count(),
            insufficient_prior=0.5,
        )
        return tool

    def attach_tools_to_scm(self) -> int:
        """Re-attach every persisted native tool onto :attr:`scm`. Returns the count attached."""

        return self.tool_registry.attach_to_scm(
            self.scm,
            topology_lock=self._cognitive_state_lock,
            on_tool_drift=self._handle_native_tool_drift,
        )

    def should_synthesize_tool(self) -> bool:
        """Run the tool foraging agent against the current substrate state.

        The ``insufficient_prior`` is derived from the unified agent's
        normalized posterior entropy: when the substrate is genuinely
        confused (high entropy → high prior on ``knowledge_insufficient``)
        the EFE math will prefer ``synthesize_tool`` over the alternatives.
        """

        try:
            coupled = self.unified_agent.decide()
        except Exception:
            return False
        # Use whichever faculty currently wins on EFE; its posterior entropy is
        # the substrate's best self-estimate of confusion.
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
        self.tool_foraging_agent.update_belief(insufficient_prior=float(insufficient_prior))
        return self.tool_foraging_agent.should_synthesize()

    # -- Proceduralization / macro lookup --------------------------------------

    def recent_intents(self, *, limit: int = 8) -> list[str]:
        try:
            rows = self.journal.recent(limit=int(limit))
        except Exception:
            return []
        return [str(r.get("intent", "") or "unknown") for r in rows]

    def find_matching_macro(
        self,
        *,
        recent_intents: Sequence[str] | None = None,
        features: torch.Tensor | None = None,
    ) -> CompiledMacro | None:
        """Return the most-observed macro whose prefix matches the recent intent tail."""

        if features is not None:
            return self.macro_registry.find_macro_by_features(
                features,
                min_cosine=self.chunking_compiler.config.hopfield_weight_min_for_oneshot,
            )
        recent = list(recent_intents) if recent_intents is not None else self.recent_intents()
        return self.macro_registry.find_macro_matching_prefix(recent)

    def macro_speech_features(self, macro: CompiledMacro) -> torch.Tensor:
        """Return the FrameDimensions.broca_feature_dim()-shaped features the macro should inject via TrainableFeatureGraft."""

        return macro_frame_features(macro)

    # -- Dynamic graft synthesis -----------------------------------------------

    def synthesize_activation_mode(
        self,
        *,
        name: str,
        prompt: str,
        slot: str = "final_hidden",
        query_mode: str = "sequence_mean",
        value_mode: str = "mean_activation",
        target_token: str | None = None,
        confidence: float = 1.0,
    ) -> CapturedActivationMode:
        """Capture and persist an activation mode for the host (System-1 LLM tool).

        The captured mode lives in :attr:`activation_memory` and can be loaded
        into a :class:`KVMemoryGraft` via
        :meth:`load_activation_modes_into_graft`.
        """

        return self.dynamic_graft_synth.synthesize(
            self.host,
            self.tokenizer,
            name=name,
            prompt=prompt,
            slot=slot,
            query_mode=query_mode,
            value_mode=value_mode,
            target_token=target_token,
            confidence=float(confidence),
        )

    def load_activation_modes_into_graft(
        self,
        graft: Any,
        *,
        names: Optional[Sequence[str]] = None,
        clear_first: bool = True,
    ) -> int:
        return self.dynamic_graft_synth.load_modes(
            graft, names=names, clear_first=clear_first
        )

    def vector_for_concept(self, name: str, *, base_sketch: torch.Tensor | None = None) -> torch.Tensor:
        """Return the substrate's preferred vector for a concept name.

        Routes through the ontology registry so frequent concepts use their
        promoted orthogonal axis; less-frequent ones still use the hashed
        sketch. Always observes the access (so the next call can flip
        promotion).
        """

        self.ontology.observe(name)
        sketch = base_sketch if base_sketch is not None else _SUBWORD.encode(name)
        promoted = self.ontology.maybe_promote(name, sketch)
        if promoted is not None:
            return promoted.axis
        return F.normalize(sketch.detach().to(torch.float32).flatten(), dim=0)

    def start_background(
        self,
        *,
        interval_s: float = 5.0,
        config: DMNConfig | None = None,
    ) -> CognitiveBackgroundWorker:
        if self._background_worker is None:
            self._background_worker = CognitiveBackgroundWorker(
                self,
                interval_s=interval_s,
                config=config,
                motor_trainer=self.motor_trainer,
            )
        else:
            self._background_worker.interval_s = max(0.1, float(interval_s))
            if config is not None:
                self._background_worker.config = config
        self._background_worker.start()
        return self._background_worker

    def stop_background(self) -> None:
        if self._background_worker is not None:
            self._background_worker.stop()

    def start_self_improve_worker(
        self,
        *,
        interval_s: float | None = None,
        enabled: bool | None = None,
    ) -> Any:
        """Start Docker-backed self-improve loop (separate from DMN background).

        See :mod:`core.workers.docker_self_improve_worker` for environment variables
        and prerequisites (``GITHUB_TOKEN``, Docker, and ``repo`` scope).
        """

        try:
            from ..workers.docker_self_improve_worker import SelfImproveConfig, SelfImproveDockerWorker
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "Could not import core.workers.docker_self_improve_worker (self-improve worker). "
                "Ensure project dependencies are installed and Docker is available on the host; "
                "see core.workers.docker_self_improve_worker module docs."
            ) from exc

        cfg = SelfImproveConfig()
        if enabled is not None:
            cfg.enabled = bool(enabled)
        if interval_s is not None:
            cfg.interval_s = max(60.0, float(interval_s))
        if self._self_improve_worker is None:
            self._self_improve_worker = SelfImproveDockerWorker(self, config=cfg)
        else:
            self._self_improve_worker.config = cfg
        self._self_improve_worker.start()
        return self._self_improve_worker

    def stop_self_improve_worker(self, timeout: float = 5.0) -> None:
        if self._self_improve_worker is not None:
            self._self_improve_worker.stop(timeout=timeout)

    def _intrinsic_scan(self, toks: list[str]) -> None:
        self.workspace.intrinsic_cues.clear()
        mu_pop = self.memory.mean_confidence()
        confidence_floor = SEMANTIC_CONFIDENCE_FLOOR if mu_pop is None else max(SEMANTIC_CONFIDENCE_FLOOR, float(mu_pop))
        toks_set = set(toks)
        for ent in self.memory.subjects():
            if ent not in toks_set:
                continue
            records = self.memory.records_for_subject(ent)
            if not records:
                self.workspace.intrinsic_cues.append(IntrinsicCue(1.0, "memory_gap", {"subject": ent}))
                continue
            best_pred, _obj, best_conf, _ev = max(records, key=lambda row: row[2])
            if best_conf < confidence_floor:
                self.workspace.intrinsic_cues.append(
                    IntrinsicCue(
                        float(confidence_floor - best_conf),
                        "memory_low_confidence",
                        {"subject": ent, "predicate": best_pred, "confidence": best_conf},
                    )
                )
        cq = self.causal_agent.qs
        if cq is not None and len(cq) >= 2:
            max_ent = math.log(len(cq))
            h_q = belief_entropy(cq)
            if max_ent > 1e-9 and h_q > 0.5 * max_ent:
                self.workspace.intrinsic_cues.append(IntrinsicCue(float(h_q / max_ent), "causal_uncertain", {"entropy": h_q}))
        logger.debug("_intrinsic_scan: cues=%d toks=%d", len(self.workspace.intrinsic_cues), len(toks))
        try:
            for cue in self.workspace.intrinsic_cues:
                self.event_bus.publish(
                    "intrinsic_cue",
                    {"urgency": float(cue.urgency), "faculty": cue.faculty, "evidence": dict(cue.evidence) if isinstance(cue.evidence, dict) else {}},
                )
        except Exception:
            logger.exception("_intrinsic_scan: event publish failed")

    def _non_actionable_frame(self, intent: UtteranceIntent, affect: AffectState) -> "CognitiveFrame":
        """Frame for utterances the substrate has nothing legitimate to say about.

        Greetings, requests, commands, and feedback do not yield a triple to
        store or a question to answer; producing a non-trivial frame for them
        only invites the grafts to bias the LLM toward content the substrate
        did not actually retrieve. Returning an explicit ``unknown`` frame
        with confidence 0 is what the rest of the pipeline keys off of to
        skip graft activation entirely.
        """

        evidence = {
            "route": "intent_gate",
            "intent_label": intent.label,
            "intent_confidence": float(intent.confidence),
            "intent_scores": dict(intent.scores),
            "is_actionable": False,
            "allows_storage": intent.allows_storage,
            "affect": _affect_evidence(affect),
        }
        return CognitiveFrame(
            "unknown",
            answer="unknown",
            confidence=0.0,
            evidence=evidence,
        )

    def _attach_perception(
        self, frame: "CognitiveFrame", intent: UtteranceIntent, affect: AffectState
    ) -> None:
        """Attach intent + affect signals to the frame's evidence in-place."""

        frame.evidence = {
            **dict(frame.evidence or {}),
            "intent_label": intent.label,
            "intent_confidence": float(intent.confidence),
            "intent_scores": dict(intent.scores),
            "is_actionable": True,
            "allows_storage": intent.allows_storage,
            "affect": _affect_evidence(affect),
        }

    def comprehend(self, utterance: str) -> CognitiveFrame:
        toks = utterance_words(utterance)
        intent, affect = self._perceive_utterance(utterance)
        with self._cognitive_state_lock:
            self._intrinsic_scan(toks)
            self._last_intent = intent
            self._last_affect = affect
            if not intent.is_actionable:
                frame = self._non_actionable_frame(intent, affect)
            else:
                frame = self.router.route(self, utterance, toks, utterance_intent=intent)
                self._attach_perception(frame, intent, affect)
            out = self._commit_frame(utterance, toks, frame)
            if bool((out.evidence or {}).get("deferred_relation_ingest")):
                journal_id = (out.evidence or {}).get("journal_id")
                if journal_id is None:
                    raise RuntimeError("deferred relation ingest frame is missing journal_id")
                self._enqueue_deferred_relation_ingest(
                    utterance,
                    toks,
                    intent,
                    journal_id=int(journal_id),
                )
            self._last_user_affect_trace_id = self.affect_trace.record(
                role="user",
                text=utterance,
                affect=affect,
                journal_id=(out.evidence or {}).get("journal_id"),
            )
        self._after_frame_commit(out, utterance, event_topic="frame.comprehend")
        return out

    def _perceive_utterance(self, utterance: str) -> tuple[UtteranceIntent, AffectState]:
        with ThreadPoolExecutor(max_workers=2) as executor:
            intent_future = executor.submit(self.intent_gate.classify, utterance)
            affect_future = executor.submit(self.affect_encoder.detect, utterance)
            return intent_future.result(), affect_future.result()

    def _commit_frame(self, utterance: str, toks: Sequence[str], frame: CognitiveFrame) -> CognitiveFrame:
        commit_ts = time.time()
        trace = self.hawkes.trace(t=commit_ts)
        frame.evidence = {**dict(frame.evidence or {}), "hawkes_trace": trace}
        jid = self.journal.append(utterance, frame, ts=commit_ts)
        frame.evidence = {**frame.evidence, "journal_id": jid}
        if self._last_journal_id is not None:
            self.episode_graph.bump(self._last_journal_id, jid)
        self._last_journal_id = jid
        logger.debug("_commit_frame: journal_id=%s intent=%s pred_error=%s", jid, frame.intent, frame.intent == "prediction_error")
        out = self.workspace.post_frame(frame)
        predicate = str((out.evidence or {}).get("predicate", ""))
        if out.intent == "memory_write" and out.subject and predicate:
            self.memory.merge_epistemic_evidence(out.subject, predicate, out.evidence)
        for tail in self.workspace.frames:
            pred = str((tail.evidence or {}).get("predicate", ""))
            if tail.intent == "synthesis_bundle" and tail.subject and pred:
                self.memory.merge_epistemic_evidence(tail.subject, pred, tail.evidence)
        logger.debug("_commit_frame: published intent=%s workspace_frames=%d", out.intent, len(self.workspace.frames))
        return out

    def retrieve_episode(self, episode_id: int) -> CognitiveFrame:
        """Reload a prior workspace episode into working memory (persistent episodic retrieval)."""

        row = self.journal.fetch(episode_id)
        if row is None:
            logger.debug("retrieve_episode: missing id=%s", episode_id)
            return CognitiveFrame(
                "unknown",
                answer="unknown",
                confidence=0.0,
                evidence={"missing_episode_id": int(episode_id)},
            )
        replay = CognitiveFrame.from_episode_row(row)
        self.workspace.post_frame(replay)
        logger.debug("retrieve_episode: id=%s intent=%s", episode_id, replay.intent)
        return replay

    def speak(self, frame: CognitiveFrame) -> str:
        """Plan-forced surface generation via :class:`LexicalPlanGraft`.

        Retained for benchmark code that scores the substrate's ability to
        produce specific tokens. Conversational use should call
        :meth:`chat_reply` so the LLM speaks freely under soft graft bias.

        Uses the same :meth:`_record_motor_replay` path as :meth:`chat_reply`
        after decoding so REM trains the residual graft on lexical-plan emits.
        """

        plan_words = frame.speech_plan()
        broca_features = self.broca_features_from_frame(frame)
        text_out, token_ids, inertia_tail = generate_from_plan(
            self.host,
            self.tokenizer,
            plan_words,
            broca_features=broca_features,
        )
        confidence = max(0.0, min(1.0, float(frame.confidence)))
        msgs = _motor_replay_messages_plan_forced(frame, plan_words)
        self._record_motor_replay(
            msgs,
            generated_token_ids=token_ids,
            broca_features=broca_features,
            substrate_confidence=confidence,
            substrate_inertia=inertia_tail,
        )
        return text_out

    def answer(self, utterance: str, *, max_new_tokens: int | None = None) -> tuple[CognitiveFrame, str]:
        """One-shot natural-language reply driven by substrate-biased decoding."""

        if max_new_tokens is None:
            return self.chat_reply([{"role": "user", "content": utterance}])
        return self.chat_reply([{"role": "user", "content": utterance}], max_new_tokens=int(max_new_tokens))

    def chat_reply(
        self,
        messages: Sequence[dict[str, str]],
        *,
        max_new_tokens: int = 256,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9,
        on_token: Callable[[str], None] | None = None,
    ) -> tuple[CognitiveFrame, str]:
        """Substrate-biased free-form chat reply.

        The last user message routes through :meth:`comprehend` to obtain a
        cognitive frame. The frame's continuous features feed
        :class:`TrainableFeatureGraft` (residual-stream bias) and a derived
        logit-bias dict over the answer's content subwords feeds
        :class:`SubstrateLogitBiasGraft` (token-level bias). The LLM then
        decodes a free-form reply through its own chat template — surface
        form, fluency, and ordering are entirely the LLM's choice. The
        sampling temperature is annealed by the frame's confidence so
        high-confidence frames produce decisive replies and ``unknown`` /
        low-confidence frames let the LLM speak freely with no bias at all.
        """

        msgs = [dict(m) for m in messages]
        if not msgs or msgs[-1].get("role") != "user":
            raise ValueError("chat_reply expects messages ending with a user turn")
        user_text = str(msgs[-1].get("content", "")).strip()
        frame = self.comprehend(user_text)

        confidence = max(0.0, min(1.0, float(frame.confidence)))
        derived_scale = self._derived_target_snr_scale(frame)
        if derived_scale <= 0.0:
            broca_features = None
            logit_bias: dict[int, float] = {}
        else:
            broca_features = self.broca_features_from_frame(frame) if frame.intent != "unknown" else None
            logit_bias = self._content_logit_bias(frame)
        eff_temperature = max(
            1e-3,
            float(temperature) * self._substrate_temperature_scale(frame, confidence),
        )
        logger.debug(
            "chat_reply: intent=%s bias_tokens=%d has_broca_features=%s confidence=%.3f eff_temperature=%.3f derived_scale=%.3f",
            frame.intent,
            len(logit_bias),
            broca_features is not None,
            confidence,
            eff_temperature,
            derived_scale,
        )
        bias_top: list[dict[str, Any]] = []
        try:
            hf_tok = getattr(self.tokenizer, "inner", None)
            if hf_tok is not None and logit_bias:
                ranked = sorted(logit_bias.items(), key=lambda kv: kv[1], reverse=True)[:8]
                for tid, val in ranked:
                    try:
                        piece = hf_tok.decode([int(tid)], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    except Exception:
                        piece = f"<{tid}>"
                    bias_top.append({"token_id": int(tid), "token": piece, "bias": float(val)})
        except Exception:
            logger.exception("chat_reply: bias_top extraction failed")

        self._last_chat_meta = {
            "intent": frame.intent,
            "subject": frame.subject,
            "answer": frame.answer,
            "confidence": float(confidence),
            "eff_temperature": float(eff_temperature),
            "bias_token_count": len(logit_bias),
            "bias_top": bias_top,
            "has_broca_features": broca_features is not None,
            "derived_target_snr_scale": float(derived_scale),
            "ts": time.time(),
        }
        try:
            self.event_bus.publish("chat.start", dict(self._last_chat_meta))
        except Exception:
            logger.exception("chat_reply: event publish failed")

        text, gen_ids, sub_inertia = self._stream_substrate_chat(
            msgs,
            broca_features=broca_features,
            logit_bias=logit_bias,
            max_new_tokens=int(max_new_tokens),
            do_sample=bool(do_sample),
            temperature=eff_temperature,
            top_p=float(top_p),
            on_token=on_token,
            substrate_confidence=confidence,
            substrate_target_snr_scale=float(derived_scale),
        )
        self._record_motor_replay(
            msgs,
            generated_token_ids=gen_ids,
            broca_features=broca_features,
            substrate_confidence=confidence,
            substrate_inertia=sub_inertia,
        )
        assistant_affect = self.affect_encoder.detect(text)
        if self._last_affect is None:
            raise RuntimeError("chat_reply cannot align affect before user affect has been recorded")
        affect_alignment = self.affect_trace.alignment(self._last_affect, assistant_affect)
        assistant_affect_trace_id = self.affect_trace.record(
            role="assistant",
            text=text,
            affect=assistant_affect,
            response_to_id=self._last_user_affect_trace_id,
            alignment=affect_alignment,
        )
        self._last_chat_meta = {
            **self._last_chat_meta,
            "assistant_affect": _affect_evidence(assistant_affect),
            "affect_alignment": affect_alignment,
            "assistant_affect_trace_id": int(assistant_affect_trace_id),
            "user_affect_trace_id": self._last_user_affect_trace_id,
        }
        try:
            self.event_bus.publish(
                "chat.complete",
                {
                    "intent": frame.intent,
                    "confidence": float(confidence),
                    "affect_alignment": float(affect_alignment["alignment"]),
                    "reply_chars": len(text),
                    "reply_preview": text[:200],
                },
            )
        except Exception:
            logger.exception("chat_reply: complete-event publish failed")
        return frame, text

    def _substrate_temperature_scale(self, frame: CognitiveFrame, confidence: float) -> float:
        """Sampling temperature multiplier derived from substrate posterior entropy.

        Couples the LLM's decoding entropy to the active-inference faculty's
        posterior over policies: when the substrate is confused (high
        normalized entropy) the LLM is given headroom to explore; when the
        substrate has collapsed onto a single policy the LLM samples nearly
        greedily so it cannot drift away from the decided answer.
        """

        if frame.intent == "unknown":
            return 1.0
        try:
            coupled = self.unified_agent.decide()
        except (RuntimeError, ValueError, IndexError):
            logger.debug("_substrate_temperature_scale: unified_agent.decide() unavailable")
            return max(1e-3, 1.0 - 0.6 * float(confidence))
        if coupled.faculty == "spatial":
            posterior = list(coupled.spatial_decision.posterior_over_policies)
        else:
            posterior = list(coupled.causal_decision.posterior_over_policies)
        n = len(posterior)
        if n < 2:
            return max(1e-3, 1.0 - 0.6 * float(confidence))
        h_q = belief_entropy(posterior)
        h_max = math.log(n)
        if h_max <= 1e-9:
            return max(1e-3, 1.0 - 0.6 * float(confidence))
        normalized_uncertainty = max(0.0, min(1.0, h_q / h_max))
        # Multiplicatively combine the substrate's posterior entropy with the
        # frame's own confidence so both signals can pull temperature down.
        return max(1e-3, normalized_uncertainty * (1.0 - 0.6 * float(confidence)))

    def _content_logit_bias(self, frame: CognitiveFrame) -> dict[int, float]:
        """Map substrate content (subject / predicate / answer) to subword token ids.

        The numeric value attached to each token is a *base bonus* that the
        :class:`SubstrateLogitBiasGraft` interprets dynamically: it is scaled
        per step by the host's current peakedness, the substrate's confidence,
        and the autoregressive inertia, so callers do not need to guess a
        magnitude that wins against an arbitrary LLM. A unit base bonus is
        therefore the right choice — bias importance comes from the substrate
        frame, not from a hand-tuned scalar.
        """

        if frame.intent == "unknown":
            return {}
        targets: list[str] = []
        if frame.subject:
            targets.append(str(frame.subject))
        if frame.answer and frame.answer.lower() != "unknown":
            targets.append(str(frame.answer))
        pred = (frame.evidence or {}).get("predicate") or (frame.evidence or {}).get("predicate_surface")
        if isinstance(pred, str) and pred:
            targets.append(pred)
        if not targets:
            return {}
        hf_tok = getattr(self.tokenizer, "inner", None)
        bias: dict[int, float] = {}
        for surface in targets:
            surface = surface.strip()
            if not surface:
                continue
            ids: list[int] = []
            if hf_tok is not None and callable(getattr(hf_tok, "encode", None)):
                ids.extend(int(t) for t in hf_tok.encode(surface, add_special_tokens=False))
                ids.extend(int(t) for t in hf_tok.encode(" " + surface, add_special_tokens=False))
            else:
                ids.extend(int(t) for t in self.tokenizer.encode(surface))
            for tid in set(ids):
                if tid < 0:
                    continue
                bias[tid] = max(bias.get(tid, 0.0), 1.0)
        return bias

    def _derived_target_snr_scale(self, frame: CognitiveFrame) -> float:
        """Compose intent / memory / conformal / affect into a graft-strength scale.

        Returns a value in ``[0, 1]`` that the host grafts multiply against
        their static SNR cap. ``0`` means *do not bias the LLM at all*;
        ``1`` means *push as hard as the cap allows*. The scale is derived
        from substrate state, never tuned.
        """

        evidence = frame.evidence or {}
        is_actionable = bool(evidence.get("is_actionable", frame.intent != "unknown"))
        actionability = 1.0 if is_actionable else 0.0
        memory_confidence = max(0.0, min(1.0, float(frame.confidence)))
        conformal_set_size = int(evidence.get("conformal_set_size", 0) or 0)
        certainty = affect_certainty(self._last_affect)
        strength = DerivedStrength.compute(
            StrengthInputs(
                intent_actionability=actionability,
                memory_confidence=memory_confidence,
                conformal_set_size=conformal_set_size,
                affect_certainty=certainty,
            )
        )
        logger.debug(
            "_derived_target_snr_scale: intent=%s actionability=%.1f mem=%.3f |C|=%d affect=%.3f -> scale=%.3f",
            frame.intent,
            actionability,
            memory_confidence,
            conformal_set_size,
            certainty,
            strength,
        )
        return float(strength)

    def _record_motor_replay(
        self,
        messages: Sequence[dict[str, str]],
        *,
        generated_token_ids: Sequence[int],
        broca_features: torch.Tensor | None,
        substrate_confidence: float,
        substrate_inertia: float,
    ) -> None:
        """Append one training target for REM-time :class:`GraftMotorTrainer`."""

        if len(generated_token_ids) == 0:
            return
        cap = DMNConfig().sleep_max_replay
        snap = broca_features.detach().cpu().clone() if broca_features is not None else None

        item: dict[str, Any] = {
            "messages": [dict(m) for m in messages],
            "speech_plan_tokens": torch.tensor(list(generated_token_ids), dtype=torch.long),
            "substrate_confidence": float(substrate_confidence),
            "substrate_inertia": float(substrate_inertia),
        }
        if snap is not None:
            item["broca_features"] = snap
        with self._cognitive_state_lock:
            self.motor_replay.append(item)
            if len(self.motor_replay) > cap:
                self.motor_replay[:] = self.motor_replay[-cap:]

    def _stream_substrate_chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        broca_features: torch.Tensor | None,
        logit_bias: dict[int, float],
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        on_token: Callable[[str], None] | None,
        substrate_confidence: float = 1.0,
        substrate_target_snr_scale: float = 1.0,
    ) -> tuple[str, list[int], float]:
        hf_tok = getattr(self.tokenizer, "inner", None)
        if hf_tok is None or not callable(getattr(hf_tok, "apply_chat_template", None)):
            raise RuntimeError("chat_reply requires a HuggingFace chat-template tokenizer at .tokenizer.inner")

        device = next(self.host.parameters()).device
        prompt = hf_tok.apply_chat_template(list(messages), add_generation_prompt=True, return_tensors="pt")
        if not isinstance(prompt, torch.Tensor):
            prompt = prompt["input_ids"]
        prompt = prompt.to(device)
        if prompt.ndim == 1:
            prompt = prompt.view(1, -1)

        eos_id = getattr(hf_tok, "eos_token_id", None)
        current = prompt[0].tolist()
        generated: list[int] = []
        bias_active = bool(logit_bias)
        feature_tensor = broca_features.to(device) if broca_features is not None else None
        target_token_set = {int(t) for t in logit_bias.keys()} if bias_active else set()
        target_emitted = False

        logger.debug(
            "_stream_substrate_chat: prompt_len=%d max_new_tokens=%d bias_active=%s feature_active=%s confidence=%.3f",
            int(prompt.shape[1]),
            int(max_new_tokens),
            bias_active,
            feature_tensor is not None,
            float(substrate_confidence),
        )
        past_key_values = None
        with torch.no_grad():
            for _step in range(max(1, int(max_new_tokens))):
                # Inertia grows with the autoregressive prefix so the bias and
                # SNR-targeted grafts can shout over a long babbling tail.
                inertia = math.log1p(float(len(current)))
                extra: dict[str, Any] = {
                    "tokenizer": self.tokenizer,
                    "substrate_confidence": float(substrate_confidence),
                    "substrate_inertia": float(inertia),
                    "substrate_target_snr_scale": float(substrate_target_snr_scale),
                    "return_past_key_values": True,
                }
                if feature_tensor is not None:
                    extra["broca_features"] = feature_tensor
                if bias_active:
                    # Semantic decay: full strength until any target subword is
                    # emitted, then fall away so the LLM is free to finish the
                    # reply naturally without being hammered into repeating it.
                    semantic_decay = 0.15 if target_emitted else 1.0
                    extra["broca_logit_bias"] = logit_bias
                    extra["broca_logit_bias_decay"] = semantic_decay
                if past_key_values is not None:
                    extra["past_key_values"] = past_key_values
                    row_t = torch.tensor([[current[-1]]], device=device, dtype=torch.long)
                    mask_t = torch.ones((1, len(current)), dtype=torch.bool, device=device)
                else:
                    row_t = torch.tensor([current], device=device, dtype=torch.long)
                    mask_t = torch.ones_like(row_t, dtype=torch.bool)
                out = self.host(row_t, mask_t, extra_state=extra)
                if isinstance(out, tuple):
                    logits, past_key_values = out
                else:
                    raise RuntimeError("LlamaBrocaHost.forward expected (logits, past_key_values) when return_past_key_values is set")
                last_pos = logits.shape[1] - 1
                logits_row = logits[0, last_pos].float()
                if do_sample:
                    scaled = logits_row / max(temperature, 1e-5)
                    probs = torch.softmax(scaled, dim=-1)
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cdf = torch.cumsum(sorted_probs, dim=-1)
                    over = (cdf > top_p).nonzero(as_tuple=False)
                    keep = int(over[0, 0].item()) + 1 if over.numel() > 0 else int(probs.numel())
                    keep = max(1, keep)
                    kept_probs = sorted_probs[:keep]
                    kept_idx = sorted_idx[:keep]
                    kept_probs = kept_probs / kept_probs.sum().clamp_min(1e-12)
                    pick = int(torch.multinomial(kept_probs, num_samples=1).item())
                    pred = int(kept_idx[pick].item())
                else:
                    pred = int(logits_row.argmax().item())
                if eos_id is not None and pred == int(eos_id):
                    break
                generated.append(pred)
                current.append(pred)
                if bias_active and not target_emitted and pred in target_token_set:
                    target_emitted = True
                if on_token is not None:
                    piece = hf_tok.decode([pred], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    if piece:
                        on_token(piece)
        reply = hf_tok.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        logger.debug("_stream_substrate_chat: emitted_tokens=%d reply_preview=%r", len(generated), reply[:200] if len(reply) > 200 else reply)
        inertia_tail = math.log1p(float(len(current)))
        return reply, generated, inertia_tail
