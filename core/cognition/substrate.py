"""SubstrateController — composition root for the cognitive substrate.

The controller holds the per-faculty objects (memory, host, grafts, encoders,
SCM, agents, …) that :class:`SubstrateBuilder` constructs at boot, and
exposes the substrate's public surface as a chain of delegations to the
manager classes that own each concern.

Each method on this class is a thin shim over the actual implementation in:

* :mod:`.builder` — construction of every faculty
* :mod:`.chat_orchestrator` — substrate-biased chat reply
* :mod:`.comprehension_pipeline` — utterance → frame
* :mod:`.plan_speaker` — plan-forced surface generation
* :mod:`.algebraic_adapter` — VSA / Hopfield / ontology
* :mod:`.preference_adapter` — Dirichlet preferences + Hawkes events
* :mod:`.native_tool_manager` — synthesized SCM equations
* :mod:`.macro_adapter` — proceduralized motif lookup
* :mod:`.deferred_relation_queue` — DMN-side claim parsing
* :mod:`.claim_refiner` — VSA/Hopfield-polished claims
* :mod:`.graft_feature_adapter` — frame → graft inputs
* :mod:`.worker_supervisor` — DMN + self-improve daemons
* :mod:`.substrate_inspector` — JSON snapshot for live UIs
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import torch

from ..comprehension import DeferredRelationIngest, LexicalTokens, TextRelevance
from ..dmn import CognitiveBackgroundWorker, DMNConfig
from ..frame import CognitiveFrame, ParsedClaim
from ..grafting.dynamic_grafts import CapturedActivationMode
from ..host.hf_tokenizer_compat import HuggingFaceBrocaTokenizer
from ..host.llama_broca_host import LlamaBrocaHost, load_llama_broca_host
from ..idletime.chunking import CompiledMacro
from ..natives.native_tools import NativeTool
from .intent_gate import UtteranceIntent
from .observation import CognitiveObservation


logger = logging.getLogger(__name__)


# Public function shims used by the rest of the codebase. Each one is one line
# and points at the canonical implementation in the comprehension package.

def _word_tokens(toks):
    return LexicalTokens.words(toks)


def _lexical_tokens(value):
    return LexicalTokens.lexical(value)


def _frame_relevance(utterance, toks, frame, text_encoder):
    return TextRelevance.frame(utterance, toks, frame, text_encoder)




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
        from .builder import SubstrateBuilder

        SubstrateBuilder.populate(
            self,
            seed=seed,
            db_path=db_path,
            namespace=namespace,
            llama_model_id=llama_model_id,
            device=device,
            hf_token=hf_token,
            lexical_target_snr=lexical_target_snr,
            preload_host_tokenizer=preload_host_tokenizer,
        )

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
        from .deferred_relation_queue import DeferredRelationQueue

        return DeferredRelationQueue(self).is_online()

    def deferred_relation_ingest_count(self) -> int:
        from .deferred_relation_queue import DeferredRelationQueue

        return DeferredRelationQueue(self).count()

    def _enqueue_deferred_relation_ingest(
        self,
        utterance: str,
        toks: Sequence[str],
        intent: UtteranceIntent,
        *,
        journal_id: int,
    ) -> DeferredRelationIngest:
        from .deferred_relation_queue import DeferredRelationQueue

        return DeferredRelationQueue(self).enqueue(
            utterance, toks, intent, journal_id=journal_id
        )

    def process_deferred_relation_ingest(self) -> list[dict[str, Any]]:
        from .deferred_relation_queue import DeferredRelationQueue

        return DeferredRelationQueue(self).process_all()

    def consolidate_once(self) -> list[dict]:
        out = self.memory.consolidate_claims_once()
        logger.debug("SubstrateController.consolidate_once: reflections=%d", len(out))
        try:
            self.event_bus.publish("consolidation", {"reflections": len(out)})
        except Exception:
            logger.exception("SubstrateController.consolidate_once: event publish failed")
        return out

    def snapshot(self) -> dict[str, Any]:
        from .substrate_inspector import SubstrateInspector

        return SubstrateInspector(self).snapshot()

    # -- New substrate plumbing -----------------------------------------------

    def _sync_preference_to_pomdp(self) -> None:
        from .preference_adapter import PreferenceAdapter

        PreferenceAdapter(self).sync_to_pomdp()

    def observe_user_feedback(self, **kwargs: Any) -> None:
        from .preference_adapter import PreferenceAdapter

        PreferenceAdapter(self).observe_user_feedback(**kwargs)

    def observe_event(self, channel: str, *, t: float | None = None) -> None:
        from .preference_adapter import PreferenceAdapter

        PreferenceAdapter(self).observe_event(channel, t=t)

    def encode_triple_vsa(self, subject: str, predicate: str, obj: str) -> torch.Tensor:
        from .algebraic_adapter import AlgebraicMemoryAdapter

        return AlgebraicMemoryAdapter(self).encode_triple(subject, predicate, obj)

    def _padded_hopfield_sketch(self, sketch: torch.Tensor) -> torch.Tensor:
        from .algebraic_adapter import AlgebraicMemoryAdapter

        return AlgebraicMemoryAdapter(self).padded_hopfield_sketch(sketch)

    def remember_hopfield(
        self,
        a_sketch: torch.Tensor,
        b_sketch: torch.Tensor,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        from .algebraic_adapter import AlgebraicMemoryAdapter

        AlgebraicMemoryAdapter(self).remember(a_sketch, b_sketch, metadata=metadata)

    def _after_frame_commit(self, out: CognitiveFrame, utterance: str, *, event_topic: str) -> None:
        from .comprehension_pipeline import ComprehensionPipeline

        ComprehensionPipeline(self).after_frame_commit(out, utterance, event_topic=event_topic)

    def _observe_frame_concepts(self, out: CognitiveFrame) -> None:
        from .comprehension_pipeline import ComprehensionPipeline

        ComprehensionPipeline(self).observe_frame_concepts(out)

    def _remember_declarative_binding(self, out: CognitiveFrame, utterance: str) -> None:
        from .comprehension_pipeline import ComprehensionPipeline

        ComprehensionPipeline(self).remember_declarative_binding(out, utterance)

    def _frame_from_observation(self, observation: CognitiveObservation) -> CognitiveFrame:
        from .comprehension_pipeline import ComprehensionPipeline

        return ComprehensionPipeline.frame_from_observation(observation)

    def _commit_observation(self, observation: CognitiveObservation) -> CognitiveFrame:
        from .comprehension_pipeline import ComprehensionPipeline

        return ComprehensionPipeline(self).commit_observation(observation)

    def perceive_image(self, image: Any, *, source: str = "image") -> CognitiveFrame:
        from .comprehension_pipeline import ComprehensionPipeline

        return ComprehensionPipeline(self).perceive_image(image, source=source)

    def perceive_video(self, frames: Any, *, source: str = "video") -> CognitiveFrame:
        from .comprehension_pipeline import ComprehensionPipeline

        return ComprehensionPipeline(self).perceive_video(frames, source=source)

    def perceive_audio(
        self,
        audio: Any,
        *,
        sampling_rate: int = 16000,
        source: str = "audio",
        language: str | None = None,
    ) -> CognitiveFrame:
        from .comprehension_pipeline import ComprehensionPipeline

        return ComprehensionPipeline(self).perceive_audio(
            audio, sampling_rate=sampling_rate, source=source, language=language
        )

    def broca_features_from_frame(self, frame: CognitiveFrame) -> torch.Tensor:
        from .graft_feature_adapter import GraftFeatureAdapter

        return GraftFeatureAdapter(self).broca_features(frame)

    def content_logit_bias_from_frame(self, frame: CognitiveFrame) -> dict[int, float]:
        from .graft_feature_adapter import GraftFeatureAdapter

        return GraftFeatureAdapter(self).content_logit_bias(frame)

    def refine_extracted_claim(
        self, utterance: str, toks: Sequence[str], claim: ParsedClaim
    ) -> ParsedClaim:
        from .claim_refiner import ClaimRefiner

        return ClaimRefiner(self).refine(utterance, toks, claim)

    # -- Native tool synthesis (delegates to NativeToolManager) -----------------

    def _handle_native_tool_drift(self, tool: NativeTool, evidence: Mapping[str, Any]) -> None:
        from .native_tool_manager import NativeToolManager

        NativeToolManager(self).handle_drift(tool, evidence)

    def synthesize_native_tool(self, *args: Any, **kwargs: Any) -> NativeTool:
        from .native_tool_manager import NativeToolManager

        return NativeToolManager(self).synthesize(*args, **kwargs)

    def attach_tools_to_scm(self) -> int:
        from .native_tool_manager import NativeToolManager

        return NativeToolManager(self).attach_to_scm()

    def should_synthesize_tool(self) -> bool:
        from .native_tool_manager import NativeToolManager

        return NativeToolManager(self).should_synthesize()

    def recent_intents(self, *, limit: int = 8) -> list[str]:
        from .macro_adapter import MacroAdapter

        return MacroAdapter(self).recent_intents(limit=limit)

    def find_matching_macro(
        self,
        *,
        recent_intents: Sequence[str] | None = None,
        features: torch.Tensor | None = None,
    ) -> CompiledMacro | None:
        from .macro_adapter import MacroAdapter

        return MacroAdapter(self).find_matching(
            recent_intents=recent_intents, features=features
        )

    def macro_speech_features(self, macro: CompiledMacro) -> torch.Tensor:
        from .macro_adapter import MacroAdapter

        return MacroAdapter.speech_features(macro)

    def synthesize_activation_mode(self, **kwargs: Any) -> CapturedActivationMode:
        return self.dynamic_graft_synth.synthesize(
            self.host, self.tokenizer, **kwargs
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
        from .algebraic_adapter import AlgebraicMemoryAdapter

        return AlgebraicMemoryAdapter(self).vector_for_concept(name, base_sketch=base_sketch)

    def start_background(
        self,
        *,
        interval_s: float = 5.0,
        config: DMNConfig | None = None,
    ) -> CognitiveBackgroundWorker:
        from .worker_supervisor import WorkerSupervisor

        return WorkerSupervisor(self).start_background(
            interval_s=interval_s, config=config
        )

    def stop_background(self) -> None:
        from .worker_supervisor import WorkerSupervisor

        WorkerSupervisor(self).stop_background()

    def start_self_improve_worker(
        self,
        *,
        interval_s: float | None = None,
        enabled: bool | None = None,
    ) -> Any:
        from .worker_supervisor import WorkerSupervisor

        return WorkerSupervisor(self).start_self_improve(
            interval_s=interval_s, enabled=enabled
        )

    def stop_self_improve_worker(self, timeout: float = 5.0) -> None:
        from .worker_supervisor import WorkerSupervisor

        WorkerSupervisor(self).stop_self_improve(timeout=timeout)

    def _intrinsic_scan(self, toks: list[str]) -> None:
        from .comprehension_pipeline import ComprehensionPipeline

        ComprehensionPipeline(self).intrinsic_scan(toks)

    def _non_actionable_frame(self, intent: UtteranceIntent, affect: AffectState) -> "CognitiveFrame":
        from .comprehension_pipeline import ComprehensionPipeline

        return ComprehensionPipeline.non_actionable_frame(intent, affect)

    def _attach_perception(self, frame: "CognitiveFrame", intent: UtteranceIntent, affect: AffectState) -> None:
        from .comprehension_pipeline import ComprehensionPipeline

        ComprehensionPipeline.attach_perception(frame, intent, affect)

    def comprehend(self, utterance: str) -> CognitiveFrame:
        from .comprehension_pipeline import ComprehensionPipeline

        return ComprehensionPipeline(self).comprehend(utterance)

    def _perceive_utterance(self, utterance: str) -> tuple[UtteranceIntent, AffectState]:
        from .comprehension_pipeline import ComprehensionPipeline

        return ComprehensionPipeline(self).perceive_utterance(utterance)

    def _commit_frame(self, utterance: str, toks: Sequence[str], frame: CognitiveFrame) -> CognitiveFrame:
        from .comprehension_pipeline import ComprehensionPipeline

        return ComprehensionPipeline(self).commit_frame(utterance, toks, frame)

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
        from .plan_speaker import PlanSpeaker

        return PlanSpeaker(self).speak(frame)

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
        """Substrate-biased free-form chat reply; delegates to ChatOrchestrator."""

        from .chat_orchestrator import ChatOrchestrator

        return ChatOrchestrator(self).run(
            messages,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            on_token=on_token,
        )

    # Thin pass-throughs the test suite reaches for directly. These are
    # implementation details of ``ChatOrchestrator`` exposed on the controller
    # so existing call sites keep working until the test surface is rewritten.

    def _derived_target_snr_scale(self, frame: CognitiveFrame) -> float:
        from .chat_orchestrator import ChatOrchestrator

        return ChatOrchestrator(self)._derived_target_snr_scale(frame)

    def _substrate_temperature_scale(self, frame: CognitiveFrame, confidence: float) -> float:
        from .chat_orchestrator import ChatOrchestrator

        return ChatOrchestrator(self)._substrate_temperature_scale(frame, confidence)

    def _content_logit_bias(self, frame: CognitiveFrame) -> dict[int, float]:
        from .chat_orchestrator import ChatOrchestrator

        return ChatOrchestrator(self)._content_logit_bias(frame)

    def _record_motor_replay(self, *args: Any, **kwargs: Any) -> None:
        from .chat_orchestrator import ChatOrchestrator

        return ChatOrchestrator(self)._record_motor_replay(*args, **kwargs)

