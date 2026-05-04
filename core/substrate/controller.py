"""SubstrateController — composition root for the cognitive substrate.

Faculties are constructed by :class:`SubstrateBuilder`. Orchestration façades
live on :attr:`runtime` (:class:`~core.substrate.facades.SubstrateRuntime`).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import torch

from core.cognition.intent_gate import UtteranceIntent
from core.cognition.observation import CognitiveObservation
from core.comprehension.deferred_relation_ingest import DeferredRelationIngest
from core.dmn.background_worker import CognitiveBackgroundWorker
from core.dmn.config import DMNConfig
from core.encoders.affect import AffectState
from core.frame import CognitiveFrame, ParsedClaim
from core.grafting.dynamic_grafts import CapturedActivationMode
from core.host.hf_tokenizer_compat import HuggingFaceBrocaTokenizer
from core.host.llama_broca_host import LlamaBrocaHost
from core.idletime.chunking import CompiledMacro
from core.natives.native_tools import NativeTool

from ..numeric import Probability
from .facades import SubstrateRuntime


logger = logging.getLogger(__name__)


class SubstrateController:
    """Cognitive substrate with the language model demoted to speech interface."""

    host: LlamaBrocaHost
    tokenizer: HuggingFaceBrocaTokenizer
    runtime: SubstrateRuntime
    probability = Probability()

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
        return self.session.background_worker

    @property
    def _self_improve_worker(self) -> Any:
        """Compatibility view for older callers; lifecycle state lives in session."""

        return self.session.self_improve_worker

    @_self_improve_worker.setter
    def _self_improve_worker(self, worker: Any) -> None:
        self.session.self_improve_worker = worker

    def deferred_relation_ingest_online(self) -> bool:
        return self.runtime.deferred_relations.is_online()

    def deferred_relation_ingest_count(self) -> int:
        return self.runtime.deferred_relations.count()

    def _enqueue_deferred_relation_ingest(
        self,
        utterance: str,
        toks: Sequence[str],
        intent: UtteranceIntent,
        *,
        journal_id: int,
    ) -> DeferredRelationIngest:
        return self.runtime.deferred_relations.enqueue(
            utterance, toks, intent, journal_id=journal_id
        )

    def process_deferred_relation_ingest(self) -> list[dict[str, Any]]:
        return self.runtime.deferred_relations.process_all()

    def consolidate_once(self) -> list[dict]:
        out = self.memory.consolidate_claims_once()
        logger.debug("SubstrateController.consolidate_once: reflections=%d", len(out))
        self.event_bus.publish("consolidation", {"reflections": len(out)})
        return out

    def snapshot(self) -> dict[str, Any]:
        return self.runtime.inspector.snapshot()

    def _sync_preference_to_pomdp(self) -> None:
        self.runtime.preference.sync_to_pomdp()

    def observe_user_feedback(self, **kwargs: Any) -> None:
        self.runtime.preference.observe_user_feedback(**kwargs)

    def observe_event(self, channel: str, *, t: float | None = None) -> None:
        self.runtime.preference.observe_event(channel, t=t)

    def encode_triple_vsa(self, subject: str, predicate: str, obj: str) -> torch.Tensor:
        return self.runtime.algebra.encode_triple(subject, predicate, obj)

    def _padded_hopfield_sketch(self, sketch: torch.Tensor) -> torch.Tensor:
        return self.runtime.algebra.padded_hopfield_sketch(sketch)

    def remember_hopfield(
        self,
        a_sketch: torch.Tensor,
        b_sketch: torch.Tensor,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.runtime.algebra.remember(a_sketch, b_sketch, metadata=metadata)

    def _after_frame_commit(self, out: CognitiveFrame, utterance: str, *, event_topic: str) -> None:
        self.runtime.comprehension.after_frame_commit(out, utterance, event_topic=event_topic)

    def _observe_frame_concepts(self, out: CognitiveFrame) -> None:
        self.runtime.comprehension.observe_frame_concepts(out)

    def _remember_declarative_binding(self, out: CognitiveFrame, utterance: str) -> None:
        self.runtime.comprehension.remember_declarative_binding(out, utterance)

    def _frame_from_observation(self, observation: CognitiveObservation) -> CognitiveFrame:
        from ..comprehension.pipeline import ComprehensionPipeline

        return ComprehensionPipeline.frame_from_observation(observation)

    def _commit_observation(self, observation: CognitiveObservation) -> CognitiveFrame:
        return self.runtime.comprehension.commit_observation(observation)

    def perceive_image(self, image: Any, *, source: str = "image") -> CognitiveFrame:
        return self.runtime.comprehension.perceive_image(image, source=source)

    def perceive_video(self, frames: Any, *, source: str = "video") -> CognitiveFrame:
        return self.runtime.comprehension.perceive_video(frames, source=source)

    def perceive_audio(
        self,
        audio: Any,
        *,
        sampling_rate: int = 16000,
        source: str = "audio",
        language: str | None = None,
    ) -> CognitiveFrame:
        return self.runtime.comprehension.perceive_audio(
            audio, sampling_rate=sampling_rate, source=source, language=language
        )

    def broca_features_from_frame(self, frame: CognitiveFrame) -> torch.Tensor:
        return self.runtime.graft_frame.broca_features(frame)

    def concept_token_ids_from_frame(self, frame: CognitiveFrame) -> dict[str, list[int]]:
        return self.runtime.graft_frame.concept_token_ids(frame)

    def repulsion_token_ids_from_frame(self, frame: CognitiveFrame) -> dict[str, list[int]]:
        return self.runtime.graft_frame.repulsion_token_ids(frame)

    def refine_extracted_claim(
        self, utterance: str, toks: Sequence[str], claim: ParsedClaim
    ) -> ParsedClaim:
        return self.runtime.claims.refine(utterance, toks, claim)

    def _handle_native_tool_drift(self, tool: NativeTool, evidence: Mapping[str, Any]) -> None:
        self.runtime.native_tools.handle_drift(tool, evidence)

    def synthesize_native_tool(self, *args: Any, **kwargs: Any) -> NativeTool:
        return self.runtime.native_tools.synthesize(*args, **kwargs)

    def attach_tools_to_scm(self) -> int:
        return self.runtime.native_tools.attach_to_scm()

    def should_synthesize_tool(self) -> bool:
        return self.runtime.native_tools.should_synthesize()

    def recent_intents(self, *, limit: int = 8) -> list[str]:
        return self.runtime.macros.recent_intents(limit=limit)

    def find_matching_macro(
        self,
        *,
        recent_intents: Sequence[str] | None = None,
        features: torch.Tensor | None = None,
    ) -> CompiledMacro | None:
        return self.runtime.macros.find_matching(
            recent_intents=recent_intents, features=features
        )

    def macro_speech_features(self, macro: CompiledMacro) -> torch.Tensor:
        from ..idletime.macro_adapter import MacroAdapter

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
        return self.runtime.algebra.vector_for_concept(name, base_sketch=base_sketch)

    def start_background(
        self,
        *,
        interval_s: float = 5.0,
        config: DMNConfig | None = None,
    ) -> CognitiveBackgroundWorker:
        return self.runtime.workers.start_background(
            interval_s=interval_s, config=config
        )

    def stop_background(self) -> None:
        self.runtime.workers.stop_background()

    def start_self_improve_worker(
        self,
        *,
        interval_s: float | None = None,
        enabled: bool | None = None,
    ) -> Any:
        return self.runtime.workers.start_self_improve(
            interval_s=interval_s, enabled=enabled
        )

    def stop_self_improve_worker(self, timeout: float = 5.0) -> None:
        self.runtime.workers.stop_self_improve(timeout=timeout)

    def _intrinsic_scan(self, toks: list[str]) -> None:
        self.runtime.comprehension.intrinsic_scan(toks)

    def _non_actionable_frame(self, intent: UtteranceIntent, affect: AffectState) -> CognitiveFrame:
        from ..comprehension.pipeline import ComprehensionPipeline

        return ComprehensionPipeline.non_actionable_frame(intent, affect)

    def _attach_perception(self, frame: CognitiveFrame, intent: UtteranceIntent, affect: AffectState) -> None:
        from ..comprehension.pipeline import ComprehensionPipeline

        ComprehensionPipeline.attach_perception(frame, intent, affect)

    def comprehend(self, utterance: str) -> CognitiveFrame:
        return self.runtime.comprehension.comprehend(utterance)

    def _perceive_utterance(self, utterance: str) -> tuple[UtteranceIntent, AffectState]:
        return self.runtime.comprehension.perceive_utterance(utterance)

    def _commit_frame(self, utterance: str, toks: Sequence[str], frame: CognitiveFrame) -> CognitiveFrame:
        return self.runtime.comprehension.commit_frame(utterance, toks, frame)

    def retrieve_episode(self, episode_id: int) -> CognitiveFrame:
        """Reload a prior workspace episode into working memory (persistent episodic retrieval)."""

        row = self.journal.fetch(episode_id)
        if row is None:
            raise ValueError(f"retrieve_episode: missing journal row for episode_id={episode_id!r}")
        replay = CognitiveFrame.from_episode_row(row)
        self.workspace.post_frame(replay)
        logger.debug("retrieve_episode: id=%s intent=%s", episode_id, replay.intent)
        return replay

    def speak(self, frame: CognitiveFrame) -> str:
        plan_words = frame.speech_plan()
        broca_features = self.broca_features_from_frame(frame)
        from ..generation import PlanForcedGenerator

        text, token_ids, inertia = PlanForcedGenerator.generate(
            self.host,
            self.tokenizer,
            plan_words,
            broca_features=broca_features,
        )
        self.motor_replay_recorder.record(
            [
                {
                    "role": "user",
                    "content": (
                        f"intent={frame.intent} | subject={frame.subject or ''} | "
                        f"answer={frame.answer or ''} | plan={' '.join(plan_words)}"
                    ),
                }
            ],
            generated_token_ids=token_ids,
            broca_features=broca_features,
            substrate_confidence=self.probability.unit_interval(frame.confidence),
            substrate_inertia=inertia,
        )
        return text

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
        """Substrate-biased free-form chat reply."""

        return self.runtime.chat.run(
            messages,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            on_token=on_token,
        )
