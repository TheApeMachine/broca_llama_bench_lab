"""CognitiveFrame — the substrate's central non-linguistic content packet.

A frame is what the substrate *means* before it is *said*. Comprehension
produces it. Memory and Reasoning enrich it. Planning finalizes the speech
plan and SNR scale on it. Grafts read it to populate residual-stream and
logit biases on the frozen LLM. The LLM then says it.

A frame has open-vocabulary ``intent`` (e.g. ``memory_write``,
``memory_lookup``, ``causal_effect``, ``active_action``, or any future
substrate-coined label). Subject and answer are open strings. Confidence is
in ``[0, 1]``. Evidence is the open-ended dict that every faculty appends to
as the frame moves through the pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

from .frame_packer import FramePacker


@dataclass
class CognitiveFrame:
    intent: str
    subject: str = ""
    answer: str = "unknown"
    confidence: float = 1.0
    evidence: dict = field(default_factory=dict)

    def speech_plan(self) -> list[str]:
        """The lexical token sequence the LLM should aim to emit.

        Honors an explicit ``evidence["speech_plan_words"]`` override when
        Planning has already finalized the wording. Otherwise lexicalizes
        intent / subject / predicate / answer in canonical order.
        """

        raw_override = self.evidence.get("speech_plan_words")
        if (
            isinstance(raw_override, list)
            and raw_override
            and all(isinstance(x, str) for x in raw_override)
        ):
            return list(raw_override)

        tokens: list[str] = []
        tokens.extend(self._lexical_tokens(self.intent))
        if self.subject:
            tokens.extend(self._lexical_tokens(self.subject))
        predicate = (self.evidence or {}).get("predicate") or (self.evidence or {}).get(
            "predicate_surface"
        )
        if predicate:
            tokens.extend(self._lexical_tokens(predicate))
        if self.answer and self.answer != "unknown":
            tokens.extend(self._lexical_tokens(self.answer))
        claimed = (self.evidence or {}).get("claimed_answer")
        if claimed:
            tokens.extend(self._lexical_tokens(claimed))
        if not tokens:
            tokens.extend(self._lexical_tokens(self.answer))
        return tokens + ["."]

    def descriptor_tokens(self) -> list[str]:
        """All lexical tokens that describe this frame, used by relevance scoring."""

        parts: list[str] = []
        for value in (self.intent, self.subject, self.answer):
            parts.extend(self._lexical_tokens(value))
        for key, value in sorted((self.evidence or {}).items()):
            parts.extend(self._lexical_tokens(key))
            if isinstance(value, (str, int, float, bool)):
                parts.extend(self._lexical_tokens(value))
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    parts.extend(self._lexical_tokens(sub_key))
                    if isinstance(sub_value, (str, int, float, bool)):
                        parts.extend(self._lexical_tokens(sub_value))
        return parts

    def to_features(self, packer: FramePacker | None = None) -> torch.Tensor:
        """Continuous feature vector at width :meth:`FrameDimensions.cognitive_frame_dim`.

        Pass an explicitly-configured :class:`FramePacker` to use a host-driven
        :class:`EmbeddingProjector`; the default constructs one with the
        lexical-only :class:`SubwordProjector`.
        """

        return (packer or FramePacker()).cognitive(
            self.intent,
            self.subject,
            self.answer,
            float(self.confidence),
            self.evidence,
        )

    def to_broca_features(
        self,
        packer: FramePacker | None = None,
        *,
        vsa_bundle: torch.Tensor | None = None,
        vsa_projection_seed: int = 0,
    ) -> torch.Tensor:
        """Continuous Broca-graft feature vector at width :meth:`FrameDimensions.broca_feature_dim`."""

        return (packer or FramePacker()).broca(
            self.intent,
            self.subject,
            self.answer,
            float(self.confidence),
            self.evidence,
            vsa_bundle=vsa_bundle,
            vsa_projection_seed=vsa_projection_seed,
        )

    @classmethod
    def from_episode_row(cls, row: dict) -> "CognitiveFrame":
        """Rehydrate a frame previously persisted to the workspace journal.

        Tags ``evidence`` with ``episodic_retrieval`` so downstream consumers
        know the frame is replayed history rather than a fresh perception.
        """

        ev = dict(row["evidence"])
        ev["retrieved_episode_id"] = row["id"]
        ev["episode_original_ts"] = row["ts"]
        inst = list(ev.get("instruments") or [])
        if "episodic_retrieval" not in inst:
            inst.append("episodic_retrieval")
        return cls(
            row["intent"],
            subject=row["subject"],
            answer=row["answer"],
            confidence=float(row["confidence"]),
            evidence=ev,
        )

    @classmethod
    def synthesize_bundle(cls, working: list["CognitiveFrame"]) -> "CognitiveFrame | None":
        """Bind a recent ``memory_*`` and ``causal_effect`` frame into a synthesis frame.

        Returns ``None`` when the working memory does not currently hold both
        kinds of evidence, or when the most recent six frames already include
        a synthesis bundle (no immediate re-binding).
        """

        if len(working) < 2:
            return None
        if any(f.intent == "synthesis_bundle" for f in working[-6:]):
            return None
        mem = None
        ce = None
        for f in reversed(working):
            if (
                mem is None
                and f.intent.startswith("memory_")
                and f.intent not in {"memory_write", "memory_conflict"}
            ):
                mem = f
            if ce is None and f.intent == "causal_effect":
                ce = f
            if mem is not None and ce is not None:
                break
        if mem is None or ce is None:
            return None
        jids: list[int] = []
        for f in (mem, ce):
            jid = (f.evidence or {}).get("journal_id")
            if jid is not None:
                jids.append(int(jid))
        geo = math.sqrt(
            max(1e-12, float(mem.confidence)) * max(1e-12, float(ce.confidence))
        )
        return cls(
            "synthesis_bundle",
            subject=mem.subject,
            answer=mem.answer,
            confidence=min(1.0, geo),
            evidence={
                "episode_ids": jids,
                "instruments": [
                    "working_memory_synthesis",
                    "semantic_memory",
                    "scm_do_readout",
                ],
                "predicate": (mem.evidence or {}).get("predicate", ""),
                "causal_ate": ce.evidence.get("ate"),
                "source_intents": [mem.intent, ce.intent],
            },
        )

    @staticmethod
    def _lexical_tokens(value: Any) -> list[str]:
        text = str(value).replace("_", " ").strip().lower()
        return [t for t in text.split() if any(ch.isalnum() for ch in t)]
