"""RecursionController — the substrate's r-round latent collaboration loop.

Orchestrates the closed-form recursive computation that ties together every
piece of Phase 0–2 infrastructure:

* Round entry: the comprehension pipeline has already populated SWM with
  per-organ slots (gliner2 hidden, gliclass hidden, structured outputs).

* Substrate algebra (per round): bundle the active organ contributions into
  a single ``recursive.thought.r{i}`` slot — the unified latent thought that
  the LLM will see this round.

* LLM inner loop: :class:`LatentDecoder` runs ``m=40`` latent steps over the
  prompt with the SWM thought injected via :class:`SWMResidualGraft` at the
  designated layer. The graft's slot pointer (``state['swm_inject_slot']``)
  advances each round.

* Round close: write Llama's last hidden state back into SWM as
  ``llama.thought.r{i}``, JL-projected up to D_swm.

* Halt check: :class:`RecursionHalt` decides whether the substrate has
  converged or hit the round cap.

The controller is training-free end-to-end: every projection is closed-form,
every algebraic operator (bind / bundle / unbind / cleanup) lives on the
existing :class:`VSACodebook`. Llama's latent rollout uses the LatentMAS Wₐ
derived from its own embedding matrices.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import torch

from ..calibration.recursion_halt import HaltDecision, RecursionHalt
from ..grafts.swm_residual_graft import (
    ACTIVE_THOUGHT_SLOT,
    SWMResidualGraft,
    SWM_INJECT_SLOT_KEY,
)
from ..host.latent_decoder import LatentDecoder
from ..swm import EncoderSWMPublisher, SWMSource, SubstrateWorkingMemory
from ..workspace import WorkspacePublisher


logger = logging.getLogger(__name__)


RECURSIVE_THOUGHT_SLOT_FMT: str = "recursive.thought.r{round}"
LLAMA_THOUGHT_SLOT_FMT: str = "llama.thought.r{round}"


@dataclass(frozen=True)
class RecursionTrace:
    """Per-round trace of what the controller did."""

    rounds: int
    halts: list[HaltDecision] = field(default_factory=list)
    thought_slots: list[str] = field(default_factory=list)
    llama_slots: list[str] = field(default_factory=list)
    final_thought_slot: str = ""
    final_llama_slot: str = ""


class RecursionController:
    """Drives the r-round substrate ↔ LLM latent collaboration loop."""

    def __init__(
        self,
        *,
        swm: SubstrateWorkingMemory,
        publisher: EncoderSWMPublisher,
        latent_decoder: LatentDecoder,
        residual_graft: SWMResidualGraft,
        halt: RecursionHalt,
    ) -> None:
        self._swm = swm
        self._publisher = publisher
        self._decoder = latent_decoder
        self._graft = residual_graft
        self._halt = halt

    @property
    def swm(self) -> SubstrateWorkingMemory:
        return self._swm

    @property
    def latent_decoder(self) -> LatentDecoder:
        return self._decoder

    @property
    def halt(self) -> RecursionHalt:
        return self._halt

    @torch.no_grad()
    def run(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        extra_state: dict[str, Any] | None = None,
    ) -> RecursionTrace:
        """Run up to ``halt.max_rounds`` rounds; return a :class:`RecursionTrace`."""

        if input_ids.ndim != 2:
            raise ValueError(
                f"RecursionController.run requires input_ids [batch, seq], got {tuple(input_ids.shape)}"
            )

        organ_slot_names = self._collect_organ_slot_names()

        if not organ_slot_names:
            raise RuntimeError(
                "RecursionController.run: no organ slots in SWM — comprehension must populate the workspace before recursion"
            )

        self._halt.reset()
        thought_slots: list[str] = []
        llama_slots: list[str] = []
        halts: list[HaltDecision] = []

        WorkspacePublisher.emit(
            "recursion.run.start",
            {
                "max_rounds": self._halt.max_rounds,
                "m_latent_steps": self._decoder.m_latent_steps,
                "organ_slot_count": len(organ_slot_names),
                "organ_slots": list(organ_slot_names),
            },
        )

        for round_idx in range(self._halt.max_rounds):
            thought_slot = RECURSIVE_THOUGHT_SLOT_FMT.format(round=round_idx)
            llama_slot = LLAMA_THOUGHT_SLOT_FMT.format(round=round_idx)

            sources_for_round = list(organ_slot_names) + (
                [LLAMA_THOUGHT_SLOT_FMT.format(round=round_idx - 1)] if round_idx > 0 else []
            )
            WorkspacePublisher.emit(
                "recursion.round.start",
                {
                    "round": round_idx,
                    "thought_slot": thought_slot,
                    "input_slot_count": len(sources_for_round),
                },
            )
            self._swm.bundle_slots(sources_for_round, into=thought_slot)
            thought_slots.append(thought_slot)

            round_state: dict[str, Any] = {SWM_INJECT_SLOT_KEY: thought_slot}

            if extra_state:
                round_state.update(extra_state)

            last_hidden, _past_kv = self._decoder.think(
                input_ids=input_ids,
                attention_mask=attention_mask,
                extra_state=round_state,
            )

            decision = self._halt.check(slot_name=thought_slot, rounds_completed=round_idx + 1)
            halts.append(decision)

            # Confidence in the rollout = how close the substrate's working memory
            # is to its previous-round state on the cosine axis. Round 0 has no
            # previous (cos = -inf) and so reports 0 confidence — full prediction
            # error, which is the right signal for "this is the rawest hypothesis."
            cos_prev = decision.cosine_to_previous
            llama_confidence = (
                max(0.0, min(1.0, float(cos_prev))) if math.isfinite(cos_prev) else 0.0
            )

            self._publisher.publish_hidden(
                source=SWMSource.LLAMA,
                hidden=last_hidden,
                confidence=llama_confidence,
            )
            self._swm.write(
                llama_slot,
                self._swm.read(EncoderSWMPublisher.slot_name_hidden(SWMSource.LLAMA)).vector,
                source=SWMSource.LLAMA,
            )
            llama_slots.append(llama_slot)

            logger.debug(
                "RecursionController.run: round=%d halt=%s reason=%s cos_prev=%.4f",
                round_idx,
                decision.halt,
                decision.reason,
                decision.cosine_to_previous,
            )

            WorkspacePublisher.emit(
                "recursion.round.complete",
                {
                    "round": round_idx,
                    "halt": decision.halt,
                    "reason": decision.reason,
                    "cosine_to_previous": decision.cosine_to_previous,
                    "rounds_completed": decision.rounds_completed,
                    "thought_slot": thought_slot,
                    "llama_slot": llama_slot,
                },
            )

            if decision.halt:
                break

        if thought_slots:
            final_thought = self._swm.read(thought_slots[-1]).vector
            self._swm.write(ACTIVE_THOUGHT_SLOT, final_thought, source=SWMSource.SUBSTRATE_ALGEBRA)

        WorkspacePublisher.emit(
            "recursion.run.complete",
            {
                "rounds": len(thought_slots),
                "final_thought_slot": thought_slots[-1] if thought_slots else "",
                "final_llama_slot": llama_slots[-1] if llama_slots else "",
                "halt_reason": halts[-1].reason if halts else "no_rounds",
            },
        )

        return RecursionTrace(
            rounds=len(thought_slots),
            halts=halts,
            thought_slots=list(thought_slots),
            llama_slots=list(llama_slots),
            final_thought_slot=thought_slots[-1] if thought_slots else "",
            final_llama_slot=llama_slots[-1] if llama_slots else "",
        )

    def _collect_organ_slot_names(self) -> list[str]:
        """Return the SWM slot names a comprehension turn writes (hidden + structured)."""

        names: list[str] = []

        for source in (SWMSource.GLINER2, SWMSource.GLICLASS):
            for kind in ("hidden", "entities", "relations", "classifications"):
                slot_name = f"{source.value}.{kind}"

                if self._swm.has(slot_name):
                    names.append(slot_name)

        return names
