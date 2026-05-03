"""Top-down cognitive control over a frozen LLM (System 1) by the Substrate (System 2).

Implements four predictive-coding mechanisms that let an external substrate exert
control over a frozen language model's generation, treating the LLM as the
associative cortex and the substrate as the strict frontal cortex:

1.  :class:`HypothesisMaskingGraft` — negative logit bias that physically blocks
    rejected hypothesis tokens, paired with :class:`IterativeHypothesisSearch`
    which iteratively prunes the search space until the LLM produces a
    hypothesis that survives evaluation.

2.  :class:`EpistemicInterruptionMonitor` — streaming generator that runs an
    evaluator every ``check_every`` tokens and, on a logical violation,
    truncates the last K tokens and re-injects a high-magnitude "re-evaluate"
    feature vector via the :class:`TrainableFeatureGraft` channel
    (``broca_features``).

3.  :class:`ModalityShiftGraft` — continuous bias toward a named cognitive
    "mood" (analytical, fluent, etc.) by injecting a unit-norm direction into
    the residual stream at every step.

4.  :class:`CausalConstraintGraft` — KV-memory graft pre-loaded with
    do-calculus facts (``P(Y|do(T=t))``) where keys are concept directions and
    values are probability-weighted blends of outcome token directions, so the
    LLM is invisibly pulled toward the SCM's verdict whenever it attends to the
    cause concept.

The grafts plug into the existing :class:`LlamaBrocaHost` slot system
(``logits``, ``final_hidden``, ``layer.{i}.post``); the orchestrators
(``IterativeHypothesisSearch``, ``EpistemicInterruptionMonitor``) drive the host
forward loop and update graft state between calls.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..grafting.grafts import (
    BaseGraft,
    KVMemoryGraft,
    snr_magnitude,
    state_confidence,
    state_inertia,
)
from ..workspace import WorkspacePublisher


logger = logging.getLogger(__name__)


def _infer_host_device(host: Any) -> torch.device:
    """Resolve a training device without assuming ``host`` has parameters."""

    p = next(host.parameters(), None)
    if p is not None:
        return p.device
    dv = getattr(host, "device", None)
    if isinstance(dv, torch.device):
        return dv
    if isinstance(dv, str) and dv:
        try:
            return torch.device(dv)
        except (TypeError, ValueError):
            pass
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# 1. Hypothesis Masking — Negative Logit Bias
# ---------------------------------------------------------------------------


class HypothesisMaskingGraft(BaseGraft):
    """Logits-slot graft that physically blocks rejected hypothesis tokens.

    The substrate populates the banned set via :meth:`ban`. Each banned token
    has its logit at the last position reduced by ``penalty`` nats so the LLM
    is mathematically forced to look for a different hypothesis.

    Reads from ``state``:

    *   ``broca_negative_bias`` — optional ``Mapping[int, float]`` merged with
        the persistent banned set. Lets per-call bans (e.g., from a controller
        that doesn't own the graft) coexist with sticky bans set via
        :meth:`ban`.

    The graft writes a debug record of every ban/unban call into
    :attr:`history`, which is invaluable when post-mortem'ing why the LLM
    landed where it did.
    """

    def __init__(
        self,
        *,
        default_penalty: float = 100.0,
        mixer_priority: float = 1.5,
    ):
        super().__init__()
        if default_penalty < 0:
            raise ValueError("default_penalty must be non-negative (penalty is subtracted)")
        self.default_penalty = float(default_penalty)
        self.mixer_priority = float(mixer_priority)
        self.banned: dict[int, float] = {}
        self.history: list[dict[str, Any]] = []

    # -- ban/unban API ------------------------------------------------------

    def ban(
        self,
        token_ids: Sequence[int],
        *,
        penalty: float | None = None,
        reason: str = "",
    ) -> None:
        """Add token ids to the banned set; ``penalty`` defaults to :attr:`default_penalty`.

        Re-banning a token raises its penalty to the maximum of the existing
        and incoming values, so a stronger ban from a later evaluator never
        regresses to a weaker one.
        """

        p = float(self.default_penalty if penalty is None else penalty)
        if p < 0:
            raise ValueError("penalty must be non-negative")
        added: list[int] = []
        for tid in token_ids:
            tid_int = int(tid)
            if tid_int < 0:
                logger.debug(
                    "HypothesisMaskingGraft.ban: skipping negative token id=%r reason=%r",
                    tid,
                    reason,
                )
                continue
            self.banned[tid_int] = max(self.banned.get(tid_int, 0.0), p)
            added.append(tid_int)
        if added:
            self.history.append(
                {"action": "ban", "tokens": added, "penalty": p, "reason": str(reason)}
            )
            logger.debug(
                "HypothesisMaskingGraft.ban: tokens=%s penalty=%.3f reason=%r total_banned=%d",
                added,
                p,
                reason,
                len(self.banned),
            )
            WorkspacePublisher.emit(
                "cog.hypothesis.ban",
                {
                    "tokens": added,
                    "penalty": p,
                    "reason": str(reason),
                    "total_banned": len(self.banned),
                },
            )

    def unban(self, token_ids: Sequence[int]) -> None:
        """Remove specific token ids from the banned set."""

        removed: list[int] = []
        for tid in token_ids:
            if int(tid) in self.banned:
                del self.banned[int(tid)]
                removed.append(int(tid))
        if removed:
            self.history.append({"action": "unban", "tokens": removed})

    def clear(self) -> None:
        """Remove every banned token but keep the history trace."""

        if self.banned:
            self.history.append({"action": "clear", "tokens": sorted(self.banned)})
            self.banned.clear()

    # -- forward ------------------------------------------------------------

    def forward(self, x: torch.Tensor, state: dict) -> torch.Tensor:
        if not self.enabled:
            return x
        extra = state.get("broca_negative_bias")
        if not self.banned and not extra:
            return x
        merged: dict[int, float] = dict(self.banned)
        if extra:
            for tid, p in dict(extra).items():
                tid_int = int(tid)
                merged[tid_int] = max(merged.get(tid_int, 0.0), float(p))
        if not merged:
            return x
        last_raw = state.get("last_indices")
        if last_raw is None:
            logger.warning(
                "HypothesisMaskingGraft: missing state key 'last_indices'; skipping logits penalty"
            )
            return x
        last = last_raw.to(x.device).long()
        rows = torch.arange(x.shape[0], device=x.device)
        out = x.clone()
        vocab = out.shape[-1]
        tids = []
        pens = []
        for tid, penalty in merged.items():
            tid_int = int(tid)
            if tid_int < 0 or tid_int >= vocab:
                continue
            tids.append(tid_int)
            pens.append(float(penalty))
        if not tids:
            return x
        tid_tensor = torch.tensor(tids, device=x.device, dtype=torch.long)
        pen_tensor = torch.tensor(pens, device=x.device, dtype=out.dtype)
        ncol = tid_tensor.numel()
        br = rows.unsqueeze(1).expand(-1, ncol).reshape(-1)
        lc = last.unsqueeze(1).expand(-1, ncol).reshape(-1)
        tk = tid_tensor.unsqueeze(0).expand(rows.shape[0], -1).reshape(-1)
        pv = pen_tensor.unsqueeze(0).expand(rows.shape[0], -1).reshape(-1)
        out[br, lc, tk] = out[br, lc, tk] - pv
        return out


@dataclass
class HypothesisVerdict:
    """Verdict returned by an evaluator about a generated hypothesis.

    ``valid`` is ``True`` if the hypothesis survives logical/mathematical
    evaluation; ``False`` otherwise. When ``valid`` is ``False``, ``ban_tokens``
    enumerates the tokens that should be physically blocked from appearing in
    future hypotheses (e.g., the wrong digit, the misleading verb).
    """

    valid: bool
    ban_tokens: tuple[int, ...] = ()
    reason: str = ""


@dataclass
class HypothesisAttempt:
    iteration: int
    token_ids: list[int]
    text: str
    verdict: HypothesisVerdict


@dataclass
class HypothesisSearchResult:
    """Outcome of an :class:`IterativeHypothesisSearch` run."""

    accepted: bool
    iterations: int
    final_token_ids: list[int]
    final_text: str
    history: list[HypothesisAttempt] = field(default_factory=list)


class IterativeHypothesisSearch:
    """Generate–evaluate–ban–retry loop driven by :class:`HypothesisMaskingGraft`.

    The search owns nothing except references to the host, tokenizer, and
    masking graft; it does not mutate other grafts.     Each iteration:

    1.  The masking graft's banned set is *not* cleared between iterations —
        that's the entire point of the search: every rejected hypothesis prunes
        the search space for the next one.
    2.  Generates ``hypothesis_max_tokens`` tokens autoregressively by calling
        ``host.forward`` (so any logits-slot grafts, including the masking
        graft, are honored).
    3.  Decodes the generated tokens and asks the evaluator for a verdict.
    4.  On rejection: bans the verdict's ``ban_tokens`` and (optionally) the
        first generated token if no explicit bans are supplied. Loops.
    5.  On acceptance: returns immediately.

    The search refuses to run if the masking graft is not actually attached to
    the host's ``logits`` slot — silent fallthrough would defeat the entire
    mechanism.
    """

    def __init__(
        self,
        host: Any,
        tokenizer: Any,
        masking_graft: HypothesisMaskingGraft,
        *,
        max_iterations: int = 5,
        hypothesis_max_tokens: int = 1,
        require_attached: bool = True,
    ):
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if hypothesis_max_tokens < 1:
            raise ValueError("hypothesis_max_tokens must be >= 1")
        self.host = host
        self.tokenizer = tokenizer
        self.masking_graft = masking_graft
        self.max_iterations = int(max_iterations)
        self.hypothesis_max_tokens = int(hypothesis_max_tokens)
        if require_attached:
            self._verify_attached()

    def _verify_attached(self) -> None:
        grafts = getattr(self.host, "grafts", None)
        if grafts is None:
            return  # Stub host without slot bookkeeping; trust the caller.
        logits_slots = grafts.get("logits")
        if logits_slots:
            for m in logits_slots:
                if m is self.masking_graft:
                    return
        raise RuntimeError(
            "HypothesisMaskingGraft must be attached specifically to host.grafts['logits'] "
            "(e.g. host.add_graft('logits', graft)) before driving IterativeHypothesisSearch; "
            "otherwise bans never reach logits."
        )

    @torch.no_grad()
    def run(
        self,
        prompt_ids: Sequence[int],
        evaluator: Callable[[list[int], str], HypothesisVerdict],
        *,
        clear_bans: bool = True,
        ban_first_token_on_reject: bool = True,
        autoregressive_extra_state: Optional[Mapping[str, Any]] = None,
    ) -> HypothesisSearchResult:
        """Drive the search until the evaluator accepts or ``max_iterations`` is reached."""

        if clear_bans:
            self.masking_graft.clear()

        device = _infer_host_device(self.host)
        prompt = list(int(t) for t in prompt_ids)
        history: list[HypothesisAttempt] = []
        last_attempt: HypothesisAttempt | None = None

        WorkspacePublisher.emit(
            "cog.hypothesis.start",
            {
                "max_iterations": self.max_iterations,
                "hypothesis_max_tokens": self.hypothesis_max_tokens,
                "prompt_len": len(prompt),
                "clear_bans": bool(clear_bans),
            },
        )

        for it in range(1, self.max_iterations + 1):
            generated = self._generate_tokens(
                prompt,
                device=device,
                extra_state=dict(autoregressive_extra_state or {}),
            )
            text = _decode_token_ids(self.tokenizer, generated)
            verdict = evaluator(list(generated), text)
            attempt = HypothesisAttempt(
                iteration=it, token_ids=list(generated), text=text, verdict=verdict
            )
            history.append(attempt)
            last_attempt = attempt

            WorkspacePublisher.emit(
                "cog.hypothesis.attempt",
                {
                    "iteration": it,
                    "tokens": list(generated),
                    "text": text[:120],
                    "valid": bool(verdict.valid),
                    "reason": str(verdict.reason or ""),
                    "ban_tokens": list(verdict.ban_tokens),
                },
            )

            if verdict.valid:
                logger.info(
                    "IterativeHypothesisSearch: accepted iter=%d tokens=%s text=%r",
                    it,
                    generated,
                    text,
                )
                WorkspacePublisher.emit(
                    "cog.hypothesis.complete",
                    {
                        "accepted": True,
                        "iterations": it,
                        "final_text": text[:120],
                        "n_attempts": len(history),
                    },
                )
                return HypothesisSearchResult(
                    accepted=True,
                    iterations=it,
                    final_token_ids=list(generated),
                    final_text=text,
                    history=history,
                )

            ban: list[int] = list(verdict.ban_tokens)
            if not ban and ban_first_token_on_reject and generated:
                ban = [generated[0]]
            if ban:
                self.masking_graft.ban(
                    ban,
                    reason=verdict.reason or f"rejected iter={it}",
                )
            else:
                logger.warning(
                    "IterativeHypothesisSearch: iter=%d rejected with no ban tokens; "
                    "next iteration will repeat the same hypothesis. text=%r",
                    it,
                    text,
                )

        accepted = False
        final_ids = list(last_attempt.token_ids) if last_attempt is not None else []
        final_text = last_attempt.text if last_attempt is not None else ""
        WorkspacePublisher.emit(
            "cog.hypothesis.complete",
            {
                "accepted": False,
                "iterations": self.max_iterations,
                "final_text": final_text[:120],
                "n_attempts": len(history),
            },
        )
        return HypothesisSearchResult(
            accepted=accepted,
            iterations=self.max_iterations,
            final_token_ids=final_ids,
            final_text=final_text,
            history=history,
        )

    def _generate_tokens(
        self,
        prompt_ids: Sequence[int],
        *,
        device: torch.device,
        extra_state: dict[str, Any],
    ) -> list[int]:
        current = list(prompt_ids)
        generated: list[int] = []
        for _ in range(self.hypothesis_max_tokens):
            ids = torch.tensor([current], dtype=torch.long, device=device)
            mask = torch.ones_like(ids, dtype=torch.bool)
            extra = dict(extra_state)
            extra.setdefault("tokenizer", self.tokenizer)
            out = self.host(ids, mask, extra_state=extra)
            logits = out[0] if isinstance(out, tuple) else out
            last_pos = ids.shape[1] - 1
            pred = int(logits[0, last_pos].argmax().item())
            generated.append(pred)
            current.append(pred)
        return generated


# ---------------------------------------------------------------------------
# 2. Epistemic Interruption — Mid-Generation Halt + Correction
# ---------------------------------------------------------------------------


@dataclass
class InterruptionVerdict:
    """Verdict from a mid-generation evaluator.

    ``halt=True`` triggers a correction pass: the monitor truncates the last
    ``truncate_tokens`` from its working buffer (so the LLM is forced off the
    bad path it was committing to), then for the next ``boost_steps`` tokens
    forwards ``correction_features`` as ``broca_features`` so the
    :class:`TrainableFeatureGraft` (or any other graft that reads
    ``broca_features``) injects a high-magnitude re-evaluation vector. If
    ``ban_tokens`` is supplied, the monitor's ``masking_graft`` (when
    attached) bans those tokens for the rest of the generation so the LLM
    cannot fall straight back into the same trap.
    """

    halt: bool
    truncate_tokens: int = 0
    correction_features: Optional[torch.Tensor] = None
    boost_steps: int = 5
    ban_tokens: tuple[int, ...] = ()
    reason: str = ""


@dataclass
class InterruptionEvent:
    step: int
    truncated: int
    boost_steps: int
    banned_tokens: tuple[int, ...]
    reason: str


@dataclass
class EpistemicInterruptionResult:
    """Outcome of an :class:`EpistemicInterruptionMonitor` run."""

    token_ids: list[int]
    text: str
    interventions: list[InterruptionEvent]
    final_step: int


class EpistemicInterruptionMonitor:
    """Streaming generator that runs an evaluator every ``check_every`` tokens.

    The monitor loop is the autoregressive equivalent of finishing someone
    else's sentence: it keeps a running buffer of generated tokens, and when
    the substrate's evaluator detects a logical collision it physically jars
    the LLM out of its hallucination rut by:

    1.  Truncating the most-recent ``truncate_tokens`` so the bad commitments
        are no longer in the prefix.
    2.  Setting ``broca_features`` to ``correction_features`` for the next
        ``boost_steps`` forwards (acting as a strong "re-evaluate" residual
        kick). When no :class:`TrainableFeatureGraft` is attached, this is
        passed as ``state["broca_features"]`` for any graft to consume.
    3.  Optionally banning the offending tokens via an attached
        :class:`HypothesisMaskingGraft` so the LLM cannot loop back into them.

    The monitor never modifies the host's other state. Confidence and inertia
    are passed in as construction-time arguments so the calling substrate
    keeps full ownership of the SNR-scaled grafts.
    """

    def __init__(
        self,
        host: Any,
        tokenizer: Any,
        *,
        check_every: int = 5,
        max_truncations: int = 4,
        masking_graft: Optional[HypothesisMaskingGraft] = None,
        substrate_confidence: float = 1.0,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ):
        if check_every < 1:
            raise ValueError("check_every must be >= 1")
        if max_truncations < 0:
            raise ValueError("max_truncations must be >= 0")
        self.host = host
        self.tokenizer = tokenizer
        self.check_every = int(check_every)
        self.max_truncations = int(max_truncations)
        self.masking_graft = masking_graft
        self.substrate_confidence = float(substrate_confidence)
        self.do_sample = bool(do_sample)
        self.temperature = float(temperature)
        self.top_p = float(top_p)

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: Sequence[int],
        *,
        max_new_tokens: int,
        evaluator: Callable[[list[int], str, int], InterruptionVerdict],
        eos_token_id: Optional[int] = None,
        on_token: Optional[Callable[[int, str], None]] = None,
        broca_features: Optional[torch.Tensor] = None,
        extra_state: Optional[Mapping[str, Any]] = None,
    ) -> EpistemicInterruptionResult:
        """Stream generation, periodically polling ``evaluator`` for halts.

        The evaluator receives ``(generated_token_ids, decoded_text, step)``.
        It must return a :class:`InterruptionVerdict`.
        """

        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")

        device = _infer_host_device(self.host)
        prompt = list(int(t) for t in prompt_ids)
        generated: list[int] = []
        interventions: list[InterruptionEvent] = []
        boost_remaining = 0
        active_correction: Optional[torch.Tensor] = None
        truncations = 0

        WorkspacePublisher.emit(
            "cog.epistemic.start",
            {
                "prompt_len": len(prompt),
                "max_new_tokens": int(max_new_tokens),
                "check_every": self.check_every,
                "max_truncations": self.max_truncations,
            },
        )

        step = 0
        while step < max_new_tokens:
            current = prompt + generated
            ids = torch.tensor([current], dtype=torch.long, device=device)
            mask = torch.ones_like(ids, dtype=torch.bool)

            inertia = math.log1p(float(len(current)))
            forwarded: dict[str, Any] = dict(extra_state or {})
            forwarded.setdefault("tokenizer", self.tokenizer)
            forwarded["substrate_confidence"] = float(self.substrate_confidence)
            forwarded["substrate_inertia"] = float(inertia)

            if boost_remaining > 0 and active_correction is not None:
                forwarded["broca_features"] = active_correction.to(device)
            elif broca_features is not None:
                forwarded["broca_features"] = broca_features.to(device)

            out = self.host(ids, mask, extra_state=forwarded)
            logits = out[0] if isinstance(out, tuple) else out
            last_pos = ids.shape[1] - 1
            pred = self._sample(logits[0, last_pos])

            if eos_token_id is not None and pred == int(eos_token_id):
                logger.debug("EpistemicInterruption: hit eos at step=%d", step)
                break

            generated.append(int(pred))
            step += 1
            if on_token is not None:
                on_token(int(pred), _decode_token_ids(self.tokenizer, [int(pred)]))

            if boost_remaining > 0:
                boost_remaining -= 1
                if boost_remaining == 0:
                    active_correction = None

            should_check = (step % self.check_every == 0) or step == max_new_tokens
            if not should_check:
                continue
            if truncations >= self.max_truncations:
                continue

            text = _decode_token_ids(self.tokenizer, generated)
            verdict = evaluator(list(generated), text, step)
            if not verdict.halt:
                continue

            truncate_n = max(0, min(int(verdict.truncate_tokens), len(generated)))
            if truncate_n > 0:
                generated = generated[: len(generated) - truncate_n]
                step = len(generated)

            if verdict.ban_tokens and self.masking_graft is not None:
                self.masking_graft.ban(
                    verdict.ban_tokens,
                    reason=verdict.reason or f"interruption@step={step}",
                )

            if verdict.correction_features is not None and verdict.boost_steps > 0:
                active_correction = verdict.correction_features.detach().clone()
                boost_remaining = int(verdict.boost_steps)
            else:
                active_correction = None
                boost_remaining = 0

            interventions.append(
                InterruptionEvent(
                    step=step,
                    truncated=truncate_n,
                    boost_steps=boost_remaining,
                    banned_tokens=tuple(int(t) for t in verdict.ban_tokens),
                    reason=str(verdict.reason),
                )
            )
            truncations += 1
            WorkspacePublisher.emit(
                "cog.epistemic.intervention",
                {
                    "step": step,
                    "truncated": truncate_n,
                    "boost_steps": boost_remaining,
                    "banned_tokens": list(int(t) for t in verdict.ban_tokens),
                    "reason": str(verdict.reason),
                    "intervention_index": len(interventions),
                },
            )

        text = _decode_token_ids(self.tokenizer, generated)
        WorkspacePublisher.emit(
            "cog.epistemic.complete",
            {
                "final_step": step,
                "n_interventions": len(interventions),
                "text": text[:120],
            },
        )
        return EpistemicInterruptionResult(
            token_ids=list(generated),
            text=text,
            interventions=interventions,
            final_step=step,
        )

    def _sample(self, logits_row: torch.Tensor) -> int:
        if not self.do_sample:
            return int(logits_row.argmax().item())
        scaled = logits_row.float() / max(self.temperature, 1e-5)
        probs = torch.softmax(scaled, dim=-1)
        if self.top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cdf = torch.cumsum(sorted_probs, dim=-1)
            over = (cdf > self.top_p).nonzero(as_tuple=False)
            keep = int(over[0, 0].item()) + 1 if over.numel() > 0 else int(probs.numel())
            keep = max(1, keep)
            kept = sorted_probs[:keep] / sorted_probs[:keep].sum().clamp_min(1e-12)
            pick = int(torch.multinomial(kept, 1).item())
            return int(sorted_idx[pick].item())
        return int(torch.multinomial(probs, 1).item())


# ---------------------------------------------------------------------------
# 3. Modality Shifting — Top-Down Attention Bias
# ---------------------------------------------------------------------------


class ModalityShiftGraft(BaseGraft):
    """Continuous bias toward a named cognitive mode (analytical / fluent / ...).

    Each mode is a unit-norm direction in ``d_model`` space. The active mode's
    direction is injected into the residual stream at every selected position
    (``position_mode``):

    *   ``"last"``   — last valid token only (cheapest; default).
    *   ``"all"``    — every valid (mask=1) token in the sequence; the mode
        permeates the prefix so even tokens already committed are seen
        through the mode.
    *   ``"first"``  — only the first token (mostly useful for "scene set"
        biases at sequence start).

    Modes can be registered from any of three sources: an explicit direction
    tensor, a captured activation mode (see ``dynamic_grafts.py``), or an
    average of token embeddings (see :meth:`mode_from_tokens`).

    State overrides:

    *   ``broca_modality`` — string mode name to use this forward only,
        bypassing :attr:`active_mode`. Lets the substrate switch modes per
        step without thread-local state.
    """

    _POSITION_MODES = ("last", "all", "first")

    def __init__(
        self,
        d_model: int,
        *,
        target_snr: float = 0.20,
        position_mode: str = "last",
        mixer_priority: float = 0.5,
    ):
        super().__init__()
        if position_mode not in self._POSITION_MODES:
            raise ValueError(
                f"position_mode must be one of {self._POSITION_MODES!r}, got {position_mode!r}"
            )
        self.d_model = int(d_model)
        self.target_snr = float(target_snr)
        self.position_mode = str(position_mode)
        self.mixer_priority = float(mixer_priority)
        self.modes: dict[str, torch.Tensor] = {}
        self.active_mode: Optional[str] = None
        self.last_mode_used: Optional[str] = None

    # -- registration -------------------------------------------------------

    def register_mode(self, name: str, direction: torch.Tensor) -> None:
        """Register a mode by name with a raw ``[d_model]``-shaped direction tensor."""

        if not name:
            raise ValueError("mode name must be a non-empty string")
        d = direction.detach().float().reshape(-1)
        if d.numel() != self.d_model:
            raise ValueError(
                f"direction must have {self.d_model} elements, got {d.numel()}"
            )
        unit = F.normalize(d.reshape(1, -1), dim=-1).reshape(-1)
        self.modes[str(name)] = unit
        logger.debug("ModalityShiftGraft.register_mode: name=%s d_model=%d", name, self.d_model)
        WorkspacePublisher.emit(
            "cog.modality_shift.register",
            {
                "name": str(name),
                "d_model": self.d_model,
                "n_modes": len(self.modes),
            },
        )

    def register_mode_from_capture(self, name: str, captured: Any) -> None:
        """Register from a :class:`CapturedActivationMode`-like object (any ``.value`` tensor)."""

        value = getattr(captured, "value", None)
        if value is None:
            raise AttributeError("captured object has no .value tensor")
        self.register_mode(name, value)

    @torch.no_grad()
    def mode_from_tokens(
        self,
        name: str,
        *,
        token_ids: Sequence[int],
        lm_head: nn.Linear,
    ) -> None:
        """Build a mode direction by averaging the LM-head rows of ``token_ids``.

        Useful for synthesizing an "analytical mode" from a list of digits +
        spelling tokens, or a "list-formatting" mode from bullet/newline
        tokens, without having to run a priming prompt.
        """

        ids = [int(t) for t in token_ids]
        if not ids:
            raise ValueError("token_ids must be non-empty")
        weight = lm_head.weight
        rows = []
        for tid in ids:
            if tid < 0 or tid >= weight.shape[0]:
                continue
            rows.append(weight[tid].detach().float().reshape(-1))
        if not rows:
            raise ValueError("no token_ids fell within lm_head's vocabulary")
        avg = torch.stack(rows, dim=0).mean(dim=0)
        self.register_mode(name, avg)

    def set_active_mode(self, name: Optional[str]) -> None:
        if name is None:
            self.active_mode = None
            WorkspacePublisher.emit("cog.modality_shift.set_active", {"name": None})
            return
        if name not in self.modes:
            raise KeyError(f"mode {name!r} is not registered")
        self.active_mode = str(name)
        WorkspacePublisher.emit(
            "cog.modality_shift.set_active",
            {"name": str(name), "n_modes": len(self.modes)},
        )

    def clear_modes(self) -> None:
        self.modes.clear()
        self.active_mode = None

    # -- forward ------------------------------------------------------------

    def forward(self, x: torch.Tensor, state: dict) -> torch.Tensor:
        if not self.enabled:
            return x
        mode_name = state.get("broca_modality") or self.active_mode
        if mode_name is None or mode_name not in self.modes:
            return x
        self.last_mode_used = str(mode_name)
        direction = self.modes[mode_name].to(device=x.device, dtype=x.dtype)
        bsz, seq_len, _ = x.shape
        confidence = state_confidence(state)
        inertia = state_inertia(state)

        mask = state.get("attention_mask")
        if mask is None:
            mask_t = torch.ones((bsz, seq_len), device=x.device, dtype=torch.bool)
        else:
            mask_t = mask.to(device=x.device, dtype=torch.bool)

        out = x.clone()
        if self.position_mode == "last":
            last_raw = state.get("last_indices")
            if last_raw is None:
                raise ValueError(
                    "ModalityShiftGraft(position_mode='last') requires state['last_indices']; "
                    "the host must populate it for TopDownControl grafts."
                )
            last = last_raw.to(x.device).long()
            rows = torch.arange(bsz, device=x.device)
            host_at_last = x[rows, last]
            magnitude = snr_magnitude(
                host_at_last,
                target_snr=self.target_snr,
                confidence=confidence,
                inertia=inertia,
            )
            out[rows, last] = out[rows, last] + direction.unsqueeze(0) * magnitude
            return out

        if self.position_mode == "first":
            magnitude = snr_magnitude(
                x[:, 0],
                target_snr=self.target_snr,
                confidence=confidence,
                inertia=inertia,
            )
            out[:, 0] = out[:, 0] + direction.unsqueeze(0) * magnitude
            return out

        # position_mode == "all"
        magnitude = snr_magnitude(
            x,
            target_snr=self.target_snr,
            confidence=confidence,
            inertia=inertia,
        )
        gate = mask_t.unsqueeze(-1).to(x.dtype)
        delta = direction.view(1, 1, -1) * magnitude * gate
        return out + delta


# ---------------------------------------------------------------------------
# 4. Causal Constraint Injection — KV-encoded do-calculus facts
# ---------------------------------------------------------------------------


@dataclass
class CausalConstraint:
    """Bookkeeping record for a constraint encoded into :class:`CausalConstraintGraft`."""

    treatment: str
    treatment_value: Any
    outcome: str
    distribution: dict[Any, float]
    concept_token: str
    outcome_token_map: dict[Any, str]
    metadata: dict[str, Any] = field(default_factory=dict)


class CausalConstraintGraft(KVMemoryGraft):
    """KV-memory graft pre-loaded with do-calculus facts as concept→outcome biases.

    Each constraint is a (key, value) pair where:

    *   ``key`` is the unit-norm LM-head row of a *concept token* (e.g. the
        token that surfaces the cause variable in language). When the host's
        residual stream sequence-mean aligns with this key, the graft fires.
    *   ``value`` is the unit-norm probability-weighted blend of the outcome
        tokens' LM-head rows under ``P(outcome=v | do(treatment=t))``. So an
        ATE of ``+0.55`` for a binary outcome whose values map to tokens
        ``"helps"`` and ``"hurts"`` produces a value direction that is a
        blend, not a pure ``"helps"`` — the LLM is pulled toward *partial*
        success, exactly as the SCM says it should be.

    The graft inherits all the geometry of :class:`KVMemoryGraft` (peakedness +
    manifold gating, top-k retrieval, SNR-scaled magnitude) so loading more
    constraints does not require any retuning of strength.

    Use :meth:`encode_treatment_effect` for the common case of "encode the
    distribution P(outcome | do(treatment=v)) as one constraint", or
    :meth:`add_constraint` for a fully custom (key, value) pair.
    """

    def __init__(
        self,
        d_model: int,
        *,
        max_items: int = 256,
        target_snr: float = 0.20,
        query_mode: str = "sequence_mean",
        mixer_priority: float = 0.4,
    ):
        super().__init__(
            d_model,
            max_items=max_items,
            query_mode=query_mode,
            target_snr=target_snr,
        )
        self.mixer_priority = float(mixer_priority)
        self.constraints: list[CausalConstraint] = []

    def clear(self) -> None:
        super().clear()
        self.constraints.clear()

    # -- low-level: arbitrary key/value -------------------------------------

    def add_constraint(
        self,
        *,
        key: torch.Tensor,
        value: torch.Tensor,
        constraint: Optional[CausalConstraint] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> CausalConstraint:
        """Add an arbitrary (key, value) pair with optional ``constraint`` bookkeeping."""

        key_v = key.detach().float().reshape(-1)
        val_v = value.detach().float().reshape(-1)
        if key_v.numel() != self.d_model or val_v.numel() != self.d_model:
            raise ValueError(
                f"key and value must have {self.d_model} elements; got "
                f"key={key_v.numel()} value={val_v.numel()}"
            )
        key_unit = F.normalize(key_v.reshape(1, -1), dim=-1).reshape(-1)
        val_unit = F.normalize(val_v.reshape(1, -1), dim=-1).reshape(-1)
        meta = dict(metadata or {})
        self.remember(key_unit.reshape(1, -1), val_unit.reshape(1, -1), metadata=meta)
        if constraint is None:
            constraint = CausalConstraint(
                treatment=str(meta.get("treatment", "")),
                treatment_value=meta.get("treatment_value"),
                outcome=str(meta.get("outcome", "")),
                distribution=dict(meta.get("distribution", {})),
                concept_token=str(meta.get("concept_token", "")),
                outcome_token_map=dict(meta.get("outcome_token_map", {})),
                metadata=meta,
            )
        self.constraints.append(constraint)
        WorkspacePublisher.emit(
            "cog.causal.constraint",
            {
                "treatment": constraint.treatment,
                "treatment_value": constraint.treatment_value,
                "outcome": constraint.outcome,
                "concept_token": constraint.concept_token,
                "distribution": {str(k): float(v) for k, v in constraint.distribution.items()},
                "n_constraints": len(self.constraints),
            },
        )
        return constraint

    # -- high-level: SCM intervention ---------------------------------------

    @torch.no_grad()
    def encode_treatment_effect(
        self,
        scm: Any,
        *,
        treatment: str,
        treatment_value: Any,
        outcome: str,
        concept_token: str,
        outcome_token_map: Mapping[Any, str],
        lm_head: nn.Linear,
        tokenizer: Any,
    ) -> CausalConstraint:
        """Encode ``P(outcome | do(treatment=treatment_value))`` as one constraint.

        ``concept_token`` is the surface form whose LM-head row we use as the
        retrieval key (e.g. ``"Treatment"``, ``"smoking"``, ``"prozac"``).

        ``outcome_token_map`` maps each value of the outcome variable in the
        SCM (as appearing in ``scm.domains[outcome]``) to a surface form
        whose LM-head row is the value-direction for that outcome value.
        Outcomes whose surface forms are missing from the tokenizer
        contribute zero weight (and the missing keys are recorded in the
        constraint's metadata).
        """

        if outcome not in scm.domains:
            raise KeyError(f"outcome {outcome!r} not in SCM domains")
        outcome_domain = list(scm.domains[outcome])

        distribution: dict[Any, float] = {}
        for v in outcome_domain:
            try:
                p = float(
                    scm.probability(
                        {outcome: v},
                        given={},
                        interventions={treatment: treatment_value},
                    )
                )
            except (ValueError, KeyError, TypeError):
                logger.exception(
                    "CausalConstraintGraft.encode_treatment_effect: scm.probability failed for "
                    "treatment_value=%r outcome=%r value=%r distribution=%s",
                    treatment_value,
                    outcome,
                    v,
                    getattr(scm, "domains", {}).get(outcome),
                )
                p = 0.0
            distribution[v] = p

        # Build value direction as probability-weighted sum of outcome token rows.
        weight = lm_head.weight
        accumulator = torch.zeros(weight.shape[1], device=weight.device, dtype=torch.float32)
        missing: list[Any] = []
        present: list[Any] = []
        for v, p in distribution.items():
            if p <= 0.0:
                continue
            tok_str = outcome_token_map.get(v)
            if tok_str is None:
                missing.append(v)
                continue
            tid = _resolve_token_id(tokenizer, str(tok_str))
            if tid is None or tid < 0 or tid >= weight.shape[0]:
                missing.append(v)
                continue
            row = F.normalize(
                weight[int(tid)].detach().float().reshape(1, -1), dim=-1
            ).reshape(-1)
            accumulator = accumulator + float(p) * row
            present.append(v)

        if accumulator.norm() <= 1e-9:
            raise ValueError(
                f"could not compute outcome direction for treatment={treatment!r} "
                f"value={treatment_value!r}: every mapped outcome token was missing "
                f"or zero-probability; missing={missing!r}"
            )

        # Concept key direction.
        concept_id = _resolve_token_id(tokenizer, str(concept_token))
        if concept_id is None:
            raise KeyError(
                f"concept_token {concept_token!r} did not resolve to a token id"
            )
        if concept_id < 0 or concept_id >= weight.shape[0]:
            raise KeyError(
                f"concept_token {concept_token!r} resolved to invalid id {concept_id!r} "
                f"(vocab={weight.shape[0]})"
            )
        key_dir = F.normalize(
            weight[int(concept_id)].detach().float().reshape(1, -1), dim=-1
        ).reshape(-1)

        meta: dict[str, Any] = {
            "treatment": treatment,
            "treatment_value": treatment_value,
            "outcome": outcome,
            "distribution": distribution,
            "concept_token": concept_token,
            "outcome_token_map": dict(outcome_token_map),
            "missing_outcome_tokens": missing,
            "present_outcome_values": present,
        }
        constraint = CausalConstraint(
            treatment=treatment,
            treatment_value=treatment_value,
            outcome=outcome,
            distribution=distribution,
            concept_token=concept_token,
            outcome_token_map=dict(outcome_token_map),
            metadata=meta,
        )
        self.add_constraint(
            key=key_dir, value=accumulator, constraint=constraint, metadata=meta
        )
        return constraint

    @torch.no_grad()
    def encode_intervention_grid(
        self,
        scm: Any,
        *,
        treatment: str,
        outcome: str,
        outcome_token_map: Mapping[Any, str],
        lm_head: nn.Linear,
        tokenizer: Any,
        treatment_concept_tokens: Optional[Mapping[Any, str]] = None,
    ) -> list[CausalConstraint]:
        """Encode one constraint per value of ``treatment`` in its SCM domain.

        ``treatment_concept_tokens`` maps each treatment value to a surface
        form; defaults to mapping every value to ``treatment`` itself (so the
        same concept token is reused for every intervention level — useful
        when the LLM tokenizes the cause concept once and the substrate just
        wants to bias the outcome distribution conditional on its own do()).
        """

        if treatment not in scm.domains:
            raise KeyError(f"treatment {treatment!r} not in SCM domains")
        treatment_domain = list(scm.domains[treatment])
        if treatment_concept_tokens is None:
            treatment_concept_tokens = {v: treatment for v in treatment_domain}
        out: list[CausalConstraint] = []
        for v in treatment_domain:
            concept_token = treatment_concept_tokens.get(v, treatment)
            out.append(
                self.encode_treatment_effect(
                    scm,
                    treatment=treatment,
                    treatment_value=v,
                    outcome=outcome,
                    concept_token=str(concept_token),
                    outcome_token_map=outcome_token_map,
                    lm_head=lm_head,
                    tokenizer=tokenizer,
                )
            )
        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _decode_token_ids(tokenizer: Any, ids: Sequence[int]) -> str:
    """Best-effort decoding that works against both lab and HF-style tokenizers."""

    if not ids:
        return ""
    dec_tokens = getattr(tokenizer, "decode_tokens", None)
    if callable(dec_tokens):
        try:
            return str(dec_tokens(list(ids)))
        except Exception:
            logger.debug("decode_tokens failed; falling back", exc_info=True)
    inner = getattr(tokenizer, "inner", None)
    if inner is not None and callable(getattr(inner, "decode", None)):
        try:
            return inner.decode(
                list(int(i) for i in ids),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        except Exception:
            logger.debug("inner.decode failed; falling back", exc_info=True)
    decode_id = getattr(tokenizer, "decode_id", None)
    if callable(decode_id):
        return " ".join(str(decode_id(int(i))) for i in ids)
    return " ".join(str(int(i)) for i in ids)


def _resolve_token_id(tokenizer: Any, surface: str) -> Optional[int]:
    """Resolve a surface form to a single token id across the tokenizer flavours we support."""

    if not surface:
        return None
    table = getattr(tokenizer, "token_to_id", None)
    if isinstance(table, dict) and surface in table:
        return int(table[surface])
    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        try:
            ids = list(encode(surface))
            if len(ids) > 1:
                logger.warning(
                    "_resolve_token_id: surface %r tokenized to multiple ids %s via tokenizer.encode; "
                    "using first id only",
                    surface,
                    ids,
                )
            if ids:
                return int(ids[0])
        except Exception:
            logger.debug("tokenizer.encode failed for %r", surface, exc_info=True)
    inner = getattr(tokenizer, "inner", None)
    if inner is not None and callable(getattr(inner, "encode", None)):
        try:
            ids = inner.encode(surface, add_special_tokens=False)
            if len(ids) > 1:
                logger.warning(
                    "_resolve_token_id: surface %r tokenized to multiple ids %s via inner.encode; "
                    "using first id only",
                    surface,
                    list(ids),
                )
            if ids:
                return int(ids[0])
        except Exception:
            logger.debug("inner.encode failed for %r", surface, exc_info=True)
    return None


__all__ = [
    "HypothesisMaskingGraft",
    "HypothesisVerdict",
    "HypothesisAttempt",
    "HypothesisSearchResult",
    "IterativeHypothesisSearch",
    "InterruptionVerdict",
    "InterruptionEvent",
    "EpistemicInterruptionResult",
    "EpistemicInterruptionMonitor",
    "ModalityShiftGraft",
    "CausalConstraint",
    "CausalConstraintGraft",
]
