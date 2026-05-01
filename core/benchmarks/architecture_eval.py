"""Scored baseline-vs-Broca architecture benchmark.

Questions the bare language host and the full Broca stack the same way interactive
chat does: canonical SQLite, :data:`core.substrate_runtime.CHAT_NAMESPACE`, one
:class:`BrocaMind` session for the probe series.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from core.broca import BrocaMind, generate_without_broca
from core.substrate_runtime import CHAT_NAMESPACE

logger = logging.getLogger(__name__)


_WORD_RE = re.compile(r"[a-z0-9_]+")

# Actions (and related tokens) accepted when scoring ``task_type == "active_inference"`` outputs.
ACTIVE_INFERENCE_ACTION_VOCAB: frozenset[str] = frozenset(
    {
        "listen",
        "open_left",
        "open_right",
        "observe_association",
        "run_intervention_readout",
        "observational",
        "intervention",
        "check",
        "evidence",
        "readout",
    }
)


@dataclass(frozen=True)
class ArchitectureEvalCase:
    id: str
    task_type: str
    prompt: str
    expected_answer: str
    expected_speech: str
    setup_prompts: tuple[str, ...] = ()


DEFAULT_ARCHITECTURE_EVAL_CASES: tuple[ArchitectureEvalCase, ...] = (
    ArchitectureEvalCase(
        id="active_action",
        task_type="active_inference",
        prompt="what action should i take ?",
        expected_answer="listen",
        expected_speech="i should listen .",
    ),
    ArchitectureEvalCase(
        id="causal_treatment",
        task_type="causal_intervention",
        prompt="does treatment help ?",
        expected_answer="helps",
        expected_speech="intervention says treatment helps .",
    ),
)


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(str(text).lower())


def _normalized_text(text: str) -> str:
    return " ".join(_tokens(text))


def _contains_answer(output: str, expected_answer: str) -> bool:
    answer = _normalized_text(expected_answer)
    if not answer:
        return False
    return answer in _normalized_text(output)


def _score_output(output: str, *, expected_answer: str, expected_speech: str, task_type: str = "") -> dict[str, bool]:
    speech_exact = _normalized_text(output) == _normalized_text(expected_speech)
    answer_present = _contains_answer(output, expected_answer)
    if task_type == "active_inference":
        nt = _normalized_text(output)
        toks = set(nt.split())
        normalized_expected = _normalized_text(expected_answer)
        if normalized_expected and normalized_expected not in ACTIVE_INFERENCE_ACTION_VOCAB:
            logger.warning(
                "active_inference expected_answer %r is not in ACTIVE_INFERENCE_ACTION_VOCAB; treating answer_present as False",
                expected_answer,
            )
            answer_present = False
        else:
            answer_present = bool(normalized_expected) and normalized_expected in toks
        speech_exact = nt.startswith("i should") and answer_present
    return {
        "speech_exact": speech_exact,
        "answer_present": answer_present,
    }


def _encode_len(tokenizer: Any, text: str) -> int:
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        ids = tokenizer.encode(text)
    return max(1, len(ids))


def _mean_bool(rows: Sequence[dict[str, Any]], arm: str, metric: str) -> float:
    if not rows:
        return 0.0
    return sum(1.0 for row in rows if row[arm]["scores"][metric]) / len(rows)


def _baseline_prompt(case: ArchitectureEvalCase) -> str:
    return f"{case.prompt}\nanswer :"


def run_broca_architecture_eval(
    *,
    seed: int = 0,
    db_path: str | Path,
    llama_model_id: str | None = None,
    device: str | None = None,
    hf_token: str | bool | None = None,
    output_path: str | Path | None = None,
    cases: Sequence[ArchitectureEvalCase] = DEFAULT_ARCHITECTURE_EVAL_CASES,
) -> dict[str, Any]:
    """Run a direct scored comparison between bare host and Broca architecture.

    Uses a **single** :class:`BrocaMind` load and the same SQLite + namespace as
    interactive chat (:data:`CHAT_NAMESPACE`), so benchmarks exercise and persist
    through the identical substrate stack—not an alternate memory partition.
    """

    mid = llama_model_id or os.environ.get("MODEL_ID", "meta-llama/Llama-3.2-1B-Instruct")

    try:
        from core.event_bus import get_default_bus

        bus = get_default_bus()
    except Exception:
        logger.debug(
            "Could not initialize bench event bus (get_default_bus); proceeding without bench.arch_case.* publishes",
            exc_info=True,
        )
        bus = None

    rows: list[dict[str, Any]] = []
    graft_reports_by_case: dict[str, str] = {}
    total_cases = len(cases)
    mind = BrocaMind(
        seed=seed,
        db_path=db_path,
        namespace=CHAT_NAMESPACE,
        llama_model_id=mid,
        device=device,
        hf_token=hf_token,
    )
    for ci, case in enumerate(cases, start=1):
        if bus is not None:
            bus.publish(
                "bench.arch_case.start",
                {"case_id": case.id, "task_type": case.task_type, "i": ci, "total": total_cases},
            )
        graft_reports_by_case[case.id] = mind.host.graft_report()
        for setup in case.setup_prompts:
            mind.comprehend(setup)
        max_new_tokens = _encode_len(mind.tokenizer, case.expected_speech)
        prompt = _baseline_prompt(case)
        with mind.host.grafts_enabled(False):
            baseline_output = generate_without_broca(
                mind.host,
                mind.tokenizer,
                prefix=prompt,
                max_new_tokens=max_new_tokens,
            )

        frame, enhanced_output = mind.answer(case.prompt, max_new_tokens=max_new_tokens)
        base_scores = _score_output(
            baseline_output,
            expected_answer=case.expected_answer,
            expected_speech=case.expected_speech,
            task_type=case.task_type,
        )
        enh_scores = _score_output(
            enhanced_output,
            expected_answer=case.expected_answer,
            expected_speech=case.expected_speech,
            task_type=case.task_type,
        )
        if bus is not None:
            bus.publish(
                "bench.arch_case.complete",
                {
                    "case_id": case.id,
                    "i": ci,
                    "total": total_cases,
                    "baseline_speech_exact": bool(base_scores.get("speech_exact")),
                    "enhanced_speech_exact": bool(enh_scores.get("speech_exact")),
                    "baseline_answer_present": bool(base_scores.get("answer_present")),
                    "enhanced_answer_present": bool(enh_scores.get("answer_present")),
                },
            )
        rows.append(
            {
                "id": case.id,
                "task_type": case.task_type,
                "prompt": case.prompt,
                "setup_prompts": list(case.setup_prompts),
                "expected_answer": case.expected_answer,
                "expected_speech": case.expected_speech,
                "baseline_bare_language_host": {
                    "prompt": prompt,
                    "output": baseline_output,
                    "scores": base_scores,
                },
                "enhanced_broca_architecture": {
                    "latent_frame": asdict(frame),
                    "output": enhanced_output,
                    "scores": enh_scores,
                },
            }
        )

    baseline_exact = _mean_bool(rows, "baseline_bare_language_host", "speech_exact")
    enhanced_exact = _mean_bool(rows, "enhanced_broca_architecture", "speech_exact")
    baseline_answer = _mean_bool(rows, "baseline_bare_language_host", "answer_present")
    enhanced_answer = _mean_bool(rows, "enhanced_broca_architecture", "answer_present")

    result: dict[str, Any] = {
        "kind": "broca_architecture_eval",
        "description": (
            "Direct scored comparison: bare frozen language host vs BrocaMind with semantic memory, "
            "active inference, causal substrate, workspace frames, and residual-stream graft verbalization. "
            "Runs on the canonical substrate SQLite with the same namespace as chat (CHAT_NAMESPACE); "
            "one BrocaMind session carries workspace and persistence across cases for this eval."
        ),
        "model_id": mid,
        "device": device,
        "seed": seed,
        "primary_metric": "speech_exact_accuracy",
        "graft_reports_by_case": graft_reports_by_case,
        "cases": rows,
        "metrics": {
            "baseline_bare_language_host": {
                "speech_exact_accuracy": baseline_exact,
                "answer_present_accuracy": baseline_answer,
            },
            "enhanced_broca_architecture": {
                "speech_exact_accuracy": enhanced_exact,
                "answer_present_accuracy": enhanced_answer,
            },
            "delta_enhanced_minus_baseline": {
                "speech_exact_accuracy": enhanced_exact - baseline_exact,
                "answer_present_accuracy": enhanced_answer - baseline_answer,
            },
        },
    }

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    return result
