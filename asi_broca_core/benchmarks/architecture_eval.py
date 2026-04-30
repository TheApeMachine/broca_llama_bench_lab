"""Scored baseline-vs-Broca architecture benchmark.

This is intentionally separate from leaderboard-style lm-eval. The benchmark
asks the bare language host and the full Broca architecture the same substrate
questions, then scores whether the produced answer/speech matches the known
faculty result.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from asi_broca_core.broca import BrocaMind, generate_without_broca


_WORD_RE = re.compile(r"[a-z0-9_]+")


@dataclass(frozen=True)
class ArchitectureEvalCase:
    id: str
    task_type: str
    prompt: str
    expected_answer: str
    expected_speech: str


DEFAULT_ARCHITECTURE_EVAL_CASES: tuple[ArchitectureEvalCase, ...] = (
    ArchitectureEvalCase(
        id="memory_ada",
        task_type="semantic_memory",
        prompt="where is ada ?",
        expected_answer="rome",
        expected_speech="ada is in rome .",
    ),
    ArchitectureEvalCase(
        id="memory_hopper",
        task_type="semantic_memory",
        prompt="where is hopper ?",
        expected_answer="lisbon",
        expected_speech="hopper is in lisbon .",
    ),
    ArchitectureEvalCase(
        id="active_action",
        task_type="active_inference",
        prompt="what action should i take ?",
        expected_answer="listen",
        expected_speech="i should listen first .",
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


def _score_output(output: str, *, expected_answer: str, expected_speech: str) -> dict[str, bool]:
    return {
        "speech_exact": _normalized_text(output) == _normalized_text(expected_speech),
        "answer_present": _contains_answer(output, expected_answer),
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
    backend: str = "tiny",
    llama_model_id: str = "meta-llama/Llama-3.2-1B-Instruct",
    device: str | None = None,
    hf_token: str | bool | None = None,
    output_path: str | Path | None = None,
    cases: Sequence[ArchitectureEvalCase] = DEFAULT_ARCHITECTURE_EVAL_CASES,
) -> dict[str, Any]:
    """Run a direct scored comparison between bare host and Broca architecture."""

    mind = BrocaMind(
        seed=seed,
        db_path=db_path,
        namespace=f"architecture_eval_{seed}",
        backend=backend,
        llama_model_id=llama_model_id,
        device=device,
        hf_token=hf_token,
    )

    rows: list[dict[str, Any]] = []
    for case in cases:
        max_new_tokens = _encode_len(mind.tokenizer, case.expected_speech)
        prompt = _baseline_prompt(case)
        with mind.host.grafts_enabled(False):
            baseline_output = generate_without_broca(
                mind.host,
                mind.tokenizer,
                prefix=prompt,
                max_new_tokens=max_new_tokens,
            )

        frame, enhanced_output = mind.answer(case.prompt)
        rows.append(
            {
                "id": case.id,
                "task_type": case.task_type,
                "prompt": case.prompt,
                "expected_answer": case.expected_answer,
                "expected_speech": case.expected_speech,
                "baseline_bare_language_host": {
                    "prompt": prompt,
                    "output": baseline_output,
                    "scores": _score_output(
                        baseline_output,
                        expected_answer=case.expected_answer,
                        expected_speech=case.expected_speech,
                    ),
                },
                "enhanced_broca_architecture": {
                    "latent_frame": asdict(frame),
                    "output": enhanced_output,
                    "scores": _score_output(
                        enhanced_output,
                        expected_answer=case.expected_answer,
                        expected_speech=case.expected_speech,
                    ),
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
            "active inference, causal substrate, workspace frames, and residual-stream graft verbalization."
        ),
        "backend": backend,
        "model_id": llama_model_id if backend == "llama" else "tiny",
        "device": device,
        "seed": seed,
        "primary_metric": "speech_exact_accuracy",
        "graft_report": mind.host.graft_report(),
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


