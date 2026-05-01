
"""Benchmark entrypoint for the Broca/Llama architecture.

There are three benchmark paths:

1. ``--engine native`` (default): a first-party HuggingFace-datasets harness.
   It loads real datasets such as BoolQ, PIQA, ARC, WinoGrande, HellaSwag,
   CommonsenseQA, OpenBookQA, MMLU subset, and GSM8K, then scores
   ``meta-llama/Llama-3.2-1B-Instruct`` locally.

2. ``--engine lm-eval``: EleutherAI lm-evaluation-harness parity.  This runs a
   vanilla HF causal LM and, for Llama checkpoints, the same model wrapped in an
   empty ``LlamaBrocaHost`` shell.  The delta should be close to zero.

Native runs default to a **paired leaderboard table** (vanilla LM vs ``LlamaBrocaHost`` shell, same
checkpoint) for Llama models. Opt out with ``--no-broca-host-compare``.

3. Broca architecture probes: scripted baseline-vs-Broca questions (orthogonal to the leaderboard table).

Examples:

  HF_TOKEN=... python -m core.benchmarks --engine native --preset quick --limit 50
  HF_TOKEN=... python -m core.benchmarks --engine native --no-broca-host-compare --preset quick --limit 50
  HF_TOKEN=... python -m core.benchmarks --engine both --preset standard --limit 100
  python -m core.benchmarks.hf_datasets_eval --model meta-llama/Llama-3.2-1B-Instruct --tasks boolq,piqa --limit 20

Use ``--model gpt2`` only for a public smoke check; the intended model is
``meta-llama/Llama-3.2-1B-Instruct``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from core.benchmarks.architecture_eval import run_broca_architecture_eval
from core.benchmarks.hf_datasets_eval import (
    DEFAULT_LLAMA_MODEL,
    DEFAULT_NATIVE_PRESETS,
    run_hf_datasets_benchmark,
    resolve_task_names,
)
from core.benchmarks.lm_eval_pair import run_paired_lm_eval
from core.device_utils import normalize_device_arg, pick_torch_device

logger = logging.getLogger(__name__)

DEFAULT_BENCHMARK_MODEL = DEFAULT_LLAMA_MODEL


LM_EVAL_PRESETS: dict[str, dict[str, str | None]] = {
    "quick": {
        "tasks": "winogrande,piqa,arc_easy",
        "limit": "80",
        "batch_size": "auto",
    },
    "standard": {
        "tasks": "winogrande,piqa,arc_easy,arc_challenge,boolq",
        "limit": "250",
        "batch_size": "auto",
    },
    "nlp": {
        "tasks": "hellaswag,winogrande,piqa,arc_challenge",
        "limit": "120",
        "batch_size": "auto",
    },
}


def hf_datasets_smoke(verbose: bool = True) -> None:
    """Verify a few canonical HF datasets load correctly."""

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Dataset smoke requires `datasets`; install requirements-benchmark.txt") from exc

    smoke_runs: list[tuple[str, Any]] = [
        ("GLUE SST-2 (sentiment)", lambda: load_dataset("glue", "sst2", split="validation[:4]")),
        ("BoolQ (reading comprehension)", lambda: load_dataset("boolq", split="validation[:4]")),
        ("HellaSwag (commonsense)", lambda: load_dataset("hellaswag", split="validation[:2]")),
    ]
    if verbose:
        print("\n--- Hugging Face datasets smoke load ---", flush=True)
    for label, fn in smoke_runs:
        if verbose:
            print(f"  loading {label} …", flush=True)
        ds = fn()
        if verbose:
            print(f"  ok  {label}: {len(ds)} row(s)", flush=True)


def normalize_limit_override(limit_override: str | int | None) -> str | None:
    if limit_override is None:
        return None
    stripped = str(limit_override).strip()
    return stripped if stripped else None


def _benchmark_seed(args: argparse.Namespace) -> int:
    if args.seed is not None:
        return int(args.seed)
    raw = os.environ.get("BENCHMARK_SEED", "0")
    try:
        return int(str(raw).strip())
    except ValueError:
        print(f"Invalid BENCHMARK_SEED={raw!r} (expected integer).", file=sys.stderr)
        raise SystemExit(2) from None


def _resolved_limit_strings_and_native_int(args: argparse.Namespace) -> tuple[str | None, int | None]:
    if args.limit is not None:
        limit_s = str(int(args.limit))
    else:
        limit_s = normalize_limit_override(os.environ.get("BENCHMARK_LIMIT"))
    if limit_s is None:
        native_limit = 50
    else:
        try:
            native_limit = int(limit_s)
        except ValueError:
            print(f"Invalid --limit / BENCHMARK_LIMIT value {limit_s!r} (expected integer).", file=sys.stderr)
            raise SystemExit(2) from None
    if native_limit < 0:
        return limit_s, None
    return limit_s, native_limit


def _try_print_architecture_suite_metrics(result: dict[str, Any], *, artifact_path: Path) -> None:
    """Log/print summary when ``artifact_path`` result carries the expected ``metrics`` tree."""

    if not isinstance(result, dict):
        logger.error("Broca architecture eval returned non-dict result: %r", result)
        return
    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        logger.error("Broca architecture eval result missing dict 'metrics': keys=%s", list(result.keys()))
        return
    arms = (
        "baseline_bare_language_host",
        "enhanced_broca_architecture",
        "delta_enhanced_minus_baseline",
    )
    subkeys = ("speech_exact_accuracy", "answer_present_accuracy")
    try:
        for arm in arms:
            block = metrics[arm]
            for sk in subkeys:
                block[sk]
    except KeyError as exc:
        logger.error(
            "Broca architecture eval metrics missing expected nested keys (%s); metrics=%r full_result=%r",
            exc,
            metrics,
            result,
        )
        return

    base = metrics["baseline_bare_language_host"]["speech_exact_accuracy"]
    enhanced = metrics["enhanced_broca_architecture"]["speech_exact_accuracy"]
    delta = metrics["delta_enhanced_minus_baseline"]["speech_exact_accuracy"]
    base_ans = metrics["baseline_bare_language_host"]["answer_present_accuracy"]
    enh_ans = metrics["enhanced_broca_architecture"]["answer_present_accuracy"]
    d_ans = metrics["delta_enhanced_minus_baseline"]["answer_present_accuracy"]
    print("\n--- Broca architecture eval (baseline vs enhanced) ---", flush=True)
    print(
        "  (speech_exact_* = verbatim match vs Broca's scripted reference line; bare LM baseline is usually 0.)",
        flush=True,
    )
    print(f"  speech_exact_accuracy: baseline={base:.3f} enhanced={enhanced:.3f} delta={delta:+.3f}", flush=True)
    print(f"  answer_present_accuracy: baseline={base_ans:.3f} enhanced={enh_ans:.3f} delta={d_ans:+.3f}", flush=True)
    print(f"  wrote {artifact_path}", flush=True)


def resolve_cli_device(device: str | None) -> tuple[str, str]:
    """Return (CLI device string, coarse device type cpu|mps|cuda)."""

    if device is None or str(device).strip() == "":
        dev_cli = str(pick_torch_device(os.environ.get("ASI_DEVICE")))
    else:
        dev_cli = str(device).strip()
    coarse = dev_cli.split(":")[0].lower()
    return dev_cli, coarse


def run_lm_eval_harness(
    *,
    model_id: str,
    preset: str,
    device: str | None,
    limit_override: str | None,
    output_dir: Path,
    num_fewshot: int = 0,
) -> tuple[int, Path | None]:
    """Run Eleuther ``simple_evaluate`` paired as a host-parity check."""

    try:
        import lm_eval  # noqa: F401
    except ImportError:
        print("Install benchmark deps: pip install -r requirements-benchmark.txt", file=sys.stderr)
        return 1, None

    cfg = LM_EVAL_PRESETS.get(preset)
    if cfg is None:
        print(f"Unknown lm-eval preset {preset!r}. Choose: {', '.join(LM_EVAL_PRESETS)}", file=sys.stderr)
        return 2, None

    dev_s, coarse = resolve_cli_device(device)
    tasks = str(cfg["tasks"])
    limit = normalize_limit_override(limit_override) or cfg.get("limit")

    out = output_dir / f"lm_eval_{preset}_{_dt.datetime.now(tz=_dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    print("\n--- EleutherAI LM Evaluation Harness (host parity check) ---", flush=True)
    print(f"preset={preset} tasks={tasks} limit={limit} device={dev_s} fewshot={num_fewshot}", flush=True)

    try:
        run_paired_lm_eval(
            model_id=model_id,
            preset_tasks=tasks,
            device_s=dev_s,
            coarse=coarse,
            limit=str(limit) if limit else None,
            out_dir=out,
            num_fewshot=num_fewshot,
        )
    except Exception as exc:
        print(f"lm-eval pair failed: {exc}", file=sys.stderr)
        return 3, None

    summary_path = out / "benchmark_summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                f"preset={preset}",
                f"model={model_id}",
                f"device={dev_s}",
                f"tasks={tasks}",
                f"limit={limit}",
                f"fewshot={num_fewshot}",
                "kind=lm_eval_host_parity",
                "pair_artifact=lm_eval_pair.json",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {summary_path}", flush=True)
    return 0, out


def run_broca_architecture_benchmark(
    *,
    llama_model_id: str,
    device: str | None,
    hf_token: str | bool | None,
    output_run_dir: Path,
) -> dict[str, Any]:
    """Score bare language host vs the active Broca architecture."""

    dev = device if (device and str(device).strip()) else str(pick_torch_device(os.environ.get("ASI_DEVICE")))
    path = output_run_dir / "broca_architecture_eval.json"

    with tempfile.TemporaryDirectory(prefix="broca_bench_mem_") as tmp:
        db_path = Path(tmp) / "benchmark.sqlite"
        try:
            result = run_broca_architecture_eval(
                seed=0,
                db_path=db_path,
                llama_model_id=llama_model_id,
                device=dev,
                hf_token=hf_token,
                output_path=path,
            )
        except Exception as exc:  # pragma: no cover
            result = {
                "kind": "broca_architecture_eval",
                "model_id": llama_model_id,
                "device": dev,
                "error": f"failed_to_load_broca_backend: {exc!r}",
            }
            path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"Broca architecture eval: could not load Llama Broca stack ({exc})", file=sys.stderr)
            return result

    _try_print_architecture_suite_metrics(result, artifact_path=path)
    return result


def write_suite_manifest(
    run_dir: Path,
    *,
    model_id: str,
    engine: str,
    native_result: dict[str, Any] | None,
    lm_eval_status: str | None,
    architecture_eval: dict[str, Any] | None,
) -> None:
    manifest = {
        "kind": "llama_benchmark_suite",
        "model_checkpoint": model_id,
        "engine": engine,
        "native_hf_datasets": native_result,
        "lm_eval_status": lm_eval_status,
        "broca_architecture_eval": architecture_eval,
    }
    p = run_dir / "benchmark_suite_manifest.json"
    p.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {p}", flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Llama 3.2 Broca + HuggingFace datasets benchmark suite.")
    parser.add_argument("--engine", choices=["native", "lm-eval", "both"], default=os.environ.get("BENCHMARK_ENGINE", "native"))
    parser.add_argument("--preset", default=os.environ.get("BENCHMARK_PRESET", "quick"), help="native preset or lm-eval preset.")
    parser.add_argument("--tasks", default=os.environ.get("BENCHMARK_TASKS", ""), help="Native HF comma-separated task override.")
    parser.add_argument(
        "--model",
        default=os.environ.get("BENCHMARK_MODEL", DEFAULT_BENCHMARK_MODEL),
        help=f"HF model id (default {DEFAULT_BENCHMARK_MODEL}).",
    )
    parser.add_argument("--device", default=None, help="Torch device (cpu, mps, cuda:0). Default: auto.")
    parser.add_argument("--limit", type=int, default=None, help="Examples per task. Use -1 for full native split.")
    parser.add_argument("--split", default=None, help="Native HF split override.")
    parser.add_argument("--output-dir", type=Path, default=Path(os.environ.get("BENCHMARK_OUTPUT_DIR", "runs/benchmarks")))
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--skip-architecture-eval", "--no-broca-probes", dest="skip_architecture_eval", action="store_true")
    parser.add_argument("--streaming", action="store_true", help="Native HF streaming.")
    parser.add_argument("--shuffle", action="store_true", help="Native shuffle before limit.")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed (default: BENCHMARK_SEED env or 0).")
    parser.add_argument("--chat-template", action="store_true", help="Native HF benchmark: use tokenizer chat template.")
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--generation-max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--no-broca-host-compare",
        action="store_true",
        help="Disable paired vanilla vs LlamaBrocaHost leaderboard table (enabled by default for Llama checkpoints).",
    )
    parser.add_argument("--num-fewshot", type=int, default=0, help="lm-eval fewshot count.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    bench_device = args.device if args.device is not None else normalize_device_arg(os.environ.get("BENCHMARK_DEVICE"))
    bench_seed = _benchmark_seed(args)
    limit_s, native_limit_or_none = _resolved_limit_strings_and_native_int(args)

    if not args.skip_smoke:
        hf_datasets_smoke(verbose=True)

    hf_tok_raw = os.environ.get("HF_TOKEN")
    hf_token: str | bool | None = hf_tok_raw if hf_tok_raw and hf_tok_raw.strip() else None

    run_root = args.output_dir
    run_root.mkdir(parents=True, exist_ok=True)

    native_result: dict[str, Any] | None = None
    lm_status: str | None = None
    architecture_eval: dict[str, Any] | None = None
    manifest_dir = run_root

    if args.engine in {"native", "both"}:
        tasks = resolve_task_names(args.tasks, preset=args.preset if args.preset in DEFAULT_NATIVE_PRESETS else "quick")
        print("\n--- Native HuggingFace-datasets benchmark ---", flush=True)
        print(f"model={args.model} tasks={','.join(tasks)} limit={native_limit_or_none} device={bench_device or 'auto'}", flush=True)
        native_result = run_hf_datasets_benchmark(
            model_id=args.model,
            tasks=tasks,
            output_dir=run_root,
            limit=native_limit_or_none,
            split=args.split,
            device=bench_device,
            hf_token=hf_token,
            streaming=args.streaming,
            seed=bench_seed,
            shuffle=args.shuffle,
            chat_template=args.chat_template,
            max_seq_len=args.max_seq_len,
            generation_max_new_tokens=args.generation_max_new_tokens,
            compare_llama_broca_host_shell=False if args.no_broca_host_compare else None,
        )

    if args.engine in {"lm-eval", "both"}:
        code, lm_dir = run_lm_eval_harness(
            model_id=args.model,
            preset=args.preset if args.preset in LM_EVAL_PRESETS else "quick",
            device=bench_device,
            limit_override=limit_s,
            output_dir=run_root,
            num_fewshot=args.num_fewshot,
        )
        lm_status = "completed" if code == 0 else f"failed:{code}"
        if lm_dir is not None:
            manifest_dir = lm_dir
        if code != 0 and args.engine == "lm-eval":
            sys.exit(code)

    # Native result contains its own timestamped artifact directory. Use that for
    # the manifest when available.
    if native_result is not None:
        # run_hf_datasets_benchmark returns relative artifact names, so find newest native dir.
        native_dirs = sorted(run_root.glob("hf_native_*"), key=lambda p: p.stat().st_mtime)
        if native_dirs:
            manifest_dir = native_dirs[-1]

    model_l = args.model.strip().lower()
    if args.skip_architecture_eval:
        pass
    elif model_l.startswith("gpt2"):
        print("\nSkipping Broca architecture eval (--model is gpt2). Use the Llama 3.2 Instruct id for this probe.", flush=True)
    else:
        architecture_eval = run_broca_architecture_benchmark(
            llama_model_id=args.model,
            device=bench_device,
            hf_token=hf_token,
            output_run_dir=manifest_dir,
        )

    write_suite_manifest(
        manifest_dir,
        model_id=args.model,
        engine=args.engine,
        native_result=native_result,
        lm_eval_status=lm_status,
        architecture_eval=architecture_eval,
    )


if __name__ == "__main__":
    main()
