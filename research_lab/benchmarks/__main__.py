"""Unified Mosaic benchmark suite (fixed wiring, no tuning flags).

Runs the native HF-datasets leaderboard (vanilla LM, Broca host shell, full substrate stack),
Eleuther LM-eval parity, dataset smoke sanity checks, and Broca architecture
baseline-vs-enhanced probes. Model id follows ``MODEL_ID`` / ``BENCHMARK_MODEL``.
Substrate probes persist to ``runs/broca_substrate.sqlite`` (pytest uses
``MOSAIC_TEST_DB``).

  HF_TOKEN=... python -m research_lab.benchmarks
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import sqlite3
import sys
import warnings
from pathlib import Path
from typing import Any, Sequence

from core.substrate.runtime import (
    BENCHMARK_ENGINE,
    BENCHMARK_FIXED_SEED,
    BENCHMARK_GEN_MAX_NEW_TOKENS,
    BENCHMARK_LIMIT,
    BENCHMARK_LM_EVAL_PRESET,
    BENCHMARK_NATIVE_PRESET,
    benchmark_output_root,
    default_model_id,
    default_substrate_sqlite_path,
    ensure_parent_dir,
)

from research_lab.benchmarks.architecture_eval import run_broca_architecture_eval
from research_lab.benchmarks.hf_datasets_eval import (
    DEFAULT_LLAMA_MODEL,
    DEFAULT_NATIVE_PRESETS,
    run_hf_datasets_benchmark,
    resolve_task_names,
)
from research_lab.benchmarks.lm_eval_pair import run_paired_lm_eval
from core.system.device import normalize_device_arg, pick_torch_device
from core.workspace import WorkspaceBuilder

logger = logging.getLogger(__name__)

DEFAULT_BENCHMARK_MODEL = DEFAULT_LLAMA_MODEL


def _touch_canonical_substrate_sqlite_early(*, model_id: str) -> None:
    """Create ``runs/broca_substrate.sqlite`` when the suite starts (non-tests, Llama backends).

    The native HF leg may open :class:`~core.cognition.substrate.SubstrateController` for the ``broca_mind`` arm when
    pairing on Llama checkpoints; lm-eval stays host-only; architecture eval runs *last*.
    Touching early avoids ``runs/`` looking empty during long leaderboard phases."""

    if os.environ.get("MOSAIC_UNDER_TEST", "").strip().lower() in {"1", "true", "yes"}:
        return
    if model_id.strip().lower().startswith("gpt2"):
        return
    p = default_substrate_sqlite_path()
    ensure_parent_dir(p)
    with sqlite3.connect(str(p)) as con:
        pass


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
        raise ImportError(
            "Dataset smoke requires `datasets`; run `uv sync --extra benchmark` or `pip install -e \".[benchmark]\"`."
        ) from exc

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


def _device_env_for_pick() -> str | None:
    raw = os.environ.get("M_DEVICE")
    if raw is not None and str(raw).strip() != "":
        return str(raw).strip()
    legacy = os.environ.get("ASI_DEVICE")
    if legacy is not None and str(legacy).strip() != "":
        warnings.warn(
            "ASI_DEVICE is deprecated; set M_DEVICE instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return str(legacy).strip()
    return None


def resolve_cli_device(device: str | None) -> tuple[str, str]:
    """Return (CLI device string, coarse device type cpu|mps|cuda)."""

    if device is None or str(device).strip() == "":
        dev_cli = str(pick_torch_device(_device_env_for_pick()))
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
        print('Install benchmark deps: uv sync --extra benchmark  (or pip install -e ".[benchmark]")', file=sys.stderr)
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

    bus = WorkspaceBuilder().process_default()
    bus.publish(
        "bench.phase.start",
        {
            "phase": "lm_eval",
            "preset": preset,
            "tasks": tasks,
            "limit": limit,
            "device": dev_s,
            "fewshot": num_fewshot,
        },
    )
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
        bus.publish("bench.phase.complete", {"phase": "lm_eval", "error": str(exc)})
        print(f"lm-eval pair failed: {exc}", file=sys.stderr)
        return 3, None
    bus.publish("bench.phase.complete", {"phase": "lm_eval", "out": str(out)})

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

    dev = device if (device and str(device).strip()) else str(pick_torch_device(_device_env_for_pick()))
    path = output_run_dir / "broca_architecture_eval.json"
    bus = WorkspaceBuilder().process_default()
    bus.publish(
        "bench.phase.start",
        {"phase": "architecture_eval", "model_id": llama_model_id, "device": dev},
    )

    db_path = default_substrate_sqlite_path()
    ensure_parent_dir(db_path)
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
        bus.publish("bench.phase.complete", {"phase": "architecture_eval", "error": str(exc)})
        return result

    _try_print_architecture_suite_metrics(result, artifact_path=path)
    metrics = result.get("metrics") if isinstance(result, dict) else None
    payload: dict[str, Any] = {"phase": "architecture_eval", "out": str(path)}
    if isinstance(metrics, dict):
        try:
            payload["baseline_speech_acc"] = float(metrics["baseline_bare_language_host"]["speech_exact_accuracy"])
            payload["enhanced_speech_acc"] = float(metrics["enhanced_broca_architecture"]["speech_exact_accuracy"])
            payload["delta_speech_acc"] = float(metrics["delta_enhanced_minus_baseline"]["speech_exact_accuracy"])
        except (KeyError, TypeError, ValueError):
            logger.debug(
                "Failed to extract architecture metrics for phase payload",
                exc_info=True,
                extra={"result": result, "metrics": metrics},
            )
    bus.publish("bench.phase.complete", payload)
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


def print_benchmark_cli_help() -> None:
    run_root = benchmark_output_root()
    print("Mosaic benchmarks — unified fixed configuration (no tuning flags).\n")
    print("Phases:")
    print("  • HF datasets smoke")
    print(f"  • Native leaderboard (preset {BENCHMARK_NATIVE_PRESET}, limit {BENCHMARK_LIMIT})")
    print(f"  • LM-eval host parity (preset {BENCHMARK_LM_EVAL_PRESET})")
    print("  • Broca architecture probes (baseline bare host vs enhanced Broca stack)\n")
    print(f"Model: MODEL_ID / BENCHMARK_MODEL (resolved at runtime → {default_model_id()})")
    print(f"Artifacts: OUTPUT_DIR={run_root}")
    print("Substrate DB: MOSAIC_TEST_DB when MOSAIC_UNDER_TEST=1 else runs/broca_substrate.sqlite\n")


def main(argv: Sequence[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    helper = argparse.ArgumentParser(add_help=False)
    helper.add_argument("-h", "--help", action="store_true")
    hargs, extra = helper.parse_known_args(argv)
    if hargs.help:
        print_benchmark_cli_help()
        return
    if extra:
        print("This benchmark entry point accepts no CLI arguments besides -h/--help.", file=sys.stderr)
        print(f"Remove: {' '.join(extra)}", file=sys.stderr)
        raise SystemExit(2)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    model_id = default_model_id()
    bench_device = normalize_device_arg(os.environ.get("BENCHMARK_DEVICE"))
    bench_seed = BENCHMARK_FIXED_SEED
    limit_s = str(BENCHMARK_LIMIT)
    native_limit = BENCHMARK_LIMIT

    _touch_canonical_substrate_sqlite_early(model_id=model_id)

    bus = WorkspaceBuilder().process_default()
    bus.publish(
        "bench.suite.start",
        {
            "engine": BENCHMARK_ENGINE,
            "preset": f"{BENCHMARK_NATIVE_PRESET}+{BENCHMARK_LM_EVAL_PRESET}",
            "model": model_id,
            "limit": native_limit,
            "device": bench_device,
            "seed": bench_seed,
        },
    )

    bus.publish("bench.phase.start", {"phase": "smoke"})
    smoke_payload: dict[str, Any] = {"phase": "smoke"}
    try:
        hf_datasets_smoke(verbose=True)
    except Exception as exc:
        smoke_payload["error"] = str(exc)
        logger.exception("HF datasets smoke phase failed")
    finally:
        bus.publish("bench.phase.complete", smoke_payload)

    hf_tok_raw = os.environ.get("HF_TOKEN")
    hf_token: str | bool | None = hf_tok_raw if hf_tok_raw and hf_tok_raw.strip() else None

    run_root = benchmark_output_root()
    run_root.mkdir(parents=True, exist_ok=True)

    native_result: dict[str, Any] | None = None
    lm_status: str | None = None
    architecture_eval: dict[str, Any] | None = None
    manifest_dir = run_root

    if BENCHMARK_ENGINE in {"native", "both"}:
        if BENCHMARK_NATIVE_PRESET in DEFAULT_NATIVE_PRESETS:
            preset = BENCHMARK_NATIVE_PRESET
        else:
            logger.warning(
                "Unknown BENCHMARK_NATIVE_PRESET=%r; falling back to %r. Allowed: %s.",
                BENCHMARK_NATIVE_PRESET,
                "quick",
                sorted(DEFAULT_NATIVE_PRESETS),
            )
            preset = "quick"
        tasks = resolve_task_names("", preset=preset)
        print("\n--- Native HuggingFace-datasets benchmark ---", flush=True)
        print(
            f"model={model_id} tasks={','.join(tasks)} limit={native_limit} device={bench_device or 'auto'}",
            flush=True,
        )
        native_result = run_hf_datasets_benchmark(
            model_id=model_id,
            tasks=tasks,
            output_dir=run_root,
            limit=native_limit,
            split=None,
            device=bench_device,
            hf_token=hf_token,
            streaming=False,
            seed=bench_seed,
            shuffle=False,
            chat_template=False,
            max_seq_len=None,
            generation_max_new_tokens=BENCHMARK_GEN_MAX_NEW_TOKENS,
            compare_llama_broca_host_shell=None,
        )

    if BENCHMARK_ENGINE in {"lm-eval", "both"}:
        if BENCHMARK_LM_EVAL_PRESET in LM_EVAL_PRESETS:
            lm_preset = BENCHMARK_LM_EVAL_PRESET
        else:
            logger.warning(
                "Unknown BENCHMARK_LM_EVAL_PRESET=%r; falling back to %r. Allowed: %s.",
                BENCHMARK_LM_EVAL_PRESET,
                "quick",
                sorted(LM_EVAL_PRESETS),
            )
            lm_preset = "quick"
        code, lm_dir = run_lm_eval_harness(
            model_id=model_id,
            preset=lm_preset,
            device=bench_device,
            limit_override=limit_s,
            output_dir=run_root,
            num_fewshot=0,
        )
        lm_status = "completed" if code == 0 else f"failed:{code}"
        if lm_dir is not None:
            manifest_dir = lm_dir
        if code != 0 and BENCHMARK_ENGINE == "lm-eval":
            sys.exit(code)

    if native_result is not None:
        native_dirs = sorted(run_root.glob("hf_native_*"), key=lambda p: p.stat().st_mtime)
        if native_dirs:
            manifest_dir = native_dirs[-1]

    model_l = model_id.strip().lower()
    if model_l.startswith("gpt2"):
        print(
            "\nSkipping Broca architecture eval (gpt2 smoke backend). "
            "Use the Llama 3.2 Instruct id for probes.",
            flush=True,
        )
    else:
        architecture_eval = run_broca_architecture_benchmark(
            llama_model_id=model_id,
            device=bench_device,
            hf_token=hf_token,
            output_run_dir=manifest_dir,
        )

    write_suite_manifest(
        manifest_dir,
        model_id=model_id,
        engine=BENCHMARK_ENGINE,
        native_result=native_result,
        lm_eval_status=lm_status,
        architecture_eval=architecture_eval,
    )
    bus.publish(
        "bench.suite.complete",
        {
            "engine": BENCHMARK_ENGINE,
            "manifest_dir": str(manifest_dir),
            "lm_eval_status": lm_status,
        },
    )


if __name__ == "__main__":
    main()
