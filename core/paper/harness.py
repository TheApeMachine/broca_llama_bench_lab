"""Populate ``paper/include/experiment`` with benchmark-backed LaTeX snippets.

Run via ``python -m core.paper`` or ``make paper-bench``. This module is
optional at runtime: failures are logged and stub TeX is written so
``main.tex`` still encodes the intended structure.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)


def repo_root() -> Path:
    """Repository root (parent of ``paper/``)."""

    return Path(__file__).resolve().parents[2]


def paper_dirs(root: Path | None = None) -> tuple[Path, Path]:
    r = repo_root() if root is None else Path(root)
    paper = r / "paper"
    exp = paper / "include" / "experiment"
    return paper, exp


def _latex_escape(s: str) -> str:
    return (
        str(s)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return int(default)
    try:
        return int(str(raw).strip())
    except ValueError:
        return int(default)


def write_vanilla_table_tex(summary: Mapping[str, Any], dest: Path) -> None:
    per_task = summary.get("per_task") or {}
    lines = [
        r"\begin{tabular}{lrr}",
        r"\hline",
        r"Task & $n$ & Accuracy \\",
        r"\hline",
    ]
    for task in sorted(per_task.keys()):
        m = per_task[task]
        if not isinstance(m, Mapping):
            continue
        safe_task = _latex_escape(str(task))
        lines.append(
            f"{safe_task} & {m.get('n', '')} & {float(m.get('accuracy', 0.0)):.4f} \\\\",
        )
    lines.extend([r"\hline", r"\end{tabular}", ""])
    dest.write_text("\n".join(lines), encoding="utf-8")


def write_comparison_table_tex(summary: Mapping[str, Any], dest: Path) -> None:
    per_v = summary.get("per_task") or {}
    comp = summary.get("comparison") or {}
    shell = (comp.get("llama_broca_shell") or {}).get("per_task") or {}
    tasks = sorted(set(per_v.keys()) & set(shell.keys()))
    if not tasks:
        dest.write_text("% (no paired Broca-shell comparison in summary)\n", encoding="utf-8")
        return
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\hline",
        r"Task & $n$ & Vanilla & Broca shell & $\Delta$ \\",
        r"\hline",
    ]
    for task in tasks:
        pv = per_v[task]
        ps = shell[task]
        if not isinstance(pv, Mapping) or not isinstance(ps, Mapping):
            continue
        acc_v = float(pv.get("accuracy", 0.0))
        acc_s = float(ps.get("accuracy", 0.0))
        n = int(pv.get("n", 0))
        safe_task = _latex_escape(str(task))
        lines.append(
            f"{safe_task} & {n} & {acc_v:.4f} & {acc_s:.4f} & {acc_s - acc_v:+.4f} \\\\",
        )
    lines.extend([r"\hline", r"\end{tabular}", ""])
    dest.write_text("\n".join(lines), encoding="utf-8")


def write_hf_native_experiment_tex(
    *,
    summary: Mapping[str, Any] | None,
    exp_dir: Path,
    error_message: str | None = None,
    accuracy_figure: str | None = None,
) -> None:
    path = exp_dir / "exp_hf_native_benchmark.tex"
    if error_message or summary is None:
        msg = _latex_escape(error_message or "benchmark not run")
        path.write_text(
            "\n".join(
                [
                    r"\subsection{HuggingFace dataset suite (native harness)}",
                    r"\label{subsec:exp-hf-native}",
                    "",
                    r"\paragraph{Status.}",
                    f"This block could not be refreshed automatically: \\texttt{{{msg}}}.",
                    r"Run \texttt{make paper-bench} with benchmark extras and \texttt{HF\_TOKEN} if needed.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return

    mid = _latex_escape(str(summary.get("model_id", "")))
    preset_tasks = list(summary.get("tasks") or [])
    tasks_display = _latex_escape(", ".join(preset_tasks))
    limit = summary.get("limit_per_task")
    seed = summary.get("seed", 0)
    stamp = _latex_escape(str(summary.get("created_at_utc", "")))
    scoring = summary.get("scoring") or {}
    mc_rule = _latex_escape(str(scoring.get("multiple_choice", "")))
    gen_rule = _latex_escape(str(scoring.get("generation", "")))

    blocks: list[str] = [
        r"\subsection{HuggingFace dataset suite (native harness)}",
        r"\label{subsec:exp-hf-native}",
        "",
        r"\paragraph{Description.}",
        (
            "We score publicly available multiple-choice and short-answer suites using the "
            "in-repository HuggingFace \\texttt{datasets} loader. Multiple-choice items use "
            f"{mc_rule}; generation-style items use {gen_rule}. "
            "When the hosted checkpoint is Llama-class, the harness can replay the same items "
            "through a Broca-wrapped decoder for a paired shell comparison."
        ),
        "",
        r"\paragraph{Setup.}",
        f"Checkpoint \\texttt{{{mid}}}; tasks \\texttt{{{tasks_display}}}; "
        f"limit per task \\texttt{{{limit}}}; seed \\texttt{{{seed}}}; run UTC stamp \\texttt{{{stamp}}}.",
        "",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Per-task accuracy for the vanilla causal LM decoder.}",
        r"\label{tab:hf-native-vanilla}",
        r"\input{include/experiment/hf_native_vanilla_table}",
        r"\end{table}",
        "",
    ]

    if isinstance(summary.get("comparison"), Mapping):
        blocks.extend(
            [
                r"\begin{table}[htbp]",
                r"\centering",
                r"\caption{Paired comparison: vanilla decoder vs.\ LlamaBrocaHost shell (same weights).}",
                r"\label{tab:hf-native-broca-shell}",
                r"\input{include/experiment/hf_native_comparison_table}",
                r"\end{table}",
                "",
            ]
        )

    if accuracy_figure:
        blocks.extend(
            [
                r"\begin{figure}[htbp]",
                r"\centering",
                rf"\includegraphics[width=0.92\linewidth]{{{accuracy_figure}}}",
                r"\caption{Per-task accuracy (vanilla LM) for the native harness run backing this draft.}",
                r"\label{fig:hf-native-acc}",
                r"\end{figure}",
                "",
            ]
        )
    else:
        blocks.extend(
            [
                r"\paragraph{Figure.}",
                r"No per-task figure was exported (matplotlib may be unavailable in the benchmark environment).",
                "",
            ]
        )

    path.write_text("\n".join(blocks) + "\n", encoding="utf-8")


def write_broca_architecture_experiment_tex(payload: Mapping[str, Any] | None, exp_dir: Path) -> None:
    path = exp_dir / "exp_broca_architecture.tex"
    if payload is None:
        path.write_text(
            "\n".join(
                [
                    r"\subsection{Broca architecture probes}",
                    r"\label{subsec:exp-broca-arch}",
                    r"\paragraph{Status.} No results available (eval not run).",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return
    if payload.get("error"):
        err = _latex_escape(str(payload.get("error")))
        path.write_text(
            "\n".join(
                [
                    r"\subsection{Broca architecture probes}",
                    r"\label{subsec:exp-broca-arch}",
                    r"\paragraph{Status.}",
                    f"Eval error: \\texttt{{{err}}}.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return

    desc = _latex_escape(str(payload.get("description", "")))[:1200]
    mid = _latex_escape(str(payload.get("model_id", "")))
    metrics = payload.get("metrics") or {}
    base = metrics.get("baseline_bare_language_host") or {}
    enh = metrics.get("enhanced_broca_architecture") or {}
    delta = metrics.get("delta_enhanced_minus_baseline") or {}

    def _fmt(prefix: str, block: Mapping[str, Any]) -> str:
        se = float(block.get("speech_exact_accuracy", 0.0))
        ap = float(block.get("answer_present_accuracy", 0.0))
        return f"{prefix} & ${se:.3f}$ & ${ap:.3f}$ \\\\"

    lines = [
        r"\subsection{Broca architecture probes}",
        r"\label{subsec:exp-broca-arch}",
        "",
        r"\paragraph{Description.}",
        desc,
        "",
        r"\paragraph{Setup.}",
        f"Model \\texttt{{{mid}}}; scripted prompts with verbatim and content-based scoring "
        r"(see \texttt{core.benchmarks.architecture\_eval}).",
        "",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Micro-averaged probe metrics (baseline bare LM vs.\ full Broca stack).}",
        r"\label{tab:broca-arch-probes}",
        r"\begin{tabular}{lcc}",
        r"\hline",
        r"Condition & Speech-exact & Answer-present \\",
        r"\hline",
        _fmt("Baseline", base if isinstance(base, Mapping) else {}),
        _fmt("Broca-enhanced", enh if isinstance(enh, Mapping) else {}),
        _fmt(r"Delta (Broca $-$ baseline)", delta if isinstance(delta, Mapping) else {}),
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_experiment_inputs_manifest(exp_dir: Path) -> None:
    """Rewrite ``_inputs.tex`` with one ``\\input`` per ``exp_*.tex`` (sorted)."""

    out = exp_dir / "_inputs.tex"
    parts = sorted(exp_dir.glob("exp_*.tex"))
    lines = [
        "% Auto-generated by `python -m core.paper` / `make paper-bench`. Do not edit by hand.",
        "",
    ]
    for p in parts:
        lines.append(f"\\input{{include/experiment/{p.stem}}}")
    lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")


def refresh_paper_experiments(*, root: Path | None = None) -> dict[str, Any]:
    """Run standard benchmarks and sync TeX + tables + figures into ``paper/``."""

    _, exp_dir = paper_dirs(root)
    exp_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {"hf_native": None, "broca_architecture": None}

    model_id = os.environ.get("PAPER_MODEL") or os.environ.get("BENCHMARK_MODEL") or "meta-llama/Llama-3.2-1B-Instruct"
    preset = (os.environ.get("PAPER_NATIVE_PRESET") or "quick").strip()
    limit = _env_int("PAPER_BENCH_LIMIT", 50)
    bench_root = repo_root() if root is None else Path(root)
    run_staging = bench_root / "runs" / "paper_harness_staging"
    run_staging.mkdir(parents=True, exist_ok=True)

    hf_tok_raw = os.environ.get("HF_TOKEN")
    hf_token: str | bool | None = hf_tok_raw if hf_tok_raw and hf_tok_raw.strip() else None

    summary: dict[str, Any] | None = None
    native_err: str | None = None
    native_run_dir: Path | None = None

    skip = os.environ.get("PAPER_SKIP_NATIVE", "").strip().lower() in {"1", "true", "yes"}

    if not skip:
        try:
            from core.benchmarks.hf_datasets_eval import (
                DEFAULT_NATIVE_PRESETS,
                resolve_task_names,
                run_hf_datasets_benchmark,
            )
            from core.device_utils import pick_torch_device

            if preset not in DEFAULT_NATIVE_PRESETS:
                preset = "quick"
            tasks = resolve_task_names("", preset=preset)
            dev = os.environ.get("PAPER_DEVICE") or os.environ.get("BENCHMARK_DEVICE") or pick_torch_device(
                os.environ.get("ASI_DEVICE"),
            )
            lim = None if limit < 0 else int(limit)
            summary_obj = run_hf_datasets_benchmark(
                model_id=model_id,
                tasks=tasks,
                output_dir=run_staging,
                limit=lim,
                device=str(dev) if dev else None,
                hf_token=hf_token,
                seed=_env_int("PAPER_BENCH_SEED", _env_int("BENCHMARK_SEED", 0)),
            )
            if isinstance(summary_obj, dict):
                summary = summary_obj
                stamp = str(summary.get("created_at_utc", "")).strip()
                if stamp:
                    native_run_dir = run_staging / f"hf_native_{stamp}"
                else:
                    cand = sorted(run_staging.glob("hf_native_*"), key=lambda p: p.stat().st_mtime)
                    native_run_dir = cand[-1] if cand else None
            report["hf_native"] = {"ok": True, "run_dir": str(native_run_dir) if native_run_dir else None}
        except Exception as exc:  # pragma: no cover - env specific
            logger.exception("paper harness: native HF benchmark failed")
            native_err = str(exc)
            report["hf_native"] = {"ok": False, "error": native_err}
    else:
        native_err = "skipped (PAPER_SKIP_NATIVE)"
        report["hf_native"] = {"ok": False, "error": native_err}

    if summary is not None:
        write_vanilla_table_tex(summary, exp_dir / "hf_native_vanilla_table.tex")
        write_comparison_table_tex(summary, exp_dir / "hf_native_comparison_table.tex")
        pdf_src = native_run_dir / "accuracy_by_task.pdf" if native_run_dir else None
        png_src = native_run_dir / "accuracy_by_task.png" if native_run_dir else None
        dst_pdf = exp_dir / "hf_native_accuracy_by_task.pdf"
        if pdf_src and pdf_src.is_file():
            shutil.copyfile(pdf_src, dst_pdf)
        elif png_src and png_src.is_file():
            try:
                shutil.copyfile(png_src, exp_dir / "hf_native_accuracy_by_task.png")
            except OSError:
                pass
        if native_run_dir and native_run_dir.is_dir():
            try:
                shutil.copyfile(native_run_dir / "summary.json", exp_dir / "hf_native_summary.json")
            except OSError:
                pass
        fig_name: str | None = None
        if (exp_dir / "hf_native_accuracy_by_task.pdf").is_file():
            fig_name = "hf_native_accuracy_by_task.pdf"
        elif (exp_dir / "hf_native_accuracy_by_task.png").is_file():
            fig_name = "hf_native_accuracy_by_task.png"
        write_hf_native_experiment_tex(summary=summary, exp_dir=exp_dir, accuracy_figure=fig_name)
    else:
        write_hf_native_experiment_tex(summary=None, exp_dir=exp_dir, error_message=native_err)

    arch_payload: dict[str, Any] | None = None
    arch_skip = os.environ.get("PAPER_SKIP_ARCH_EVAL", "").strip().lower() in {"1", "true", "yes"}
    if arch_skip:
        report["broca_architecture"] = {"ok": False, "skipped": True}
        write_broca_architecture_experiment_tex(None, exp_dir)
    elif model_id.strip().lower().startswith("gpt2"):
        report["broca_architecture"] = {"ok": False, "skipped": True, "reason": "gpt2 backend"}
        write_broca_architecture_experiment_tex(None, exp_dir)
    else:
        try:
            from core.benchmarks.__main__ import run_broca_architecture_benchmark
            from core.device_utils import pick_torch_device

            dev = os.environ.get("PAPER_DEVICE") or os.environ.get("BENCHMARK_DEVICE") or pick_torch_device(
                os.environ.get("ASI_DEVICE"),
            )
            result = run_broca_architecture_benchmark(
                llama_model_id=model_id,
                device=str(dev) if dev else None,
                hf_token=hf_token,
                output_run_dir=exp_dir,
            )
            arch_payload = result if isinstance(result, dict) else None
            report["broca_architecture"] = {"ok": not bool(arch_payload and arch_payload.get("error"))}
        except Exception as exc:  # pragma: no cover
            logger.exception("paper harness: architecture eval failed")
            arch_payload = {"error": str(exc)}
            report["broca_architecture"] = {"ok": False, "error": str(exc)}
        write_broca_architecture_experiment_tex(arch_payload, exp_dir)

    write_experiment_inputs_manifest(exp_dir)
    return report
