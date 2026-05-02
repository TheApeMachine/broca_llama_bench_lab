"""Textual TUI for ``make bench``: live benchmark dashboard.

Run:

  python -m research_lab.tui.bench
  # or: python -m research_lab.tui.bench

The benchmark suite executes in a Textual ``@work(thread=True)`` worker so the
UI stays responsive. The benchmark code (``research_lab.benchmarks``) publishes
progress on the shared :mod:`core.system.event_bus` (``bench.suite.*``,
``bench.phase.*``, ``bench.task.*``, ``bench.example``,
``bench.arch_case.*``) and the TUI subscribes from the same in-process bus.

Layout, mirroring the brand identity of :mod:`core.tui.chat`:

* Header + suite status bar (engine, preset, model, device, elapsed).
* Left column: phase status panel, current-task progress bar, running tally.
* Center: live results :class:`DataTable` with one row per (arm, task), plus
  a streaming activity log of bench events and forwarded log records.
* Right column: aggregate vs broca comparison panel, architecture case
  scoreboard, suite summary card.

Threading: the worker calls :func:`research_lab.benchmarks.__main__.main` directly
(no subprocess), since both the bench code and the TUI live in the same
process and share ``get_default_bus()``.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import logging
import os
import sys
import time
from collections import deque
from typing import Any

try:
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, Vertical, VerticalScroll
    from textual.reactive import reactive
    from textual.widgets import (
        DataTable,
        Footer,
        Header,
        Label,
        ProgressBar,
        RichLog,
        Sparkline,
        Static,
    )
    from textual.worker import Worker, WorkerState
    from textual import work
    from rich.text import Text
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "bench TUI requires Textual. Install with:\n\n"
        "  uv sync --extra tui\n"
        "  # or: pip install -e \".[tui]\"\n"
    ) from exc

from core.infra.constants import BRAND, BRAND_BG, BRAND_DEEP, BRAND_SOFT, OFFLINE, ONLINE, WARNING

from core.cli import attach_core_logs_to_bus, configure_lab_session, default_bus, detach_core_log_handler
from core.system.event_bus import EventBus
from .state import StatePanel

logger = logging.getLogger(__name__)

# ``DataTable`` column keys (explicit keys avoid fragile positional indexing).
RESULT_COL_ARM = "arm"
RESULT_COL_TASK = "task"
RESULT_COL_N = "n"
RESULT_COL_ACC = "acc"
RESULT_COL_DELTA = "delta"
RESULT_COL_SECS = "secs"
RESULT_COL_STATUS = "status"


def _system_exit_code(exc: SystemExit) -> int:
    code = exc.code
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    try:
        return int(code)
    except (TypeError, ValueError):
        return 0


def _safe_float(val: Any, *, default: float = 0.0, field: str = "") -> float:
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        logger.warning("bench_tui: ignoring non-numeric float field%s: %r", f" {field}" if field else "", val)
        return default


def _safe_int(val: Any, *, default: int = 0, field: str = "") -> int:
    if val is None:
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        logger.warning("bench_tui: ignoring non-numeric int field%s: %r", f" {field}" if field else "", val)
        return default


class _LinePublisher(io.TextIOBase):
    """File-like wrapper that publishes complete lines onto an event bus."""

    def __init__(self, bus: EventBus, topic: str) -> None:
        super().__init__()
        self._bus = bus
        self._topic = topic
        self._buf = ""

    def writable(self) -> bool:  # noqa: D401 - file-like API
        return True

    def write(self, s: str) -> int:
        if not s:
            return 0
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.rstrip("\r")
            if line:
                try:
                    self._bus.publish(self._topic, {"line": line})
                except Exception:
                    pass
        return len(s)

    def flush(self) -> None:
        if self._buf:
            try:
                self._bus.publish(self._topic, {"line": self._buf})
            except Exception:
                pass
            self._buf = ""

    def close(self) -> None:
        self.flush()
        super().close()


def _fmt_pct(v: float | None, prec: int = 1) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v) * 100:.{prec}f}%"
    except (TypeError, ValueError):
        return str(v)


def _fmt_delta(v: float | None, prec: int = 3) -> str:
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    sign = "+" if f >= 0 else ""
    return f"{sign}{f:.{prec}f}"


def _fmt_float(v: Any, prec: int = 3) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.{prec}f}"
    except (TypeError, ValueError):
        return str(v)


# ---------------------------------------------------------------------------
# The app
# ---------------------------------------------------------------------------


class BenchApp(App):
    """Real-time dashboard for ``python -m research_lab.benchmarks``."""

    CSS = f"""
    Screen {{
        layout: vertical;
        background: $surface;
    }}
    Header {{
        background: {BRAND_DEEP};
        color: $text;
    }}
    Footer {{
        background: {BRAND_DEEP};
    }}
    #suitebar {{
        height: 1;
        background: {BRAND} 25%;
        color: $text;
        padding: 0 1;
    }}
    #main {{
        height: 1fr;
    }}
    #left {{
        width: 36;
        min-width: 32;
        padding: 1;
        border-right: solid {BRAND} 40%;
    }}
    #right {{
        width: 46;
        min-width: 40;
        padding: 1;
        border-left: solid {BRAND} 40%;
    }}
    #center {{
        width: 1fr;
        padding: 0 1;
    }}
    #results {{
        height: 1fr;
        border: round {BRAND} 70%;
    }}
    #activity {{
        height: 14;
        border: round {BRAND} 40%;
    }}
    #status {{
        height: 1;
        background: {BRAND} 20%;
        color: $text;
        padding: 0 1;
    }}
    #task-progress {{
        margin: 1 0;
    }}
    ProgressBar > .bar--bar {{
        color: {BRAND};
    }}
    ProgressBar > .bar--complete {{
        color: {ONLINE};
    }}
    Sparkline {{
        height: 3;
        margin-bottom: 1;
    }}
    Sparkline > .sparkline--max-color {{
        color: {BRAND};
    }}
    Sparkline > .sparkline--min-color {{
        color: {BRAND_SOFT};
    }}
    DataTable > .datatable--header {{
        background: {BRAND_DEEP};
        color: $text;
    }}
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+d", "quit", "Quit"),
    ]

    busy: reactive[bool] = reactive(False)

    def __init__(
        self,
        *,
        bus: EventBus,
        bench_argv: list[str],
    ) -> None:
        super().__init__()
        self.bus = bus
        self.bench_argv = list(bench_argv)
        self._sub_id = self.bus.subscribe("*")
        self._suite_started_at: float | None = None
        self._suite_done: bool = False
        self._current_phase: str | None = None
        self._current_arm: str | None = None
        self._current_task: str | None = None
        self._current_label: str | None = None
        self._current_total: int = 0
        self._current_i: int = 0
        # (arm, task) -> dict
        self._results: dict[tuple[str, str], dict[str, Any]] = {}
        self._row_keys: dict[tuple[str, str], Any] = {}
        # arm -> (n, correct)
        self._totals: dict[str, list[int]] = {}
        self._arch_cases: list[dict[str, Any]] = []
        self._native_summary: dict[str, Any] | None = None
        self._lm_eval_summary: dict[str, Any] | None = None
        self._arch_summary: dict[str, Any] | None = None
        self._acc_trend: deque[float] = deque(maxlen=80)
        # --- live cognition / organ state for the new panels --------------
        self._last_intent: dict[str, Any] | None = None
        self._intent_label_counts: dict[str, int] = {}
        self._last_derived_strength: dict[str, Any] | None = None
        self._last_relation_extract: dict[str, Any] | None = None
        self._relation_outcome_counts: dict[str, int] = {}
        self._last_predictive_coding: dict[str, Any] | None = None
        self._hypothesis_state: dict[str, Any] = {
            "running": False,
            "iteration": 0,
            "max_iterations": 0,
            "n_bans": 0,
            "n_completed": 0,
            "n_accepted": 0,
            "last_text": "",
            "last_reason": "",
        }
        self._epistemic_state: dict[str, Any] = {
            "running": False,
            "step": 0,
            "max_new_tokens": 0,
            "n_interventions": 0,
            "last_reason": "",
            "n_completed": 0,
        }
        self._modality_state: dict[str, Any] = {
            "active": None,
            "modes": [],
            "n_registered": 0,
        }
        self._causal_state: dict[str, Any] = {
            "n_constraints": 0,
            "last": None,
        }
        # organ name -> {"calls": int, "avg_ms": float, "last_ms": float, "method": str}
        self._organ_stats: dict[str, dict[str, Any]] = {}
        self._loaded_organs: dict[str, dict[str, Any]] = {}
        self._last_affect: dict[str, Any] | None = None
        self._last_extraction: dict[str, Any] | None = None
        self._last_perception: dict[str, dict[str, Any]] = {}
        self._last_binding: dict[str, Any] | None = None
        self._last_auditory: dict[str, Any] | None = None
        # substrate (already-published) topics
        self._last_frame: dict[str, Any] | None = None
        self._last_intrinsic_cue: dict[str, Any] | None = None
        self._last_dmn_tick: dict[str, Any] | None = None
        self._dmn_ticks_seen: int = 0
        self._consolidations_seen: int = 0
        self._last_chat_complete: dict[str, Any] | None = None

    # ------------------------------------------------------------------ layout
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("", id="suitebar")
        with Horizontal(id="main"):
            with VerticalScroll(id="left"):
                yield StatePanel("Suite", id="panel-suite")
                yield StatePanel("Current phase", id="panel-phase")
                yield Label("Current task", classes="dim")
                yield ProgressBar(total=100, id="task-progress", show_eta=True)
                yield StatePanel("Running tally", id="panel-tally")
                yield Sparkline([0.0], id="spark-acc")
                yield Label("Per-example running accuracy", classes="dim")
            with Vertical(id="center"):
                yield DataTable(id="results", zebra_stripes=True, cursor_type="row")
                yield RichLog(id="activity", wrap=False, markup=True, highlight=False)
            with VerticalScroll(id="right"):
                yield StatePanel("Native aggregate", id="panel-native")
                yield StatePanel("Vanilla vs Broca shell", id="panel-compare")
                yield StatePanel("Architecture eval", id="panel-arch")
                yield StatePanel("LM-eval parity", id="panel-lmeval")
                yield StatePanel("Substrate (DMN / frames)", id="panel-substrate")
                yield StatePanel("Cognition (intent / strength / relation)", id="panel-cognition")
                yield StatePanel("Top-down control", id="panel-topdown")
                yield StatePanel("Organs", id="panel-organs")
                yield StatePanel("Suite summary", id="panel-summary")
        yield Static("", id="status")
        yield Footer()

    # ------------------------------------------------------------------ lifecycle
    def on_mount(self) -> None:
        self.title = "Mosaic benchmarks"
        self.sub_title = " ".join(self.bench_argv) if self.bench_argv else "default"
        results = self.query_one("#results", DataTable)
        # Per Textual docs, ``add_columns`` takes either bare labels (auto-keyed)
        # or ``(label, key)`` tuples for stable keys we can target with
        # ``update_cell`` later. Passing a ``Column`` dataclass triggers a
        # silent wrong-branch in ``add_columns`` and corrupts column metadata.
        results.add_columns(
            ("arm", RESULT_COL_ARM),
            ("task", RESULT_COL_TASK),
            ("n", RESULT_COL_N),
            ("acc", RESULT_COL_ACC),
            ("Δ", RESULT_COL_DELTA),
            ("secs", RESULT_COL_SECS),
            ("status", RESULT_COL_STATUS),
        )
        self.set_interval(0.25, self._tick)
        self.set_interval(1.0, self._refresh_status)
        self._refresh_panels_static()
        self._refresh_status()
        self._kick_off()

    def on_unmount(self) -> None:
        try:
            self.bus.unsubscribe(self._sub_id)
        except Exception:
            pass

    @work(thread=True, exclusive=True)
    def _kick_off(self) -> None:
        # Run the unified ``research_lab.benchmarks`` entrypoint in-process (same bus).
        from research_lab.benchmarks.__main__ import main as bench_main

        self.app.call_from_thread(self._on_suite_starting)
        out_stream = _LinePublisher(self.bus, "bench.stdout")
        err_stream = _LinePublisher(self.bus, "bench.stderr")
        try:
            with contextlib.redirect_stdout(out_stream), contextlib.redirect_stderr(err_stream):
                try:
                    bench_main(list(self.bench_argv) if self.bench_argv else [])
                except SystemExit as exc:
                    self.app.call_from_thread(self._on_suite_systemexit, _system_exit_code(exc))
                    return
                except Exception as exc:
                    logger.exception("bench TUI: bench_main failed")
                    self.app.call_from_thread(self._on_suite_error, str(exc))
                    return
        finally:
            try:
                out_stream.flush()
                err_stream.flush()
                out_stream.close()
                err_stream.close()
            except Exception:
                pass
        self.app.call_from_thread(self._on_suite_finished)

    # ------------------------------------------------------------------ tick
    def _tick(self) -> None:
        events = self.bus.drain(self._sub_id)
        if not events:
            self._refresh_phase_panel()
            return
        activity = self.query_one("#activity", RichLog)
        for ev in events:
            self._handle_event(ev.topic, ev.payload or {}, ev.ts, activity)
        self._refresh_phase_panel()
        self._refresh_summary_panels()

    def _handle_event(self, topic: str, payload: dict[str, Any], ts: float, activity: RichLog) -> None:
        ts_s = time.strftime("%H:%M:%S", time.localtime(ts))
        if topic == "bench.suite.start":
            self._suite_started_at = ts
            engine = payload.get("engine")
            preset = payload.get("preset")
            model = payload.get("model")
            limit = payload.get("limit")
            activity.write(
                f"[{BRAND}]{ts_s}[/{BRAND}] suite start [b]{engine}[/b] preset={preset} "
                f"limit={limit} model={model}"
            )
        elif topic == "bench.suite.complete":
            self._suite_done = True
            activity.write(
                f"[{ONLINE}]{ts_s}[/{ONLINE}] suite complete  manifest={payload.get('manifest_dir')}"
            )
        elif topic == "bench.phase.start":
            phase = str(payload.get("phase", ""))
            arm = payload.get("arm")
            self._current_phase = phase
            self._current_arm = arm
            self._current_task = None
            self._current_total = 0
            self._current_i = 0
            self._reset_progress()
            label = phase + (f"/{arm}" if arm else "")
            activity.write(f"[{BRAND_SOFT}]{ts_s}[/{BRAND_SOFT}] phase start [b]{label}[/b]")
        elif topic == "bench.phase.complete":
            phase = str(payload.get("phase", ""))
            err = payload.get("error")
            color = "red" if err else ONLINE
            extra = f" error={err}" if err else ""
            activity.write(f"[{color}]{ts_s}[/{color}] phase complete [b]{phase}[/b]{extra}")
            if phase == "native":
                self._native_summary = payload
            elif phase == "lm_eval":
                self._lm_eval_summary = payload
            elif phase == "architecture_eval":
                self._arch_summary = payload
        elif topic == "bench.task.start":
            self._current_task = str(payload.get("task") or "")
            self._current_label = str(payload.get("label") or self._current_task)
            self._current_total = _safe_int(payload.get("total"), default=0, field="total")
            self._current_i = 0
            self._reset_progress(total=self._current_total)
            activity.write(
                f"[{BRAND}]{ts_s}[/{BRAND}] task start  {self._current_label}"
                f"  n={self._current_total}"
            )
            arm = self._current_arm or "vanilla_lm"
            self._upsert_row(arm, self._current_task, n=0, acc=None, secs=None, status="running")
        elif topic == "bench.example":
            self._current_i = _safe_int(payload.get("i"), default=0, field="i")
            running_acc = payload.get("running_acc")
            self._update_progress(self._current_i, self._current_total)
            if running_acc is not None:
                try:
                    self._acc_trend.append(float(running_acc))
                except (TypeError, ValueError):
                    pass
        elif topic == "bench.task.complete":
            arm = str(payload.get("arm") or self._current_arm or "vanilla_lm")
            task = str(payload.get("task") or "")
            n = _safe_int(payload.get("n"), default=0, field="n")
            acc = _safe_float(payload.get("accuracy"), default=0.0, field="accuracy")
            secs = _safe_float(payload.get("seconds"), default=0.0, field="seconds")
            correct = _safe_int(payload.get("correct"), default=0, field="correct")
            self._upsert_row(arm, task, n=n, acc=acc, secs=secs, status="done")
            self._update_arm_totals(arm, n=n, correct=correct)
            activity.write(
                f"[{ONLINE}]{ts_s}[/{ONLINE}] task done    "
                f"{arm}:{task}  n={n} acc={_fmt_pct(acc)} ({secs:.1f}s)"
            )
            # Compute deltas if both arms present.
            self._refresh_deltas(task)
        elif topic == "bench.arch_case.start":
            activity.write(
                f"[{BRAND_SOFT}]{ts_s}[/{BRAND_SOFT}] arch case  "
                f"{payload.get('case_id')}  ({payload.get('i')}/{payload.get('total')})"
            )
        elif topic == "bench.arch_case.complete":
            self._arch_cases.append(dict(payload))
            be = "✓" if payload.get("baseline_speech_exact") else "·"
            ee = "✓" if payload.get("enhanced_speech_exact") else "·"
            activity.write(
                f"[{BRAND}]{ts_s}[/{BRAND}] arch result  {payload.get('case_id'):<22}"
                f"  baseline={be}  enhanced={ee}"
            )
        elif topic.startswith("log."):
            level = payload.get("level", "INFO")
            msg = payload.get("msg", "")
            color = {"DEBUG": "dim", "INFO": "white", "WARNING": WARNING, "ERROR": "red", "CRITICAL": "red"}.get(level, "white")
            name = payload.get("name", "")
            activity.write(f"[{color}]{ts_s} {level:7} {name}[/{color}]  {msg}")
        elif topic == "bench.stdout":
            line = str(payload.get("line", ""))
            if line.strip():
                activity.write(f"[dim]{ts_s}[/dim]  {line}")
        elif topic == "bench.stderr":
            line = str(payload.get("line", ""))
            if line.strip():
                activity.write(f"[{WARNING}]{ts_s}[/{WARNING}]  {line}")
        # ------------------ substrate-emitted topics ----------------------
        elif topic == "frame.comprehend":
            self._last_frame = dict(payload)
            activity.write(
                f"[cyan]{ts_s}[/cyan] frame [b]{payload.get('intent') or '—'}[/b]"
                f"  conf={_fmt_float(payload.get('confidence'))}"
                f"  subject={payload.get('subject') or '—'}"
            )
        elif topic == "intrinsic_cue":
            self._last_intrinsic_cue = dict(payload)
            activity.write(
                f"[yellow]{ts_s}[/yellow] cue [b]{payload.get('faculty')}[/b]"
                f"  urgency={_fmt_float(payload.get('urgency'))}"
            )
        elif topic == "dmn.tick":
            self._last_dmn_tick = dict(payload)
            self._dmn_ticks_seen += 1
            duration_ms = float(payload.get("duration_ms") or 0.0)
            activity.write(
                f"[magenta]{ts_s}[/magenta] dmn tick {payload.get('iteration')}"
                f"  {duration_ms:.0f}ms  reflections={payload.get('reflections', 0)}"
            )
        elif topic == "consolidation":
            self._consolidations_seen += 1
            activity.write(
                f"[green]{ts_s}[/green] consolidation reflections={payload.get('reflections', 0)}"
            )
        elif topic == "chat.start":
            activity.write(
                f"[{BRAND_SOFT}]{ts_s}[/{BRAND_SOFT}] chat start  intent={payload.get('intent') or '—'}"
            )
        elif topic == "chat.complete":
            self._last_chat_complete = dict(payload)
            activity.write(
                f"[{ONLINE}]{ts_s}[/{ONLINE}] chat done   intent={payload.get('intent') or '—'}"
                f"  conf={_fmt_float(payload.get('confidence'))}"
            )
        # ------------------ cognition: gating / strength / extraction ------
        elif topic == "cog.intent":
            self._last_intent = dict(payload)
            label = str(payload.get("label") or "—")
            self._intent_label_counts[label] = self._intent_label_counts.get(label, 0) + 1
            actionable = "✓" if payload.get("is_actionable") else "·"
            activity.write(
                f"[cyan]{ts_s}[/cyan] intent [b]{label}[/b]"
                f"  conf={_fmt_float(payload.get('confidence'))}  actionable={actionable}"
            )
        elif topic == "cog.derived_strength":
            self._last_derived_strength = dict(payload)
            activity.write(
                f"[cyan]{ts_s}[/cyan] strength={_fmt_float(payload.get('strength'))}"
                f"  intent={_fmt_float(payload.get('intent_actionability'))}"
                f"  mem={_fmt_float(payload.get('memory_confidence'))}"
                f"  sharp={_fmt_float(payload.get('conformal_sharpness'))}"
                f"  affect={_fmt_float(payload.get('affect_certainty'))}"
                f"  weakest={payload.get('gated_by') or '—'}"
            )
        elif topic == "cog.relation_extract":
            self._last_relation_extract = dict(payload)
            outcome = str(payload.get("outcome") or "")
            self._relation_outcome_counts[outcome] = self._relation_outcome_counts.get(outcome, 0) + 1
            color = {"extracted": ONLINE, "gated_out": WARNING, "no_relations": "dim"}.get(outcome, "white")
            if outcome == "extracted":
                activity.write(
                    f"[{color}]{ts_s}[/{color}] relation [{payload.get('subject')!s} -{payload.get('predicate')!s}-> {payload.get('object')!s}]"
                    f"  conf={_fmt_float(payload.get('claim_confidence'))}"
                )
            else:
                activity.write(
                    f"[{color}]{ts_s}[/{color}] relation {outcome}  intent={payload.get('intent_label') or '—'}"
                )
        elif topic == "cog.predictive_coding":
            self._last_predictive_coding = dict(payload)
            activity.write(
                f"[cyan]{ts_s}[/cyan] pred-coding gap={_fmt_float(payload.get('gap'))}"
                f"  ce_g={_fmt_float(payload.get('ce_graft'))}  ce_p={_fmt_float(payload.get('ce_plain'))}"
                f"  n={payload.get('n_targets')}  path={payload.get('path')}"
            )
        # ------------------ cognition: top-down control --------------------
        elif topic == "cog.hypothesis.start":
            self._hypothesis_state.update(
                running=True,
                iteration=0,
                max_iterations=int(payload.get("max_iterations") or 0),
                last_text="",
                last_reason="",
            )
            activity.write(
                f"[{BRAND}]{ts_s}[/{BRAND}] hypothesis search start"
                f"  max_iter={payload.get('max_iterations')}  prompt_len={payload.get('prompt_len')}"
            )
        elif topic == "cog.hypothesis.attempt":
            self._hypothesis_state["iteration"] = int(payload.get("iteration") or 0)
            self._hypothesis_state["last_text"] = str(payload.get("text") or "")
            self._hypothesis_state["last_reason"] = str(payload.get("reason") or "")
            mark = "✓" if payload.get("valid") else "·"
            activity.write(
                f"[{BRAND}]{ts_s}[/{BRAND}] hyp attempt {payload.get('iteration')}"
                f"  {mark} text={payload.get('text')!r}  bans={len(payload.get('ban_tokens') or [])}"
            )
        elif topic == "cog.hypothesis.ban":
            self._hypothesis_state["n_bans"] = int(payload.get("total_banned") or 0)
            activity.write(
                f"[{WARNING}]{ts_s}[/{WARNING}] hyp ban tokens={payload.get('tokens')}"
                f"  total={payload.get('total_banned')}  reason={payload.get('reason')!r}"
            )
        elif topic == "cog.hypothesis.complete":
            accepted = bool(payload.get("accepted"))
            self._hypothesis_state.update(
                running=False,
                iteration=int(payload.get("iterations") or 0),
            )
            self._hypothesis_state["n_completed"] += 1
            if accepted:
                self._hypothesis_state["n_accepted"] += 1
            color = ONLINE if accepted else "red"
            tag = "accepted" if accepted else "exhausted"
            activity.write(
                f"[{color}]{ts_s}[/{color}] hyp {tag}  iters={payload.get('iterations')}"
                f"  text={payload.get('final_text')!r}"
            )
        elif topic == "cog.epistemic.start":
            self._epistemic_state.update(
                running=True,
                step=0,
                max_new_tokens=int(payload.get("max_new_tokens") or 0),
                n_interventions=0,
                last_reason="",
            )
            activity.write(
                f"[{BRAND}]{ts_s}[/{BRAND}] epistemic monitor start"
                f"  max_new={payload.get('max_new_tokens')}  check_every={payload.get('check_every')}"
            )
        elif topic == "cog.epistemic.intervention":
            self._epistemic_state["step"] = int(payload.get("step") or 0)
            self._epistemic_state["n_interventions"] = int(payload.get("intervention_index") or 0)
            self._epistemic_state["last_reason"] = str(payload.get("reason") or "")
            activity.write(
                f"[{WARNING}]{ts_s}[/{WARNING}] epistemic halt step={payload.get('step')}"
                f"  trunc={payload.get('truncated')}  bans={len(payload.get('banned_tokens') or [])}"
                f"  reason={payload.get('reason')!r}"
            )
        elif topic == "cog.epistemic.complete":
            self._epistemic_state.update(
                running=False,
                step=int(payload.get("final_step") or 0),
                n_interventions=int(payload.get("n_interventions") or 0),
            )
            self._epistemic_state["n_completed"] = self._epistemic_state.get("n_completed", 0) + 1
            activity.write(
                f"[{ONLINE}]{ts_s}[/{ONLINE}] epistemic done  steps={payload.get('final_step')}"
                f"  interventions={payload.get('n_interventions')}"
            )
        elif topic == "cog.modality_shift.register":
            self._modality_state["n_registered"] = int(payload.get("n_modes") or 0)
            modes = list(self._modality_state.get("modes") or [])
            name = payload.get("name")
            if name and name not in modes:
                modes.append(name)
            self._modality_state["modes"] = modes
            activity.write(
                f"[cyan]{ts_s}[/cyan] modality register name={name}  total={payload.get('n_modes')}"
            )
        elif topic == "cog.modality_shift.set_active":
            self._modality_state["active"] = payload.get("name")
            activity.write(
                f"[cyan]{ts_s}[/cyan] modality active={payload.get('name')}"
            )
        elif topic == "cog.causal.constraint":
            self._causal_state["n_constraints"] = int(payload.get("n_constraints") or 0)
            self._causal_state["last"] = dict(payload)
            activity.write(
                f"[cyan]{ts_s}[/cyan] causal constraint  do({payload.get('treatment')}={payload.get('treatment_value')!r})"
                f" → {payload.get('outcome')}  total={payload.get('n_constraints')}"
            )
        # ------------------ organs ----------------------------------------
        elif topic == "organ.load":
            self._loaded_organs[str(payload.get("name"))] = {
                "model_id": payload.get("model_id"),
                "device": payload.get("device"),
                "load_ms": float(payload.get("load_ms") or 0.0),
            }
            activity.write(
                f"[cyan]{ts_s}[/cyan] organ load  {payload.get('name')}  "
                f"({payload.get('model_id')})  {float(payload.get('load_ms') or 0.0):.0f}ms"
            )
        elif topic == "organ.unload":
            self._loaded_organs.pop(str(payload.get("name")), None)
            activity.write(f"[dim]{ts_s}[/dim] organ unload {payload.get('name')}")
        elif topic == "organ.call":
            name = str(payload.get("name") or "—")
            stats = self._organ_stats.setdefault(name, {"calls": 0, "avg_ms": 0.0, "last_ms": 0.0, "method": ""})
            stats["calls"] = int(payload.get("total_calls") or stats["calls"] + 1)
            stats["avg_ms"] = float(payload.get("avg_latency_ms") or stats["avg_ms"])
            stats["last_ms"] = float(payload.get("latency_ms") or 0.0)
            stats["method"] = str(payload.get("method") or "process")
        elif topic == "organ.affect":
            self._last_affect = dict(payload)
            activity.write(
                f"[cyan]{ts_s}[/cyan] affect [b]{payload.get('dominant_emotion')}[/b]"
                f"  score={_fmt_float(payload.get('dominant_score'))}"
                f"  val={_fmt_float(payload.get('valence'))}"
                f"  ar={_fmt_float(payload.get('arousal'))}"
                f"  pref={payload.get('preference_signal') or '—'}"
            )
        elif topic.startswith("organ.extraction."):
            self._last_extraction = {"topic": topic, **dict(payload)}
            kind = topic.split(".")[-1]
            n_key = {"entities": "n_entities", "relations": "n_relations", "classify": "selected"}.get(kind)
            count = payload.get(n_key) if n_key else None
            if isinstance(count, list):
                count_str = str(len(count))
            else:
                count_str = str(count) if count is not None else "—"
            activity.write(
                f"[cyan]{ts_s}[/cyan] extraction.{kind}  n={count_str}"
                f"  ms={_fmt_float(payload.get('latency_ms'), prec=0)}"
            )
        elif topic.startswith("organ.perception."):
            stream = topic.split(".")[-1]
            self._last_perception[stream] = dict(payload)
            activity.write(
                f"[cyan]{ts_s}[/cyan] perception.{stream}  ms={_fmt_float(payload.get('latency_ms'), prec=0)}"
            )
        elif topic == "organ.binding.encode":
            self._last_binding = dict(payload)
            activity.write(
                f"[cyan]{ts_s}[/cyan] binding {payload.get('modality')}"
                f"  ms={_fmt_float(payload.get('latency_ms'), prec=0)}"
            )
        elif topic.startswith("organ.auditory."):
            kind = topic.split(".")[-1]
            self._last_auditory = {"topic": topic, **dict(payload)}
            extra = (
                f" text={payload.get('transcription')!r}" if kind == "transcribe" else f" frames={payload.get('n_frames')}"
            )
            activity.write(
                f"[cyan]{ts_s}[/cyan] auditory.{kind}{extra}"
                f"  ms={_fmt_float(payload.get('latency_ms'), prec=0)}"
            )
        # else: ignore unknown topics

    # ------------------------------------------------------------------ helpers
    def _reset_progress(self, total: int = 100) -> None:
        try:
            bar = self.query_one("#task-progress", ProgressBar)
        except Exception:
            return
        bar.update(total=max(1, int(total)), progress=0)

    def _update_progress(self, i: int, total: int) -> None:
        try:
            bar = self.query_one("#task-progress", ProgressBar)
        except Exception:
            return
        if total > 0:
            bar.update(total=int(total), progress=int(i))

    def _arm_key(self, arm: str | None) -> str:
        return arm or "vanilla_lm"

    def _row_key(self, arm: str | None, task: str) -> tuple[str, str]:
        return (self._arm_key(arm), task)

    def _upsert_row(
        self,
        arm: str,
        task: str,
        *,
        n: int,
        acc: float | None,
        secs: float | None,
        status: str,
    ) -> None:
        table = self.query_one("#results", DataTable)
        key = self._row_key(arm, task)
        prior = self._results.get(key, {})
        row = {
            "arm": arm,
            "task": task,
            "n": n,
            "acc": acc if acc is not None else prior.get("acc"),
            "secs": secs if secs is not None else prior.get("secs"),
            "status": status,
            "delta": prior.get("delta"),
        }
        self._results[key] = row
        cells = (
            arm,
            task,
            str(n) if n else (str(prior.get("n", 0)) if prior else "—"),
            _fmt_pct(row["acc"]) if row["acc"] is not None else "…",
            _fmt_delta(row["delta"]) if row["delta"] is not None else "—",
            f"{secs:.1f}s" if secs is not None else (f"{prior.get('secs'):.1f}s" if prior.get("secs") else "—"),
            self._render_status(status),
        )
        if key in self._row_keys:
            rk = self._row_keys[key]
            mapping = (
                (RESULT_COL_ARM, cells[0]),
                (RESULT_COL_TASK, cells[1]),
                (RESULT_COL_N, cells[2]),
                (RESULT_COL_ACC, cells[3]),
                (RESULT_COL_DELTA, cells[4]),
                (RESULT_COL_SECS, cells[5]),
                (RESULT_COL_STATUS, cells[6]),
            )
            for col_key, val in mapping:
                try:
                    table.update_cell(rk, col_key, val)
                except Exception:
                    pass
        else:
            self._row_keys[key] = table.add_row(*cells)

    def _render_status(self, status: str) -> Text:
        color = {
            "running": BRAND,
            "done": ONLINE,
            "error": "red",
            "queued": OFFLINE,
        }.get(status, "white")
        return Text(status, style=color)

    def _refresh_deltas(self, task: str) -> None:
        v = self._results.get(("vanilla_lm", task))
        if not v or v.get("acc") is None:
            return
        try:
            acc_v = float(v["acc"])
        except (TypeError, ValueError):
            return
        table = self.query_one("#results", DataTable)
        for arm in ("broca_shell", "broca_mind"):
            b = self._results.get((arm, task))
            if not b or b.get("acc") is None:
                continue
            try:
                delta = float(b["acc"]) - acc_v
            except (TypeError, ValueError):
                continue
            b["delta"] = delta
            rk = self._row_keys.get((arm, task))
            if rk is not None:
                try:
                    table.update_cell(rk, RESULT_COL_DELTA, _fmt_delta(delta))
                except Exception:
                    pass

    def _update_arm_totals(self, arm: str, *, n: int, correct: int) -> None:
        cur = self._totals.setdefault(arm, [0, 0])
        cur[0] += int(n)
        cur[1] += int(correct)

    # ------------------------------------------------------------------ panels
    def _refresh_panels_static(self) -> None:
        suite_lines = [
            f"argv: [b]{' '.join(self.bench_argv) or 'default'}[/b]",
            "[dim]bench runs in this process; events stream from the shared bus[/dim]",
        ]
        self.query_one("#panel-suite", StatePanel).set_lines(suite_lines)
        for pid in (
            "panel-phase",
            "panel-tally",
            "panel-native",
            "panel-compare",
            "panel-arch",
            "panel-lmeval",
            "panel-substrate",
            "panel-cognition",
            "panel-topdown",
            "panel-organs",
            "panel-summary",
        ):
            self.query_one(f"#{pid}", StatePanel).set_lines([])

    def _refresh_phase_panel(self) -> None:
        phase = self._current_phase or "—"
        arm = self._current_arm or "—"
        task = self._current_label or "—"
        progress = (
            f"{self._current_i}/{self._current_total}"
            if self._current_total
            else "—"
        )
        lines = [
            f"phase: [b]{phase}[/b]",
            f"arm:   {arm}",
            f"task:  {task}",
            f"prog:  {progress}",
        ]
        self.query_one("#panel-phase", StatePanel).set_lines(lines)
        # Tally
        tally_lines: list[str] = []
        for arm_name, (n, correct) in sorted(self._totals.items()):
            acc = correct / max(1, n)
            tally_lines.append(f"{arm_name:<14} n={n:5d}  acc={_fmt_pct(acc)}")
        if not tally_lines:
            tally_lines.append("[dim]no completed tasks yet[/dim]")
        self.query_one("#panel-tally", StatePanel).set_lines(tally_lines)
        # Sparkline
        if self._acc_trend:
            try:
                self.query_one("#spark-acc", Sparkline).data = list(self._acc_trend)
            except Exception:
                pass

    def _refresh_summary_panels(self) -> None:
        # Native
        if self._native_summary:
            ns = self._native_summary
            lines = [
                f"macro acc: [b]{_fmt_pct(ns.get('macro_accuracy'))}[/b]",
                f"micro acc: {_fmt_pct(ns.get('micro_accuracy'))}",
                f"comparison: {'on' if ns.get('comparison') else 'off'}",
                f"summary: [dim]{ns.get('summary_path') or '—'}[/dim]",
            ]
            self.query_one("#panel-native", StatePanel).set_lines(lines)
        # Vanilla vs Broca delta panel
        v_total = self._totals.get("vanilla_lm")
        b_total = self._totals.get("broca_shell")
        m_total = self._totals.get("broca_mind")
        comp_lines: list[str] = []
        if v_total:
            comp_lines.append(f"vanilla_lm   n={v_total[0]:5d}  acc={_fmt_pct(v_total[1] / max(1, v_total[0]))}")
        if b_total:
            comp_lines.append(f"broca_shell  n={b_total[0]:5d}  acc={_fmt_pct(b_total[1] / max(1, b_total[0]))}")
        if m_total:
            comp_lines.append(f"broca_mind   n={m_total[0]:5d}  acc={_fmt_pct(m_total[1] / max(1, m_total[0]))}")
        if v_total and b_total:
            v_acc = v_total[1] / max(1, v_total[0])
            b_acc = b_total[1] / max(1, b_total[0])
            d = b_acc - v_acc
            color = ONLINE if d >= 0 else WARNING
            comp_lines.append(f"[{color}]Δ shell       {_fmt_delta(d)}[/{color}]")
        if v_total and m_total:
            v_acc = v_total[1] / max(1, v_total[0])
            m_acc = m_total[1] / max(1, m_total[0])
            d2 = m_acc - v_acc
            color = ONLINE if d2 >= 0 else WARNING
            comp_lines.append(f"[{color}]Δ mind        {_fmt_delta(d2)}[/{color}]")
        if not comp_lines:
            comp_lines.append("[dim]waiting for first task[/dim]")
        self.query_one("#panel-compare", StatePanel).set_lines(comp_lines)
        # Architecture eval
        arch_lines: list[str] = []
        if self._arch_summary:
            base = self._arch_summary.get("baseline_speech_acc")
            enh = self._arch_summary.get("enhanced_speech_acc")
            d = self._arch_summary.get("delta_speech_acc")
            arch_lines.append(f"speech baseline: {_fmt_pct(base)}")
            arch_lines.append(f"speech enhanced: {_fmt_pct(enh)}")
            color = ONLINE if (d or 0) >= 0 else WARNING
            arch_lines.append(f"[{color}]Δ speech: {_fmt_delta(d)}[/{color}]")
        if self._arch_cases:
            arch_lines.append(f"cases: {len(self._arch_cases)}")
            for c in self._arch_cases[-4:]:
                be = "✓" if c.get("baseline_speech_exact") else "·"
                ee = "✓" if c.get("enhanced_speech_exact") else "·"
                arch_lines.append(f"  [dim]{c.get('case_id'):<24}[/dim] base={be} enh={ee}")
        if not arch_lines:
            arch_lines.append("[dim]not started[/dim]")
        self.query_one("#panel-arch", StatePanel).set_lines(arch_lines)
        # LM-eval
        lm_lines: list[str] = []
        if self._lm_eval_summary:
            err = self._lm_eval_summary.get("error")
            if err:
                err_str = err if isinstance(err, str) else str(err)
                lm_lines.append(f"[red]error: {err_str[:48]}[/red]")
            else:
                lm_lines.append(f"out: [dim]{self._lm_eval_summary.get('out')}[/dim]")
                lm_lines.append("[dim]see lm_eval_pair.json for per-task[/dim]")
        else:
            lm_lines.append("[dim]not run[/dim]")
        self.query_one("#panel-lmeval", StatePanel).set_lines(lm_lines)
        # Substrate (DMN, frames, cues, consolidation, chat)
        sub_lines: list[str] = []
        if self._last_frame:
            sub_lines.append(
                f"frame:    [b]{self._last_frame.get('intent') or '—'}[/b]"
                f"  conf={_fmt_float(self._last_frame.get('confidence'))}"
            )
            subj = self._last_frame.get("subject")
            if subj:
                sub_lines.append(f"  subject={subj}")
            ans = self._last_frame.get("answer")
            if ans:
                sub_lines.append(f"  answer={str(ans)[:48]}")
        if self._last_dmn_tick:
            sub_lines.append(
                f"dmn:      iter={self._last_dmn_tick.get('iteration')}"
                f"  reflections={self._last_dmn_tick.get('reflections', 0)}"
                f"  total_ticks={self._dmn_ticks_seen}"
            )
        if self._last_intrinsic_cue:
            sub_lines.append(
                f"cue:      [b]{self._last_intrinsic_cue.get('faculty') or '—'}[/b]"
                f"  urgency={_fmt_float(self._last_intrinsic_cue.get('urgency'))}"
            )
        if self._consolidations_seen:
            sub_lines.append(f"consolidations: {self._consolidations_seen}")
        if self._last_chat_complete:
            sub_lines.append(
                f"chat:     intent={self._last_chat_complete.get('intent') or '—'}"
                f"  conf={_fmt_float(self._last_chat_complete.get('confidence'))}"
            )
        if not sub_lines:
            sub_lines.append("[dim]no substrate events yet[/dim]")
        self.query_one("#panel-substrate", StatePanel).set_lines(sub_lines)

        # Cognition (intent gate, derived strength, relation extractor, predictive coding)
        cog_lines: list[str] = []
        if self._last_intent:
            actionable = "[green]act[/green]" if self._last_intent.get("is_actionable") else "[dim]non-act[/dim]"
            storable = "[green]store[/green]" if self._last_intent.get("allows_storage") else "[dim]no-store[/dim]"
            cog_lines.append(
                f"intent:   [b]{self._last_intent.get('label') or '—'}[/b]"
                f"  conf={_fmt_float(self._last_intent.get('confidence'))}  {actionable}/{storable}"
            )
            if self._intent_label_counts:
                top = sorted(self._intent_label_counts.items(), key=lambda kv: -kv[1])[:4]
                cog_lines.append(
                    "  hist: " + " ".join(f"{l}={n}" for l, n in top)
                )
        if self._last_derived_strength:
            ds = self._last_derived_strength
            cog_lines.append(
                f"strength: {_fmt_float(ds.get('strength'))}"
                f"  weakest={ds.get('gated_by') or '—'}"
            )
            cog_lines.append(
                f"  intent={_fmt_float(ds.get('intent_actionability'))}"
                f"  mem={_fmt_float(ds.get('memory_confidence'))}"
                f"  sharp={_fmt_float(ds.get('conformal_sharpness'))}"
                f"  affect={_fmt_float(ds.get('affect_certainty'))}"
            )
        if self._last_relation_extract:
            re = self._last_relation_extract
            outcome = re.get("outcome") or "—"
            color = {"extracted": ONLINE, "gated_out": WARNING, "no_relations": "dim"}.get(outcome, "white")
            cog_lines.append(f"relation: [{color}]{outcome}[/{color}]")
            if outcome == "extracted":
                cog_lines.append(
                    f"  ({re.get('subject')}, {re.get('predicate')}, {re.get('object')})"
                    f"  conf={_fmt_float(re.get('claim_confidence'))}"
                )
            if self._relation_outcome_counts:
                cog_lines.append(
                    "  hist: " + " ".join(
                        f"{k}={v}" for k, v in sorted(self._relation_outcome_counts.items())
                    )
                )
        if self._last_predictive_coding:
            pc = self._last_predictive_coding
            cog_lines.append(
                f"pred-coding: gap={_fmt_float(pc.get('gap'))}"
                f"  ce_g={_fmt_float(pc.get('ce_graft'))}  ce_p={_fmt_float(pc.get('ce_plain'))}"
            )
            cog_lines.append(
                f"  n_targets={pc.get('n_targets')}  path={pc.get('path')}"
            )
        if not cog_lines:
            cog_lines.append("[dim]no cognition events yet[/dim]")
        self.query_one("#panel-cognition", StatePanel).set_lines(cog_lines)

        # Top-down control (hypothesis search, epistemic monitor, modality, causal)
        td_lines: list[str] = []
        hs = self._hypothesis_state
        if hs["running"] or hs["n_completed"]:
            state_str = "[yellow]running[/yellow]" if hs["running"] else "idle"
            td_lines.append(
                f"hypothesis: {state_str}  iter={hs['iteration']}/{hs['max_iterations']}"
            )
            td_lines.append(
                f"  bans={hs['n_bans']}  done={hs['n_completed']}  accepted={hs['n_accepted']}"
            )
            if hs.get("last_text"):
                td_lines.append(f"  last={str(hs['last_text'])[:40]!r}")
        es = self._epistemic_state
        if es["running"] or es.get("n_completed"):
            state_str = "[yellow]running[/yellow]" if es["running"] else "idle"
            td_lines.append(
                f"epistemic:  {state_str}  step={es['step']}/{es['max_new_tokens']}"
            )
            td_lines.append(
                f"  interventions={es['n_interventions']}  runs={es.get('n_completed', 0)}"
            )
            if es.get("last_reason"):
                td_lines.append(f"  reason={str(es['last_reason'])[:40]!r}")
        if self._modality_state["n_registered"]:
            ms = self._modality_state
            active = ms.get("active") or "—"
            td_lines.append(
                f"modality:   active=[b]{active}[/b]  n_registered={ms['n_registered']}"
            )
            if ms.get("modes"):
                td_lines.append(f"  modes: {', '.join(ms['modes'][:6])}")
        if self._causal_state["n_constraints"]:
            cs = self._causal_state
            td_lines.append(f"causal:     constraints={cs['n_constraints']}")
            last = cs.get("last") or {}
            if last:
                td_lines.append(
                    f"  last: do({last.get('treatment')}={last.get('treatment_value')!r})"
                    f" → {last.get('outcome')}"
                )
        if not td_lines:
            td_lines.append("[dim]no top-down control events yet[/dim]")
        self.query_one("#panel-topdown", StatePanel).set_lines(td_lines)

        # Organs panel — per-organ call counters and last-emotion / last-perception
        org_lines: list[str] = []
        if self._loaded_organs:
            org_lines.append(f"loaded: {len(self._loaded_organs)}")
            for name, info in sorted(self._loaded_organs.items()):
                org_lines.append(
                    f"  [b]{name}[/b]  {str(info.get('model_id') or '')[:24]}"
                    f"  load={float(info.get('load_ms') or 0.0):.0f}ms"
                )
        if self._organ_stats:
            org_lines.append("activity:")
            for name, st in sorted(self._organ_stats.items(), key=lambda kv: -int(kv[1].get("calls", 0))):
                org_lines.append(
                    f"  {name:<18} n={st['calls']:4d}"
                    f"  avg={st['avg_ms']:5.0f}ms  last={st['last_ms']:5.0f}ms"
                    f"  [{st['method']}]"
                )
        if self._last_affect:
            af = self._last_affect
            org_lines.append(
                f"affect: [b]{af.get('dominant_emotion')}[/b]"
                f"  val={_fmt_float(af.get('valence'))}  ar={_fmt_float(af.get('arousal'))}"
            )
            pref = af.get("preference_signal")
            if pref:
                org_lines.append(
                    f"  pref={pref}  strength={_fmt_float(af.get('preference_strength'))}"
                )
        if self._last_extraction:
            org_lines.append(
                f"extraction: {self._last_extraction.get('topic', '').split('.')[-1]}"
            )
        if self._last_perception:
            for stream, info in sorted(self._last_perception.items()):
                org_lines.append(
                    f"perception.{stream}: ms={_fmt_float(info.get('latency_ms'), prec=0)}"
                )
        if self._last_binding:
            org_lines.append(
                f"binding: {self._last_binding.get('modality')}"
                f"  ms={_fmt_float(self._last_binding.get('latency_ms'), prec=0)}"
            )
        if self._last_auditory:
            org_lines.append(
                f"auditory: {str(self._last_auditory.get('topic', '')).split('.')[-1]}"
            )
        if not org_lines:
            org_lines.append("[dim]no organ activity yet[/dim]")
        self.query_one("#panel-organs", StatePanel).set_lines(org_lines)

        # Suite summary
        sum_lines: list[str] = []
        if self._suite_started_at:
            elapsed = time.time() - self._suite_started_at
            sum_lines.append(f"elapsed: {elapsed:6.1f}s")
        sum_lines.append(f"tasks done: {sum(1 for r in self._results.values() if r.get('status') == 'done')}")
        sum_lines.append(f"arch cases: {len(self._arch_cases)}")
        sum_lines.append(f"status: {'[green]complete[/green]' if self._suite_done else '[yellow]running[/yellow]'}")
        self.query_one("#panel-summary", StatePanel).set_lines(sum_lines)

    def _refresh_status(self) -> None:
        elapsed = (
            f"{time.time() - self._suite_started_at:6.1f}s"
            if self._suite_started_at
            else "  —"
        )
        state = "complete" if self._suite_done else ("running" if self._suite_started_at else "starting")
        argv_s = " ".join(self.bench_argv) if self.bench_argv else "(default)"
        suitebar = (
            f" mosaic-bench  state={state}  elapsed={elapsed}  argv={argv_s}"
        )
        try:
            self.query_one("#suitebar", Static).update(suitebar)
        except Exception:
            pass
        try:
            self.query_one("#status", Static).update(
                f" press ctrl+c to quit  ·  events stream live from core.system.event_bus"
            )
        except Exception:
            pass

    # ------------------------------------------------------------------ worker callbacks
    def _on_suite_starting(self) -> None:
        self._suite_started_at = time.time()
        self.busy = True
        try:
            self.query_one("#activity", RichLog).write(
                f"[{BRAND}]bench starting:[/{BRAND}] {' '.join(self.bench_argv) or '(default args)'}"
            )
        except Exception:
            pass

    def _on_suite_finished(self) -> None:
        self.busy = False
        self._suite_done = True
        try:
            self.query_one("#activity", RichLog).write(f"[{ONLINE}]bench finished[/{ONLINE}]")
        except Exception:
            pass
        self._refresh_summary_panels()

    def _on_suite_error(self, err: str) -> None:
        self.busy = False
        self._suite_done = True
        try:
            self.query_one("#activity", RichLog).write(f"[red]bench error: {err}[/red]")
        except Exception:
            pass

    def _on_suite_systemexit(self, code: int) -> None:
        self.busy = False
        self._suite_done = True
        try:
            color = ONLINE if code == 0 else "red"
            self.query_one("#activity", RichLog).write(
                f"[{color}]bench exited with code {code}[/{color}]"
            )
        except Exception:
            pass

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:  # noqa: ARG002
        if event.state in (WorkerState.SUCCESS, WorkerState.ERROR, WorkerState.CANCELLED):
            self.busy = False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
    description="Textual dashboard for `python -m core bench` / `python -m research_lab.benchmarks` (see -h below).",
    )


def run_bench_tui(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    helper = argparse.ArgumentParser(add_help=False)
    helper.add_argument("-h", "--help", action="store_true")
    hpre, trailing = helper.parse_known_args(argv)

    if hpre.help:
        parser = _build_parser()
        parser.print_help()
        print()
        from research_lab.benchmarks.__main__ import print_benchmark_cli_help

        print_benchmark_cli_help()

        return

    parser = _build_parser()
    _, benchmark_argv = parser.parse_known_args(trailing)

    os.environ.setdefault("LOG_SILENT", "1")
    os.environ.setdefault("MPLBACKEND", "Agg")

    configure_lab_session(silent_stderr_default=True)

    bus = default_bus()
    handler = attach_core_logs_to_bus(bus)

    try:
        app = BenchApp(bus=bus, bench_argv=list(benchmark_argv))
        app.run()
    finally:
        detach_core_log_handler(handler)


def main() -> None:
    """Entry point for ``python -m research_lab.tui.bench``."""

    run_bench_tui()


if __name__ == "__main__":
    main()
