"""Textual TUI for ``make bench``: live benchmark dashboard.

Run:

  python -m core.bench_tui --engine both --preset standard --limit 250

The benchmark suite executes in a Textual ``@work(thread=True)`` worker so the
UI stays responsive. The benchmark code (``core.benchmarks``) publishes
progress on the shared :mod:`core.event_bus` (``bench.suite.*``,
``bench.phase.*``, ``bench.task.*``, ``bench.example``,
``bench.arch_case.*``) and the TUI subscribes from the same in-process bus.

Layout, mirroring the brand identity of :mod:`core.chat_tui`:

* Header + suite status bar (engine, preset, model, device, elapsed).
* Left column: phase status panel, current-task progress bar, running tally.
* Center: live results :class:`DataTable` with one row per (arm, task), plus
  a streaming activity log of bench events and forwarded log records.
* Right column: aggregate vs broca comparison panel, architecture case
  scoreboard, suite summary card.

Threading: the worker calls :func:`core.benchmarks.__main__.main` directly
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
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "core.bench_tui requires Textual. Install with:\n\n"
        "  uv sync --extra tui\n"
        "  # or: pip install -e \".[tui]\"\n"
    ) from exc

from .chat_tui import (
    BRAND,
    BRAND_BG,
    BRAND_DEEP,
    BRAND_SOFT,
    OFFLINE,
    ONLINE,
    WARNING,
    StatePanel,
)
from .event_bus import EventBus, LogToBusHandler, get_default_bus
from .logging_setup import configure_lab_logging

logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# The app
# ---------------------------------------------------------------------------


class BenchApp(App):
    """Real-time dashboard for ``python -m core.benchmarks``."""

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
    #left, #right {{
        width: 36;
        min-width: 32;
        padding: 1;
        border-right: solid {BRAND} 40%;
    }}
    #right {{
        border-right: none;
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
                yield StatePanel("Suite summary", id="panel-summary")
        yield Static("", id="status")
        yield Footer()

    # ------------------------------------------------------------------ lifecycle
    def on_mount(self) -> None:
        self.title = "Mosaic benchmarks"
        self.sub_title = " ".join(self.bench_argv) if self.bench_argv else "default"
        results = self.query_one("#results", DataTable)
        results.add_columns("arm", "task", "n", "acc", "Δ", "secs", "status")
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
        # Run the same entry point as ``python -m core.benchmarks``. We wrap
        # argv so the bench parser sees the arguments the TUI was started with.
        from core.benchmarks.__main__ import main as bench_main

        self.app.call_from_thread(self._on_suite_starting)
        # Save and replace argv so argparse picks up our flags.
        saved = sys.argv[:]
        sys.argv = ["core.benchmarks"] + self.bench_argv
        # Redirect stdout/stderr so the bench's print() calls don't fight the
        # Textual full-screen app. Lines go through the bus as bench.print
        # events and surface in the activity log.
        out_stream = _LinePublisher(self.bus, "bench.stdout")
        err_stream = _LinePublisher(self.bus, "bench.stderr")
        try:
            with contextlib.redirect_stdout(out_stream), contextlib.redirect_stderr(err_stream):
                try:
                    bench_main()
                except SystemExit as exc:
                    self.app.call_from_thread(self._on_suite_systemexit, int(exc.code or 0))
                    return
                except Exception as exc:
                    logger.exception("bench TUI: bench_main failed")
                    self.app.call_from_thread(self._on_suite_error, str(exc))
                    return
        finally:
            sys.argv = saved
            try:
                out_stream.flush()
                err_stream.flush()
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
            self._current_total = int(payload.get("total") or 0)
            self._current_i = 0
            self._reset_progress(total=self._current_total)
            activity.write(
                f"[{BRAND}]{ts_s}[/{BRAND}] task start  {self._current_label}"
                f"  n={self._current_total}"
            )
            arm = self._current_arm or "vanilla_lm"
            self._upsert_row(arm, self._current_task, n=0, acc=None, secs=None, status="running")
        elif topic == "bench.example":
            self._current_i = int(payload.get("i") or 0)
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
            n = int(payload.get("n", 0))
            acc = float(payload.get("accuracy", 0.0))
            secs = float(payload.get("seconds", 0.0))
            self._upsert_row(arm, task, n=n, acc=acc, secs=secs, status="done")
            self._update_arm_totals(arm, n=n, correct=int(payload.get("correct", 0)))
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
            for i, val in enumerate(cells):
                try:
                    table.update_cell(rk, table.ordered_columns[i].key, val)
                except Exception:
                    pass
        else:
            self._row_keys[key] = table.add_row(*cells)

    def _render_status(self, status: str) -> str:
        color = {
            "running": BRAND,
            "done": ONLINE,
            "error": "red",
            "queued": OFFLINE,
        }.get(status, "white")
        return f"[{color}]{status}[/{color}]"

    def _refresh_deltas(self, task: str) -> None:
        v = self._results.get(("vanilla_lm", task))
        b = self._results.get(("broca_shell", task))
        if not (v and b and v.get("acc") is not None and b.get("acc") is not None):
            return
        delta = float(b["acc"]) - float(v["acc"])
        b["delta"] = delta
        # Update the broca_shell row's delta cell.
        table = self.query_one("#results", DataTable)
        rk = self._row_keys.get(("broca_shell", task))
        if rk is not None:
            try:
                table.update_cell(rk, table.ordered_columns[4].key, _fmt_delta(delta))
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
        for pid in ("panel-phase", "panel-tally", "panel-native", "panel-compare", "panel-arch", "panel-lmeval", "panel-summary"):
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
        comp_lines: list[str] = []
        if v_total:
            comp_lines.append(f"vanilla_lm   n={v_total[0]:5d}  acc={_fmt_pct(v_total[1] / max(1, v_total[0]))}")
        if b_total:
            comp_lines.append(f"broca_shell  n={b_total[0]:5d}  acc={_fmt_pct(b_total[1] / max(1, b_total[0]))}")
        if v_total and b_total:
            v_acc = v_total[1] / max(1, v_total[0])
            b_acc = b_total[1] / max(1, b_total[0])
            d = b_acc - v_acc
            color = ONLINE if d >= 0 else WARNING
            comp_lines.append(f"[{color}]Δ            {_fmt_delta(d)}[/{color}]")
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
                lm_lines.append(f"[red]error: {err[:48]}[/red]")
            else:
                lm_lines.append(f"out: [dim]{self._lm_eval_summary.get('out')}[/dim]")
                lm_lines.append("[dim]see lm_eval_pair.json for per-task[/dim]")
        else:
            lm_lines.append("[dim]not run[/dim]")
        self.query_one("#panel-lmeval", StatePanel).set_lines(lm_lines)
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
                f" press ctrl+c to quit  ·  events stream live from core.event_bus"
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
    p = argparse.ArgumentParser(
        description=(
            "Textual TUI wrapping `python -m core.benchmarks`. All flags are forwarded "
            "verbatim to the bench entry point."
        ),
        add_help=False,
    )
    # Defer help / argument parsing to the underlying benchmarks parser. We
    # accept any args and pass them through.
    p.add_argument("--tui-log-level", default=os.environ.get("TUI_LOG_LEVEL", "INFO"))
    p.add_argument("-h", "--help", action="store_true")
    return p


def main() -> None:
    parser = _build_parser()
    known, rest = parser.parse_known_args()
    if known.help:
        # Show the underlying bench parser's help as well.
        parser.print_help()
        print("\n--- forwarded to core.benchmarks ---\n")
        from core.benchmarks.__main__ import build_arg_parser as _bp
        _bp().print_help()
        return

    os.environ.setdefault("LOG_SILENT", "1")
    # The bench worker runs off the main thread; matplotlib's macOS backend
    # raises if created from a worker thread. Force Agg before any plotting
    # import so generate_artifacts() can write PNG/PDF files safely.
    os.environ.setdefault("MPLBACKEND", "Agg")
    configure_lab_logging()

    bus = get_default_bus()
    log_level = getattr(logging, known.tui_log_level.upper(), logging.INFO)
    handler = LogToBusHandler(bus, level=log_level)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger("core").addHandler(handler)

    app = BenchApp(bus=bus, bench_argv=list(rest))
    try:
        app.run()
    finally:
        try:
            logging.getLogger("core").removeHandler(handler)
        except Exception:
            pass


if __name__ == "__main__":
    main()
