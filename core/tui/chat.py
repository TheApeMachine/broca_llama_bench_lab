"""Chat TUI for the Mosaic system.

Run:

  make tui

Architecture:

* The chat interface is a Textual ``@work(thread=True)`` worker so the UI
  remains responsive during a slow ``chat_reply``.
* Token deltas flow back to the UI via thread-safe ``call_from_thread``.
* A 5Hz timer polls the control plane and drains the event bus
  to refresh side panels and the activity feed.
"""

from __future__ import annotations

import argparse
import logging
import time
from collections import deque
from typing import Any
from .state import StatePanel
from .systems import SystemsMatrix
from .components import (
    _activity_line_consolidation,
    _activity_line_dmn_tick,
    _activity_line_frame_comprehend,
    _activity_line_intrinsic_cue,
    _activity_line_log,
    _activity_line_self_improve_complete,
    _activity_line_self_improve_start,
)
from . import styles as tui_styles

try:
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, Vertical, VerticalScroll
    from textual.reactive import reactive
    from textual.widgets import Footer, Header, Input, Label, RichLog, Sparkline, Static
    from textual.worker import Worker, WorkerState
    from textual import work
except ImportError as exc:
    raise SystemExit(
        "core.tui.chat requires Textual. Install with:\n\n"
        "  uv sync --extra tui\n"
        "  # or: pip install -e \".[tui]\"\n"
    ) from exc

from core.cognition.substrate import SubstrateController
from core.system.event_bus import EventBus
from core.substrate.runtime import (
    CHAT_DO_SAMPLE,
    CHAT_MAX_NEW_TOKENS,
    CHAT_NAMESPACE,
    CHAT_TEMPERATURE,
    CHAT_TOP_P,
)

logger = logging.getLogger(__name__)

class Chat(App):
    CSS = tui_styles.CSS

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+d", "quit", "Quit"),
        ("ctrl+l", "clear_chat", "Clear chat"),
    ]

    busy: reactive[bool] = reactive(False)

    def __init__(self, *, mind: SubstrateController, bus: EventBus) -> None:
        super().__init__()

        self.mind = mind
        self.bus = bus

        self._messages: list[dict[str, str]] = []
        self._reply_buffer: list[str] = []
        self._sub_id = self.bus.subscribe("*")
        self._confidence_trend: deque[float] = deque(maxlen=64)
        self._dmn_duration_trend: deque[float] = deque(maxlen=32)
        self._memory_count_trend: deque[float] = deque(maxlen=64)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Horizontal(id="main"):
            with VerticalScroll(id="left"):
                yield SystemsMatrix(id="panel-systems")
                yield StatePanel("Cognitive frame", id="panel-frame")
                yield StatePanel("Working memory", id="panel-working")
                yield StatePanel("Intrinsic cues", id="panel-cues")
                yield StatePanel("Logit bias (top)", id="panel-bias")

            with Vertical(id="center"):
                yield RichLog(id="chatlog", wrap=True, markup=True, highlight=False)
                yield Static("", id="streaming")
                yield Input(placeholder="Speak to the substrate.  /quit to exit.", id="input")
                yield RichLog(id="activity", wrap=False, markup=True, highlight=False)

            with VerticalScroll(id="right"):
                yield StatePanel("Semantic memory", id="panel-memory")
                yield Sparkline([0.0], id="spark-confidence")
                yield Label("Confidence (recent)", classes="dim")
                yield StatePanel("DMN background", id="panel-dmn")
                yield Sparkline([0.0], id="spark-dmn")
                yield Label("DMN tick duration (ms)", classes="dim")
                yield StatePanel("Self-improve worker", id="panel-self-improve")
                yield StatePanel("Substrate", id="panel-substrate")
                yield StatePanel("Hawkes intensity", id="panel-hawkes")

        yield Static("", id="status")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "Mosaic substrate chat"
        self.sub_title = self.mind.llama_model_id

        self.set_interval(0.2, self._tick)

        chatlog = self.query_one("#chatlog", RichLog)
        chatlog.write("[dim]Substrate biases the LLM via grafts; the LLM still chooses surface form.[/dim]")
        chatlog.write(
            f"[dim]db={self.mind.db_path}  namespace={self.mind.namespace}[/dim]\n"
        )

        self._refresh_status()
        self._refresh_panels()

    def on_unmount(self) -> None:
        try:
            self.bus.unsubscribe(self._sub_id)
        except Exception:
            pass

    def _tick(self) -> None:
        self._drain_events()
        self._refresh_panels()

    def _drain_events(self) -> None:
        events = self.bus.drain(self._sub_id)

        if not events:
            return

        activity = self.query_one("#activity", RichLog)

        for ev in events:
            topic = ev.topic
            payload = ev.payload or {}
            ts = time.strftime("%H:%M:%S", time.localtime(ev.ts))

            try:
                if topic == "frame.comprehend":
                    activity.write(_activity_line_frame_comprehend(ts, payload))

                    conf = payload.get("confidence")

                    if conf is not None:
                        self._confidence_trend.append(float(conf))

                elif topic == "intrinsic_cue":
                    activity.write(_activity_line_intrinsic_cue(ts, payload))

                elif topic == "consolidation":
                    activity.write(_activity_line_consolidation(ts, payload))

                elif topic == "dmn.tick":
                    duration_ms = float(payload.get("duration_ms", 0))
                    self._dmn_duration_trend.append(duration_ms)

                    activity.write(_activity_line_dmn_tick(ts, payload, duration_ms))

                elif topic == "self_improve.cycle_start":
                    activity.write(_activity_line_self_improve_start(ts, payload))

                elif topic == "self_improve.cycle_complete":
                    activity.write(_activity_line_self_improve_complete(ts, payload))

                elif topic.startswith("log."):
                    activity.write(_activity_line_log(ts, payload))

                else:
                    activity.write(f"[dim]{ts} {topic}[/dim]  {payload}")
            except Exception as exc:
                logger.exception(
                    "TUI chat: failed handling bus event topic=%r ts=%s payload=%r",
                    topic,
                    ev.ts,
                    payload,
                )
                activity.write(
                    f"[red]{ts}[/red] bad event topic={topic!r} payload={payload!r} err={exc!r}"
                )

    def _sync_sparkline(self, css_id: str, trend: deque[float]) -> None:
        if not trend:
            return

        self.query_one(css_id, Sparkline).data = list(trend)

    # -- systems matrix ------------------------------------------------

    def _refresh_systems_matrix(self, snap: dict[str, Any]) -> None:
        """Map the SubstrateController snapshot into an at-a-glance online/offline grid."""

        sub = snap.get("substrate") or {}
        bg = snap.get("background") or {}
        si = snap.get("self_improve") or {}
        ws = snap.get("workspace") or {}
        memory = snap.get("memory") or {}
        model = snap.get("model") or {}

        def status_count(value: int | None, label: str) -> tuple[str, str, str]:
            n = int(value or 0)

            return ("on" if n > 0 else "off", label, f"{n}")

        entries: list[tuple[str, str, str]] = []

        host_status = "on" if model.get("id") else "off"
        entries.append((host_status, "llama host", str(model.get("device") or "—")))

        mem_n = int(memory.get("count") or 0)
        entries.append(("on" if mem_n > 0 else "off", "semantic mem", f"{mem_n} rows"))

        wn = int(ws.get("frames_total") or 0)
        entries.append(("on" if wn > 0 else "off", "workspace", f"{wn} frames"))

        if bg.get("error") is not None and not bg.get("running"):
            entries.append(("warn", "dmn worker", "err"))
        elif bg.get("running"):
            entries.append(("on", "dmn worker", f"{int(bg.get('iterations') or 0)} iters"))
        else:
            entries.append(("off", "dmn worker", "idle"))

        if not si.get("enabled"):
            entries.append(("off", "self-improve", "disabled"))
        elif si.get("running"):
            entries.append(("on", "self-improve", f"{int(si.get('iterations') or 0)} iters"))
        else:
            entries.append(("warn", "self-improve", "stopped"))

        entries.append(status_count(sub.get("vsa_atoms"), "vsa codebook"))

        hop_stored = int(sub.get("hopfield_stored") or 0)
        hop_max = int(sub.get("hopfield_max_items") or 0)
        entries.append(("on" if hop_stored > 0 else "off", "hopfield", f"{hop_stored}/{hop_max}"))

        entries.append(status_count(sub.get("hawkes_channels"), "hawkes proc"))
        entries.append(status_count(sub.get("tools"), "tools"))
        entries.append(status_count(sub.get("macros"), "macros"))
        entries.append(status_count(sub.get("ontology_axes"), "ontology"))

        entries.append((
            "on" if sub.get("discovered_scm") else "off",
            "scm discov.",
            "ready" if sub.get("discovered_scm") else "—",
        ))

        entries.append(("on", "event bus", "5Hz"))

        try:
            self.query_one("#panel-systems", SystemsMatrix).set_entries(entries)
        except Exception:
            logger.exception("TUI: refresh systems matrix failed")

    def _refresh_panels(self) -> None:
        try:
            snap = self.mind.snapshot()
        except Exception:
            logger.exception("TUI: snapshot failed")

            return

        mem = snap.get("memory") or {}
        ws = snap.get("workspace") or {}
        sub = snap.get("substrate") or {}
        bg = snap.get("background") or {}
        si = snap.get("self_improve") or {}

        if isinstance(mem.get("count"), int):
            self._memory_count_trend.append(float(mem["count"]))

        self._refresh_systems_matrix(snap)

        last = snap.get("last_chat") or {}
        lf = ws.get("latest_frame") or {}

        if last:
            frame_lines = _frame_summary_lines(last)
            frame_lines.append(f"eff temp: {_fmt_float(last.get('eff_temperature'))}")
            frame_lines.append(f"bias tokens: {last.get('bias_token_count', 0)}")
            frame_lines.append(f"broca features: {'on' if last.get('has_broca_features') else 'off'}")
        elif lf:
            frame_lines = _frame_summary_lines(lf)
        else:
            frame_lines = []

        self.query_one("#panel-frame", StatePanel).set_lines(frame_lines)

        journal = snap.get("journal") or {}
        recent = journal.get("recent") or []
        wm_lines = [f"journal rows: {journal.get('count', 0)}"]

        for r in recent[-5:][::-1]:
            wm_lines.append(
                f"#{r.get('id')} [b]{_fmt_intent(r.get('intent'))}[/b] "
                f"{_fmt_float(r.get('confidence'), 2)}  {(r.get('utterance') or '')[:32]}"
            )

        self.query_one("#panel-working", StatePanel).set_lines(wm_lines)

        cues = ws.get("intrinsic_cues") or []
        cue_lines = [f"workspace frames: {ws.get('frames_total', 0)}"]

        if not cues:
            cue_lines.append("[dim]no active cues[/dim]")
        else:
            for c in cues[-6:]:
                cue_lines.append(
                    f"[yellow]{c.get('faculty')}[/yellow] u={_fmt_float(c.get('urgency'), 2)}"
                )

        self.query_one("#panel-cues", StatePanel).set_lines(cue_lines)

        bias_lines: list[str] = []

        for b in (last.get("bias_top") or [])[:8]:
            tok = (b.get("token") or "").replace("\n", "\\n")
            bias_lines.append(f"{tok!r}  {_fmt_float(b.get('bias'), 2)}")

        if not bias_lines:
            bias_lines.append(tui_styles._DIM_EM_DASH)

        self.query_one("#panel-bias", StatePanel).set_lines(bias_lines)

        mem_lines = [
            f"records: {mem.get('count', 0)}",
            f"subjects: {mem.get('subjects', 0)}",
            f"mean conf: {_fmt_float(mem.get('mean_confidence'))}",
        ]

        for c in (mem.get("recent_claims") or [])[-3:]:
            mem_lines.append(
                f"[dim]{c.get('subject')} {c.get('predicate')} {c.get('object')}[/dim]"
                f"  {_fmt_float(c.get('confidence'), 2)}"
            )

        self.query_one("#panel-memory", StatePanel).set_lines(mem_lines)

        dmn_lines = [
            f"running: {_rich_yes_no_strong(bool(bg.get('running')))}",
            f"iterations: {bg.get('iterations', 0)}",
            f"interval: {_fmt_float(bg.get('interval_s'), 1)}s",
            f"idle: {_fmt_float(bg.get('idle_seconds'), 1)}s",
        ]

        last_phase = bg.get("last_phase_summary") or {}

        for name, summary in list(last_phase.items())[:6]:
            if not isinstance(summary, dict):
                continue

            dur = summary.get("duration_ms")
            refl = summary.get("reflections")
            dmn_lines.append(f"[dim]{name}[/dim]  {dur or '—'}ms  refl={refl or 0}")

        if bg.get("last_error"):
            dmn_lines.append(f"[red]err: {str(bg['last_error'])[:40]}[/red]")

        self.query_one("#panel-dmn", StatePanel).set_lines(dmn_lines)

        si_lines = [
            f"enabled: {_rich_yes_no_soft(bool(si.get('enabled')))}",
            f"running: {_rich_yes_no_strong(bool(si.get('running')))}",
            f"iterations: {si.get('iterations', 0)}",
        ]

        if si.get("last_summary"):
            si_lines.append(f"last: {str(si['last_summary'])[:40]}")

        if si.get("last_error"):
            si_lines.append(f"[red]err: {str(si['last_error'])[:40]}[/red]")

        self.query_one("#panel-self-improve", StatePanel).set_lines(si_lines)

        sub_lines = [
            f"vsa atoms: {sub.get('vsa_atoms', 0)}",
            f"hopfield: {sub.get('hopfield_stored', 0)}/{sub.get('hopfield_max_items', 0)}",
            f"hawkes channels: {sub.get('hawkes_channels', 0)}",
            f"tools: {sub.get('tools', 0)}",
            f"macros: {sub.get('macros', 0)}",
            f"ontology axes: {sub.get('ontology_axes', 0)}",
            f"discovered SCM: {_rich_yes_no_soft(bool(sub.get('discovered_scm')))}",
        ]

        self.query_one("#panel-substrate", StatePanel).set_lines(sub_lines)

        hk = sub.get("hawkes_intensity") or {}
        hk_lines: list[str] = []

        for name, val in sorted(hk.items(), key=lambda kv: -float(kv[1] or 0.0))[:8]:
            hk_lines.append(f"{name[:20]:<20}  {_fmt_float(val, 3)}")

        if not hk_lines:
            hk_lines.append(tui_styles._DIM_EM_DASH)

        self.query_one("#panel-hawkes", StatePanel).set_lines(hk_lines)

        self._sync_sparkline("#spark-confidence", self._confidence_trend)
        self._sync_sparkline("#spark-dmn", self._dmn_duration_trend)

    def _refresh_status(self) -> None:
        try:
            device = next(self.mind.host.parameters()).device
        except Exception:
            device = "?"

        status = (
            f" model={self.mind.llama_model_id}   device={device}   "
            f"db={self.mind.db_path}   namespace={self.mind.namespace}"
        )

        self.query_one("#status", Static).update(status)

    # ------------------------------------------------------------------ chat

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()

        if not text:
            return

        if text.lower() in {"/quit", "/exit", ":q"}:
            self.exit()

            return

        if self.busy:
            return

        event.input.value = ""

        chatlog = self.query_one("#chatlog", RichLog)
        chatlog.write(f"[bold cyan]You[/bold cyan]  {text}")

        self._messages.append({"role": "user", "content": text})
        self._reply_buffer.clear()

        self.query_one("#streaming", Static).update("[bold magenta]Assistant[/bold magenta]  …")
        self.busy = True

        self._run_chat()

    @work(thread=True, exclusive=True)
    def _run_chat(self) -> None:
        def on_token(piece: str) -> None:
            self.app.call_from_thread(self._on_token, piece)

        try:
            frame, reply = self.mind.chat_reply(
                self._messages,
                max_new_tokens=CHAT_MAX_NEW_TOKENS,
                do_sample=CHAT_DO_SAMPLE,
                temperature=CHAT_TEMPERATURE,
                top_p=CHAT_TOP_P,
                on_token=on_token,
            )

            self.app.call_from_thread(self._on_reply_done, frame.intent, frame.confidence, reply)
        except Exception as exc:
            logger.exception("TUI chat_reply failed")

            self.app.call_from_thread(self._on_reply_error, str(exc))

    def _on_token(self, piece: str) -> None:
        self._reply_buffer.append(piece)

        running = "".join(self._reply_buffer)

        self.query_one("#streaming", Static).update(f"[bold magenta]Assistant[/bold magenta]  {running}")

    def _on_reply_done(self, _intent: str, _confidence: float, reply: str) -> None:
        self.busy = False

        text = "".join(self._reply_buffer) or reply
        text = text.strip() or "[empty reply]"

        self._messages.append({"role": "assistant", "content": text})

        chatlog = self.query_one("#chatlog", RichLog)
        chatlog.write(f"[bold magenta]Assistant[/bold magenta]  {text}")

        self.query_one("#streaming", Static).update("")
        self._reply_buffer.clear()

    def _on_reply_error(self, err: str) -> None:
        self.busy = False

        chatlog = self.query_one("#chatlog", RichLog)
        chatlog.write(f"[red]error: {err}[/red]")

        self.query_one("#streaming", Static).update("")

        if self._messages and self._messages[-1].get("role") == "user":
            self._messages.pop()

    # ------------------------------------------------------------------ actions

    def action_clear_chat(self) -> None:
        self._messages.clear()

        chatlog = self.query_one("#chatlog", RichLog)
        chatlog.clear()
        chatlog.write("[dim]chat cleared[/dim]")

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:  # noqa: ARG002
        if event.state in (WorkerState.SUCCESS, WorkerState.ERROR, WorkerState.CANCELLED):
            self.busy = False


def _build_chat_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="Mosaic chat TUI (fixed runtime).")


def run_chat_tui(argv: list[str] | None = None) -> None:
    from core.cli import (
        attach_core_logs_to_bus,
        build_substrate_controller,
        configure_lab_session,
        default_bus,
        detach_core_log_handler,
        start_background_stack,
        stop_background_stack,
    )

    if argv is None:
        argv = []

    _build_chat_parser().parse_args(argv)

    configure_lab_session(silent_stderr_default=True)

    bus = default_bus()
    handler = attach_core_logs_to_bus(bus)
    mind: SubstrateController | None = None

    try:
        mind = build_substrate_controller(bus=bus)
        start_background_stack(mind)

        app = Chat(mind=mind, bus=bus)
        app.run()
    finally:
        if mind is not None:
            stop_background_stack(mind)

        detach_core_log_handler(handler)


def main() -> None:
    run_chat_tui()


if __name__ == "__main__":
    main()
