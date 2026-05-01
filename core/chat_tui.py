"""Textual TUI for BrocaMind: chat in the middle, substrate state on the sides.

Run:

  python -m core.chat_tui --broca-db runs/broca_chat.sqlite

The center column streams the substrate-biased chat (same path as
``chat_cli.py`` ``--broca``). The left column shows comprehension state
(current cognitive frame, working memory, intrinsic cues, logit bias) and
the right column shows substrate dynamics (semantic memory, DMN background,
self-improve worker, Hawkes intensities, confidence sparkline). The bottom
bar carries model/device/db/namespace status and a tail of log records
forwarded through :class:`core.event_bus.LogToBusHandler`.

Architecture:

* Inference runs on a Textual ``@work(thread=True)`` worker so the UI
  remains responsive during a slow ``chat_reply``.
* Token deltas flow back to the UI via thread-safe ``call_from_thread``.
* A 5Hz timer polls :meth:`BrocaMind.snapshot` and drains the event bus
  to refresh side panels and the activity feed.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from collections import deque
from pathlib import Path
from typing import Any

try:
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, Vertical, VerticalScroll
    from textual.reactive import reactive
    from textual.widgets import Footer, Header, Input, Label, RichLog, Sparkline, Static
    from textual.worker import Worker, WorkerState
    from textual import work
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "core.chat_tui requires Textual. Install with:\n\n"
        "  uv sync --extra tui\n"
        "  # or: pip install -e \".[tui]\"\n"
    ) from exc

from .broca import BrocaMind
from .device_utils import pick_torch_device
from .event_bus import EventBus, LogToBusHandler, get_default_bus
from .llama_broca_host import quiet_transformers_benchmark_log_warnings, resolve_hf_hub_token
from .logging_setup import configure_lab_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Side-panel widgets
# ---------------------------------------------------------------------------


class StatePanel(Static):
    """A titled panel that renders a dict of key/value pairs."""

    DEFAULT_CSS = """
    StatePanel {
        border: round $primary 60%;
        padding: 0 1;
        height: auto;
        margin-bottom: 1;
    }
    StatePanel > .title {
        text-style: bold;
        color: $accent;
    }
    """

    def __init__(self, title: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._title = title
        self._lines: list[str] = []

    def render(self) -> str:
        head = f"[b]{self._title}[/b]"
        if not self._lines:
            return f"{head}\n[dim]—[/dim]"
        return head + "\n" + "\n".join(self._lines)

    def set_lines(self, lines: list[str]) -> None:
        self._lines = lines
        self.refresh()


def _fmt_float(v: Any, prec: int = 3) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.{prec}f}"
    except (TypeError, ValueError):
        return str(v)


def _fmt_intent(intent: str | None) -> str:
    return intent or "—"


# ---------------------------------------------------------------------------
# The app
# ---------------------------------------------------------------------------


class BrocaChatApp(App):
    CSS = """
    Screen {
        layout: vertical;
    }
    #main {
        height: 1fr;
    }
    #left, #right {
        width: 32;
        min-width: 28;
        padding: 1;
        border-right: solid $primary 30%;
    }
    #right {
        border-right: none;
        border-left: solid $primary 30%;
    }
    #center {
        width: 1fr;
        padding: 0 1;
    }
    #chatlog {
        height: 1fr;
        border: round $primary 60%;
    }
    #input {
        height: 3;
        border: round $accent 60%;
    }
    #streaming {
        height: auto;
        max-height: 8;
        padding: 0 1;
        color: $text-muted;
    }
    #status {
        height: 1;
        background: $primary 20%;
        color: $text;
        padding: 0 1;
    }
    #activity {
        height: 8;
        border: round $primary 30%;
    }
    Sparkline {
        height: 3;
        margin-bottom: 1;
    }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+d", "quit", "Quit"),
        ("ctrl+l", "clear_chat", "Clear chat"),
    ]

    busy: reactive[bool] = reactive(False)

    def __init__(
        self,
        *,
        mind: BrocaMind,
        bus: EventBus,
        max_new_tokens: int = 512,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9,
        debug_substrate: bool = False,
    ) -> None:
        super().__init__()
        self.mind = mind
        self.bus = bus
        self.max_new_tokens = int(max_new_tokens)
        self.do_sample = bool(do_sample)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.debug_substrate = bool(debug_substrate)

        self._messages: list[dict[str, str]] = []
        self._reply_buffer: list[str] = []
        self._sub_id = self.bus.subscribe("*")
        self._confidence_trend: deque[float] = deque(maxlen=64)
        self._dmn_duration_trend: deque[float] = deque(maxlen=32)
        self._memory_count_trend: deque[float] = deque(maxlen=64)

    # ------------------------------------------------------------------ layout
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main"):
            with VerticalScroll(id="left"):
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

    # ------------------------------------------------------------------ lifecycle
    def on_mount(self) -> None:
        self.title = "Broca substrate chat"
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

    # ------------------------------------------------------------------ tick
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
            if topic == "frame.comprehend":
                activity.write(
                    f"[cyan]{ts}[/cyan] frame [b]{_fmt_intent(payload.get('intent'))}[/b]"
                    f"  conf={_fmt_float(payload.get('confidence'))}"
                    f"  subject={payload.get('subject') or '—'}"
                )
                conf = payload.get("confidence")
                if conf is not None:
                    self._confidence_trend.append(float(conf))
            elif topic == "intrinsic_cue":
                activity.write(
                    f"[yellow]{ts}[/yellow] cue [b]{payload.get('faculty')}[/b]"
                    f"  urgency={_fmt_float(payload.get('urgency'))}"
                )
            elif topic == "consolidation":
                activity.write(
                    f"[green]{ts}[/green] consolidation reflections={payload.get('reflections', 0)}"
                )
            elif topic == "dmn.tick":
                duration = float(payload.get("duration_ms", 0))
                self._dmn_duration_trend.append(duration)
                activity.write(
                    f"[magenta]{ts}[/magenta] dmn tick {payload.get('iteration')}"
                    f"  {duration:.0f}ms  reflections={payload.get('reflections', 0)}"
                )
            elif topic == "self_improve.cycle_start":
                activity.write(
                    f"[blue]{ts}[/blue] self-improve start run={payload.get('run_id', '')[:8]}"
                )
            elif topic == "self_improve.cycle_complete":
                err = payload.get("error")
                if err:
                    activity.write(
                        f"[red]{ts}[/red] self-improve fail run={payload.get('run_id', '')[:8]}  {err[:80]}"
                    )
                else:
                    activity.write(
                        f"[blue]{ts}[/blue] self-improve done run={payload.get('run_id', '')[:8]}  {payload.get('summary') or ''}"
                    )
            elif topic.startswith("log."):
                level = payload.get("level", "INFO")
                msg = payload.get("msg", "")
                color = {"DEBUG": "dim", "INFO": "white", "WARNING": "yellow", "ERROR": "red", "CRITICAL": "red"}.get(level, "white")
                activity.write(f"[{color}]{ts} {level:7} {payload.get('name', '')}[/{color}]  {msg}")
            else:
                activity.write(f"[dim]{ts} {topic}[/dim]  {payload}")

    def _refresh_panels(self) -> None:
        try:
            snap = self.mind.snapshot()
        except Exception:
            logger.exception("TUI: snapshot failed")
            return

        # Track memory count for an unused-but-handy trend.
        mem = snap.get("memory") or {}
        if isinstance(mem.get("count"), int):
            self._memory_count_trend.append(float(mem["count"]))

        # --- Frame
        frame_lines: list[str] = []
        last = snap.get("last_chat") or {}
        if last:
            frame_lines.append(f"intent: [b]{_fmt_intent(last.get('intent'))}[/b]")
            frame_lines.append(f"subject: {last.get('subject') or '—'}")
            frame_lines.append(f"answer: {last.get('answer') or '—'}")
            frame_lines.append(f"confidence: {_fmt_float(last.get('confidence'))}")
            frame_lines.append(f"eff temp: {_fmt_float(last.get('eff_temperature'))}")
            frame_lines.append(f"bias tokens: {last.get('bias_token_count', 0)}")
            frame_lines.append(f"broca features: {'on' if last.get('has_broca_features') else 'off'}")
        else:
            ws = snap.get("workspace") or {}
            lf = ws.get("latest_frame")
            if lf:
                frame_lines.append(f"intent: [b]{_fmt_intent(lf.get('intent'))}[/b]")
                frame_lines.append(f"subject: {lf.get('subject') or '—'}")
                frame_lines.append(f"answer: {lf.get('answer') or '—'}")
                frame_lines.append(f"confidence: {_fmt_float(lf.get('confidence'))}")
        self.query_one("#panel-frame", StatePanel).set_lines(frame_lines)

        # --- Working memory (recent journal)
        journal = snap.get("journal") or {}
        recent = journal.get("recent") or []
        wm_lines = [f"journal rows: {journal.get('count', 0)}"]
        for r in recent[-5:][::-1]:
            wm_lines.append(
                f"#{r.get('id')} [b]{_fmt_intent(r.get('intent'))}[/b] "
                f"{_fmt_float(r.get('confidence'), 2)}  {(r.get('utterance') or '')[:32]}"
            )
        self.query_one("#panel-working", StatePanel).set_lines(wm_lines)

        # --- Intrinsic cues
        ws = snap.get("workspace") or {}
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

        # --- Logit bias top
        bias_lines: list[str] = []
        for b in (last.get("bias_top") or [])[:8]:
            tok = (b.get("token") or "").replace("\n", "\\n")
            bias_lines.append(f"{tok!r}  {_fmt_float(b.get('bias'), 2)}")
        if not bias_lines:
            bias_lines.append("[dim]—[/dim]")
        self.query_one("#panel-bias", StatePanel).set_lines(bias_lines)

        # --- Memory
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

        # --- DMN
        bg = snap.get("background") or {}
        dmn_lines = [
            f"running: {'[green]yes[/green]' if bg.get('running') else '[red]no[/red]'}",
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

        # --- Self-improve
        si = snap.get("self_improve") or {}
        si_lines = [
            f"enabled: {'[green]yes[/green]' if si.get('enabled') else '[dim]no[/dim]'}",
            f"running: {'[green]yes[/green]' if si.get('running') else '[red]no[/red]'}",
            f"iterations: {si.get('iterations', 0)}",
        ]
        if si.get("last_summary"):
            si_lines.append(f"last: {str(si['last_summary'])[:40]}")
        if si.get("last_error"):
            si_lines.append(f"[red]err: {str(si['last_error'])[:40]}[/red]")
        self.query_one("#panel-self-improve", StatePanel).set_lines(si_lines)

        # --- Substrate
        sub = snap.get("substrate") or {}
        sub_lines = [
            f"vsa atoms: {sub.get('vsa_atoms', 0)}",
            f"hopfield: {sub.get('hopfield_stored', 0)}/{sub.get('hopfield_max_items', 0)}",
            f"hawkes channels: {sub.get('hawkes_channels', 0)}",
            f"tools: {sub.get('tools', 0)}",
            f"macros: {sub.get('macros', 0)}",
            f"ontology axes: {sub.get('ontology_axes', 0)}",
            f"discovered SCM: {'[green]yes[/green]' if sub.get('discovered_scm') else '[dim]no[/dim]'}",
        ]
        self.query_one("#panel-substrate", StatePanel).set_lines(sub_lines)

        # --- Hawkes intensity
        hk = (snap.get("substrate") or {}).get("hawkes_intensity") or {}
        hk_lines: list[str] = []
        for name, val in sorted(hk.items(), key=lambda kv: -float(kv[1] or 0.0))[:8]:
            hk_lines.append(f"{name[:20]:<20}  {_fmt_float(val, 3)}")
        if not hk_lines:
            hk_lines.append("[dim]—[/dim]")
        self.query_one("#panel-hawkes", StatePanel).set_lines(hk_lines)

        # --- Sparklines
        if self._confidence_trend:
            self.query_one("#spark-confidence", Sparkline).data = list(self._confidence_trend)
        if self._dmn_duration_trend:
            self.query_one("#spark-dmn", Sparkline).data = list(self._dmn_duration_trend)

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
        self._run_chat(text)

    @work(thread=True, exclusive=True)
    def _run_chat(self, _user_text: str) -> None:
        def on_token(piece: str) -> None:
            self.app.call_from_thread(self._on_token, piece)

        try:
            frame, reply = self.mind.chat_reply(
                self._messages,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature,
                top_p=self.top_p,
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

    def _on_reply_done(self, intent: str, confidence: float, reply: str) -> None:
        self.busy = False
        text = "".join(self._reply_buffer) or reply
        text = text.strip() or "[empty reply]"
        self._messages.append({"role": "assistant", "content": text})
        chatlog = self.query_one("#chatlog", RichLog)
        chatlog.write(f"[bold magenta]Assistant[/bold magenta]  {text}")
        if self.debug_substrate:
            chatlog.write(f"[dim]substrate: intent={intent} confidence={confidence:.2f}[/dim]")
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
        # Keep the busy flag honest if a worker dies unexpectedly.
        if event.state in (WorkerState.SUCCESS, WorkerState.ERROR, WorkerState.CANCELLED):
            self.busy = False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Textual TUI for the Broca substrate chat.")
    p.add_argument(
        "--model",
        default=os.environ.get("MODEL_ID", "meta-llama/Llama-3.2-1B-Instruct"),
        help="HF model id (default: MODEL_ID env or Llama-3.2-1B-Instruct).",
    )
    p.add_argument("--device", default=os.environ.get("M_DEVICE"), help="Torch device override (cpu, mps, cuda:0).")
    p.add_argument(
        "--token",
        default=None,
        help="HF hub token string, or omit to use HF_TOKEN / huggingface-cli login.",
    )
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--sample", action="store_true", help="Use sampling instead of greedy decoding.")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument(
        "--broca-db",
        type=Path,
        default=None,
        help="SQLite path for BrocaMind. Default: runs/broca_chat.sqlite",
    )
    p.add_argument("--broca-namespace", default="chat", help="Semantic memory namespace.")
    p.add_argument("--no-background", action="store_true", help="Disable DMN background consolidation.")
    p.add_argument("--background-interval", type=float, default=5.0)
    p.add_argument("--self-improve", action="store_true", help="Enable Docker-backed self-improve worker.")
    p.add_argument("--no-self-improve", action="store_true")
    p.add_argument("--self-improve-interval", type=float, default=None)
    p.add_argument("--debug-substrate", action="store_true")
    p.add_argument(
        "--log-level",
        default=os.environ.get("TUI_LOG_LEVEL", "INFO"),
        help="Level forwarded into the activity feed (DEBUG, INFO, WARNING, ERROR). Default: INFO.",
    )
    return p


def main() -> None:
    # Silence the stderr stream handler so logging doesn't fight the TUI; the
    # rotating file handler still records the full event stream.
    os.environ.setdefault("LOG_SILENT", "1")
    configure_lab_logging()
    args = _build_parser().parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    quiet_transformers_benchmark_log_warnings()

    bus = get_default_bus()
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    handler = LogToBusHandler(bus, level=log_level)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger("core").addHandler(handler)

    resolved_device = pick_torch_device(args.device)
    if args.token is None:
        token_kw: str | bool | None = resolve_hf_hub_token(None)
    elif args.token.strip() == "":
        token_kw = True
    else:
        token_kw = args.token.strip()

    db_path = args.broca_db or Path("runs/broca_chat.sqlite")
    db_path.parent.mkdir(parents=True, exist_ok=True)

    mind = BrocaMind(
        seed=0,
        db_path=db_path,
        namespace=args.broca_namespace,
        llama_model_id=args.model,
        device=resolved_device,
        hf_token=token_kw,
    )
    mind.event_bus = bus  # ensure shared bus

    if not args.no_background:
        mind.start_background(interval_s=max(0.1, float(args.background_interval)))

    si_env = os.environ.get("BROCA_SELF_IMPROVE", "").strip().lower() in {"1", "true", "yes", "on"}
    enable_self_improve = (args.self_improve or si_env) and not args.no_self_improve
    if enable_self_improve:
        mind.start_self_improve_worker(interval_s=args.self_improve_interval, enabled=True)

    app = BrocaChatApp(
        mind=mind,
        bus=bus,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.sample,
        temperature=args.temperature,
        top_p=args.top_p,
        debug_substrate=args.debug_substrate,
    )
    try:
        app.run()
    finally:
        mind.stop_background()
        mind.stop_self_improve_worker()
        try:
            logging.getLogger("core").removeHandler(handler)
        except Exception:
            pass


if __name__ == "__main__":
    main()
