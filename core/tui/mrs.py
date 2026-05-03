"""MRS debug TUI — detailed real-time view of the recursive substrate.

This view inverts the priorities of :mod:`core.tui.chat`:

* Chat is shrunk to a compact strip at the bottom (input + last reply).
* The top three quarters of the screen are MRS instrumentation:
  SWM slot table, recursion trace, prediction-error bars, alignment
  registry, latent decoder configuration, active-thought cosine
  matches, and a high-detail MRS event stream.

Run:

  make mrs
"""

from __future__ import annotations

import argparse
import logging
import time
from collections import deque
from typing import Any

from .components import _fmt_float, _fmt_intent
from .mrs_widgets import (
    ActiveThoughtPanel,
    AlignmentPanel,
    LatentDecoderPanel,
    MRSActivityPanel,
    PredictionErrorPanel,
    RecursionPanel,
    SWMSlotsPanel,
)
from . import styles as tui_styles

try:
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, Vertical, VerticalScroll
    from textual.reactive import reactive
    from textual.widgets import Footer, Header, Input, RichLog, Sparkline, Static
    from textual.worker import Worker, WorkerState
    from textual import work
except ImportError as exc:
    raise SystemExit(
        "core.tui.mrs requires Textual. Install with:\n\n"
        "  uv sync --extra tui\n"
    ) from exc

from ..cognition.substrate import SubstrateController
from ..workspace import BaseWorkspace as EventBus
from ..substrate.runtime import (
    CHAT_DO_SAMPLE,
    CHAT_MAX_NEW_TOKENS,
    CHAT_TEMPERATURE,
    CHAT_TOP_P,
)
from ..infra.constants import BRAND, BRAND_BG, BRAND_DEEP, BRAND_SOFT


logger = logging.getLogger(__name__)


_MRS_CSS = f"""
Screen {{
    layout: vertical;
    background: {BRAND_BG};
}}
Header {{
    background: {BRAND_DEEP};
    color: $text;
}}
Footer {{
    background: {BRAND_DEEP};
}}
#mrs-top {{
    height: 3fr;
}}
#mrs-bottom {{
    height: 1fr;
    border-top: solid {BRAND} 50%;
    padding: 0 1;
}}
#mrs-col-left, #mrs-col-mid, #mrs-col-right {{
    width: 1fr;
    padding: 0 1;
}}
#mrs-col-left {{
    border-right: solid {BRAND} 30%;
}}
#mrs-col-right {{
    border-left: solid {BRAND} 30%;
}}
#mrs-chatlog {{
    height: 1fr;
    border: round {BRAND} 60%;
    margin-bottom: 0;
}}
#mrs-input {{
    height: 3;
    border: round {BRAND_SOFT} 80%;
}}
#mrs-streaming {{
    height: auto;
    max-height: 4;
    padding: 0 1;
    color: $text-muted;
}}
#mrs-status {{
    height: 1;
    background: {BRAND} 25%;
    color: $text;
    padding: 0 1;
}}
#mrs-joint-spark {{
    height: 3;
    margin-bottom: 1;
}}
"""


class MRSDebug(App):
    """MRS-first TUI: substrate visualisation up top, compact chat at bottom."""

    CSS = _MRS_CSS

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
        self._joint_efe_trend: deque[float] = deque(maxlen=64)

    # -- layout --------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Vertical(id="mrs-top"):
            with Horizontal():
                with VerticalScroll(id="mrs-col-left"):
                    yield SWMSlotsPanel("Substrate working memory", id="mrs-swm-slots")
                    yield AlignmentPanel(id="mrs-alignments")
                with VerticalScroll(id="mrs-col-mid"):
                    yield RecursionPanel(id="mrs-recursion")
                    yield LatentDecoderPanel(id="mrs-latent-decoder")
                    yield ActiveThoughtPanel(id="mrs-active-thought")
                with VerticalScroll(id="mrs-col-right"):
                    yield PredictionErrorPanel(id="mrs-prediction-error")
                    yield Sparkline([0.0], id="mrs-joint-spark")
                    yield MRSActivityPanel(id="mrs-activity")

        with Vertical(id="mrs-bottom"):
            yield RichLog(id="mrs-chatlog", wrap=True, markup=True, highlight=False)
            yield Static("", id="mrs-streaming")
            yield Input(placeholder="MRS chat (compact). /quit to exit.", id="mrs-input")

        yield Static("", id="mrs-status")
        yield Footer()

    # -- lifecycle -----------------------------------------------------------

    def on_mount(self) -> None:
        self.title = "Mosaic Recursive Substrate — debug view"
        self.sub_title = self.mind.llama_model_id

        self.set_interval(0.2, self._tick)

        chatlog = self.query_one("#mrs-chatlog", RichLog)
        chatlog.write(
            f"[dim]Substrate visualisation above; chat below.  "
            f"Each turn flows: comprehend → recursion → decode.[/dim]"
        )
        chatlog.write(
            f"[dim]m_latent={self.mind.latent_decoder.m_latent_steps}  "
            f"r_max={self.mind.recursion_halt.max_rounds}  "
            f"D_swm={self.mind.swm.dim}[/dim]"
        )

        # Static configuration for the latent decoder panel never changes.
        self.query_one("#mrs-latent-decoder", LatentDecoderPanel).update_static(
            decoder=self.mind.latent_decoder,
            halt=self.mind.recursion_halt,
        )
        self._refresh_status()
        self._refresh_panels()

    def on_unmount(self) -> None:
        try:
            self.bus.unsubscribe(self._sub_id)
        except Exception:
            pass

    # -- per-tick refresh ----------------------------------------------------

    def _tick(self) -> None:
        self._drain_events()
        self._refresh_panels()

    def _drain_events(self) -> None:
        events = self.bus.drain(self._sub_id)
        if not events:
            return

        activity = self.query_one("#mrs-activity", MRSActivityPanel)
        recursion = self.query_one("#mrs-recursion", RecursionPanel)
        latent = self.query_one("#mrs-latent-decoder", LatentDecoderPanel)

        for ev in events:
            topic = ev.topic
            payload = ev.payload or {}
            ts = time.strftime("%H:%M:%S", time.localtime(ev.ts))

            try:
                self._route_event(topic, ts, payload, activity, recursion, latent)
            except Exception:
                logger.exception(
                    "MRS TUI: failed to handle event topic=%r payload=%r",
                    topic,
                    payload,
                )

    def _route_event(
        self,
        topic: str,
        ts: str,
        payload: dict[str, Any],
        activity: MRSActivityPanel,
        recursion: RecursionPanel,
        latent: LatentDecoderPanel,
    ) -> None:
        if topic == "swm.write":
            activity.append_event(
                f"[cyan]{ts}[/cyan] swm.write [b]{payload.get('slot')}[/b] "
                f"src={payload.get('source')}  |v|={_fmt_float(payload.get('norm'), 3)}"
            )
        elif topic == "recursion.run.start":
            recursion.reset_run()
            activity.append_event(
                f"[magenta]{ts}[/magenta] recursion.start "
                f"r_max={payload.get('max_rounds')}  m={payload.get('m_latent_steps')}  "
                f"organs={payload.get('organ_slot_count')}"
            )
        elif topic == "recursion.round.start":
            activity.append_event(
                f"[magenta]{ts}[/magenta] round.start r{payload.get('round')} "
                f"slot={payload.get('thought_slot')}  inputs={payload.get('input_slot_count')}"
            )
        elif topic == "recursion.round.complete":
            recursion.record_round(payload)
            cos = payload.get("cosine_to_previous")
            cos_str = "—" if cos is None or cos == float("-inf") else f"{float(cos):.3f}"
            activity.append_event(
                f"[magenta]{ts}[/magenta] round.done r{payload.get('round')} "
                f"halt={payload.get('halt')}  reason={payload.get('reason')}  cos={cos_str}"
            )
        elif topic == "recursion.run.complete":
            recursion.record_run_complete(payload)
            activity.append_event(
                f"[green]{ts}[/green] recursion.done rounds={payload.get('rounds')} "
                f"halt={payload.get('halt_reason')}"
            )
        elif topic == "latent.think.start":
            latent.record_think_start(payload)
            activity.append_event(
                f"[blue]{ts}[/blue] latent.start m={payload.get('m_latent_steps')}  "
                f"prompt_seq={payload.get('prompt_seq_len')}"
            )
        elif topic == "latent.think.complete":
            latent.record_think_complete(payload)
            activity.append_event(
                f"[blue]{ts}[/blue] latent.done seq={payload.get('final_seq_len')}  "
                f"|h|={_fmt_float(payload.get('last_hidden_norm'), 3)}"
            )
        elif topic == "prediction_error.record":
            joint = float(payload.get("joint_free_energy") or 0.0)
            self._joint_efe_trend.append(joint)
            activity.append_event(
                f"[yellow]{ts}[/yellow] err.record [b]{payload.get('source')}[/b] "
                f"e={_fmt_float(payload.get('error'), 3)}  joint={joint:.3f}"
            )
        elif topic == "imagination.cycle":
            activity.append_event(
                f"[#a995ff]{ts}[/#a995ff] imagination K={payload.get('k_trajectories')} "
                f"T={payload.get('t_horizon')}  chosen={payload.get('chosen_index')} "
                f"efe={_fmt_float(payload.get('chosen_efe'), 3)}"
            )
        elif topic == "frame.comprehend":
            activity.append_event(
                f"[white]{ts}[/white] comprehend [b]{_fmt_intent(payload.get('intent'))}[/b] "
                f"conf={_fmt_float(payload.get('confidence'), 3)}"
            )

    def _refresh_panels(self) -> None:
        try:
            self.query_one("#mrs-swm-slots", SWMSlotsPanel).update_from_swm(self.mind.swm)
            self.query_one("#mrs-alignments", AlignmentPanel).update_from_registry(
                self.mind.alignment_registry
            )
            self.query_one("#mrs-prediction-error", PredictionErrorPanel).update_from_vector(
                self.mind.prediction_errors
            )
            self.query_one("#mrs-active-thought", ActiveThoughtPanel).update_from_swm(
                self.mind.swm, self.mind.vsa
            )
        except Exception:
            logger.exception("MRS TUI: failed to refresh panels")

        if self._joint_efe_trend:
            self.query_one("#mrs-joint-spark", Sparkline).data = list(self._joint_efe_trend)

    def _refresh_status(self) -> None:
        try:
            device = next(self.mind.host.parameters()).device
        except Exception:
            device = "?"

        status = (
            f" model={self.mind.llama_model_id}   device={device}   "
            f"db={self.mind.db_path}   namespace={self.mind.namespace}   "
            f"D_swm={self.mind.swm.dim}"
        )
        self.query_one("#mrs-status", Static).update(status)

    # -- chat ----------------------------------------------------------------

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

        chatlog = self.query_one("#mrs-chatlog", RichLog)
        chatlog.write(f"[bold cyan]You[/bold cyan]  {text}")

        self._messages.append({"role": "user", "content": text})
        self._reply_buffer.clear()
        self.query_one("#mrs-streaming", Static).update("[bold magenta]Assistant[/bold magenta]  …")
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
            logger.exception("MRS TUI chat_reply failed")
            self.app.call_from_thread(self._on_reply_error, str(exc))

    def _on_token(self, piece: str) -> None:
        self._reply_buffer.append(piece)
        running = "".join(self._reply_buffer)
        self.query_one("#mrs-streaming", Static).update(
            f"[bold magenta]Assistant[/bold magenta]  {running}"
        )

    def _on_reply_done(self, _intent: str, _confidence: float, reply: str) -> None:
        self.busy = False
        text = "".join(self._reply_buffer) or reply
        text = text.strip() or "[empty reply]"
        self._messages.append({"role": "assistant", "content": text})
        chatlog = self.query_one("#mrs-chatlog", RichLog)
        chatlog.write(f"[bold magenta]Assistant[/bold magenta]  {text}")
        self.query_one("#mrs-streaming", Static).update("")
        self._reply_buffer.clear()

    def _on_reply_error(self, err: str) -> None:
        self.busy = False
        chatlog = self.query_one("#mrs-chatlog", RichLog)
        chatlog.write(f"[red]error: {err}[/red]")
        self.query_one("#mrs-streaming", Static).update("")
        if self._messages and self._messages[-1].get("role") == "user":
            self._messages.pop()

    def action_clear_chat(self) -> None:
        self._messages.clear()
        chatlog = self.query_one("#mrs-chatlog", RichLog)
        chatlog.clear()
        chatlog.write("[dim]chat cleared[/dim]")

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:  # noqa: ARG002
        if event.state in (WorkerState.SUCCESS, WorkerState.ERROR, WorkerState.CANCELLED):
            self.busy = False


def _build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="Mosaic Recursive Substrate debug TUI.")


def run_mrs_tui(argv: list[str] | None = None) -> None:
    from ..cli import (
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

    _build_parser().parse_args(argv)

    configure_lab_session(silent_stderr_default=True)

    bus = default_bus()
    handler = attach_core_logs_to_bus(bus)
    mind: SubstrateController | None = None

    try:
        mind = build_substrate_controller(bus=bus)
        start_background_stack(mind)

        app = MRSDebug(mind=mind, bus=bus)
        app.run()
    finally:
        if mind is not None:
            stop_background_stack(mind)

        detach_core_log_handler(handler)


def main() -> None:
    run_mrs_tui()


if __name__ == "__main__":
    main()
