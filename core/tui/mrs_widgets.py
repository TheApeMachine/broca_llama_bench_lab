"""Textual widgets for the MRS (Mosaic Recursive Substrate) debug TUI.

Each widget renders one concern of the substrate at high detail:

* :class:`SWMSlotsPanel` — the substrate working memory: every slot, its
  source organ, write tick, and L2 norm. Sorted by recency.
* :class:`RecursionPanel` — the per-round trace of the latest recursive
  rollout: round index, halt decision, cosine to previous, slot pointers.
* :class:`PredictionErrorPanel` — bar chart of per-organ prediction
  errors plus the joint free energy.
* :class:`AlignmentPanel` — every closed-form alignment matrix registered
  on the substrate, with input/output dims.
* :class:`LatentDecoderPanel` — the chat-time decoder's m, host, and
  alignment status.
* :class:`ActiveThoughtPanel` — current active.thought slot: norm, ticks
  since last write, top-k cosine matches against the VSA codebook.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch
from textual.widgets import Static

from ..infra.constants import BRAND_SOFT, ONLINE, WARNING

from .components import _fmt_float, _rich_section_title, _titled_placeholder
from .styles import _CSS_BRAND_PANEL_BODY, _DIM_EM_DASH


_BAR_FULL = "█"
_BAR_HALF = "▌"
_BAR_EMPTY = "░"


def _bar(value: float, *, width: int = 12, vmax: float = 1.0) -> str:
    if vmax <= 0.0:
        return _BAR_EMPTY * width

    norm = max(0.0, min(1.0, float(value) / float(vmax)))
    full = int(norm * width)
    half = 1 if (norm * width - full) >= 0.5 else 0

    if full + half > width:
        full = width
        half = 0

    return _BAR_FULL * full + _BAR_HALF * half + _BAR_EMPTY * (width - full - half)


class _BasePanel(Static):
    """Common base: bordered, titled, set_lines.

    Uses :meth:`Static.update` so Textual's auto-height layout pass picks up
    the new content size — overriding ``render()`` and calling ``refresh()``
    redraws but does not re-measure, which silently clipped the panels to
    their initial single-line content.
    """

    DEFAULT_CSS = f"""
    _BasePanel {{
{_CSS_BRAND_PANEL_BODY}    }}
    """

    def __init__(self, title: str, **kwargs: Any) -> None:
        super().__init__("", **kwargs)
        self._title = title
        self._lines: list[str] = []
        self._render_now()

    def set_lines(self, lines: Iterable[str]) -> None:
        self._lines = list(lines)
        self._render_now()

    def _render_now(self) -> None:
        head = _rich_section_title(self._title)
        if not self._lines:
            self.update(_titled_placeholder(head))
            return
        self.update(head + "\n" + "\n".join(self._lines))


class SWMSlotsPanel(_BasePanel):
    """List substrate working memory slots, most-recently-written first."""

    DEFAULT_CSS = f"""
    SWMSlotsPanel {{
{_CSS_BRAND_PANEL_BODY}        min-height: 14;
    }}
    """

    def update_from_swm(self, swm: Any, *, max_rows: int = 30) -> None:
        slots = sorted(list(swm), key=lambda s: -int(s.written_at_tick))
        lines = [
            f"[dim]dim={swm.dim}  slots={len(swm)}  showing {min(len(slots), max_rows)}[/dim]",
        ]

        if not slots:
            lines.append(_DIM_EM_DASH)
        else:
            for slot in slots[:max_rows]:
                norm = float(slot.vector.norm().item())
                lines.append(
                    f"[{BRAND_SOFT}]#{slot.written_at_tick:>4}[/{BRAND_SOFT}] "
                    f"{slot.name[:30]:<30} "
                    f"[dim]{slot.source.value[:18]:<18}[/dim] "
                    f"|v|={_fmt_float(norm, 2)}"
                )
            if len(slots) > max_rows:
                lines.append(f"[dim]… {len(slots) - max_rows} more slots[/dim]")

        self.set_lines(lines)


class RecursionPanel(_BasePanel):
    """Display the most recent recursion trace."""

    DEFAULT_CSS = f"""
    RecursionPanel {{
{_CSS_BRAND_PANEL_BODY}        min-height: 12;
    }}
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(title="Recursion", **kwargs)
        self._trace_summary: dict[str, Any] | None = None
        self._round_history: list[dict[str, Any]] = []

    def record_round(self, payload: dict[str, Any]) -> None:
        self._round_history.append(dict(payload))
        if len(self._round_history) > 8:
            self._round_history = self._round_history[-8:]
        self._render_lines()

    def record_run_complete(self, payload: dict[str, Any]) -> None:
        self._trace_summary = dict(payload)
        self._render_lines()

    def reset_run(self) -> None:
        self._round_history = []
        self._render_lines()

    def _render_lines(self) -> None:
        lines: list[str] = []
        if self._trace_summary:
            lines.append(
                f"last run: {int(self._trace_summary.get('rounds', 0))} round(s) "
                f"halt=[{ONLINE}]{self._trace_summary.get('halt_reason') or '—'}[/{ONLINE}]"
            )
            lines.append(
                f"final thought: [dim]{(self._trace_summary.get('final_thought_slot') or '—')[:32]}[/dim]"
            )
        else:
            lines.append("[dim]no recursion run yet[/dim]")

        if self._round_history:
            lines.append("")
            lines.append("[dim]rounds:[/dim]")
            for h in self._round_history:
                halt_glyph = f"[{ONLINE}]●[/{ONLINE}]" if h.get("halt") else f"[{WARNING}]○[/{WARNING}]"
                cos = h.get("cosine_to_previous")
                cos_str = "—" if cos is None or cos == float("-inf") else f"{float(cos):.3f}"
                lines.append(
                    f"  {halt_glyph} r{int(h.get('round', 0))} "
                    f"{h.get('reason', '—')[:14]:<14} "
                    f"cos={cos_str}"
                )

        self.set_lines(lines)


class PredictionErrorPanel(_BasePanel):
    """Bar chart of per-organ prediction errors + joint free energy."""

    DEFAULT_CSS = f"""
    PredictionErrorPanel {{
{_CSS_BRAND_PANEL_BODY}        min-height: 10;
    }}
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(title="Prediction error (per organ)", **kwargs)

    def update_from_vector(self, errors: Any) -> None:
        sources = errors.sources()
        if not sources:
            self.set_lines(["[dim]no organ has reported error yet[/dim]"])
            return

        lines: list[str] = []
        joint = 0.0

        for src in sources:
            entry = errors.get(src)
            err = float(entry.error)
            joint += err
            lines.append(
                f"{src.value[:10]:<10} {_bar(err, width=12)} {err:.3f}"
            )

        lines.append("")
        lines.append(f"[{BRAND_SOFT}]joint free energy[/{BRAND_SOFT}] = {joint:.3f}  ({len(sources)} organs)")
        self.set_lines(lines)


class AlignmentPanel(_BasePanel):
    """List every registered closed-form alignment matrix."""

    DEFAULT_CSS = f"""
    AlignmentPanel {{
{_CSS_BRAND_PANEL_BODY}        min-height: 8;
    }}
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(title="Closed-form alignments", **kwargs)

    def update_from_registry(self, registry: Any) -> None:
        items = sorted(list(registry), key=lambda a: a.name)
        if not items:
            self.set_lines([_DIM_EM_DASH])
            return

        lines = [f"[dim]registered: {len(items)}[/dim]"]
        for a in items:
            lines.append(
                f"[{BRAND_SOFT}]{type(a).__name__[:12]:<12}[/{BRAND_SOFT}] "
                f"{a.name[:18]:<18} "
                f"{a.d_in:>5} → {a.d_out:<5}"
            )
        self.set_lines(lines)


class LatentDecoderPanel(_BasePanel):
    """Latent decoder configuration + most-recent rollout activity."""

    DEFAULT_CSS = f"""
    LatentDecoderPanel {{
{_CSS_BRAND_PANEL_BODY}        min-height: 10;
    }}
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(title="Latent decoder", **kwargs)
        self._think_count: int = 0
        self._last_event: dict[str, Any] | None = None

    def update_static(self, *, decoder: Any, halt: Any) -> None:
        self._decoder = decoder
        self._halt = halt
        self._render_lines()

    def record_think_start(self, payload: dict[str, Any]) -> None:
        self._last_event = {"type": "start", **payload}
        self._render_lines()

    def record_think_complete(self, payload: dict[str, Any]) -> None:
        self._think_count += 1
        self._last_event = {"type": "complete", **payload}
        self._render_lines()

    def _render_lines(self) -> None:
        if not getattr(self, "_decoder", None):
            self.set_lines(["[dim]not constructed[/dim]"])
            return

        lines = [
            f"m_latent_steps: [b]{self._decoder.m_latent_steps}[/b]",
            f"max_rounds: [b]{self._halt.max_rounds}[/b]",
            f"convergence floor: {self._halt.convergence_floor:.3f}",
            f"alignment: [{BRAND_SOFT}]{self._decoder.alignment.name}[/{BRAND_SOFT}] "
            f"({self._decoder.alignment.d_in}→{self._decoder.alignment.d_out})",
            "",
            f"think calls: {self._think_count}",
        ]

        if self._last_event:
            ev_type = self._last_event.get("type", "?")
            if ev_type == "start":
                lines.append(
                    f"[dim]last start: m={self._last_event.get('m_latent_steps')} "
                    f"prompt={self._last_event.get('prompt_seq_len')}[/dim]"
                )
            else:
                lines.append(
                    f"[dim]last done: seq={self._last_event.get('final_seq_len')} "
                    f"|h|={_fmt_float(self._last_event.get('last_hidden_norm'), 3)}[/dim]"
                )

        self.set_lines(lines)


class ActiveThoughtPanel(_BasePanel):
    """Current active.thought slot: norm, age, top cosine matches."""

    DEFAULT_CSS = f"""
    ActiveThoughtPanel {{
{_CSS_BRAND_PANEL_BODY}        min-height: 12;
    }}
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(title="Active thought", **kwargs)

    def update_from_swm(self, swm: Any, codebook: Any) -> None:
        if not swm.has("active.thought"):
            self.set_lines(["[dim]no active thought (no recursion has run yet)[/dim]"])
            return

        slot = swm.read("active.thought")
        norm = float(slot.vector.norm().item())
        lines = [
            f"slot: [b]active.thought[/b]",
            f"|v| = {norm:.4f}",
            f"written at tick: #{slot.written_at_tick}",
            f"source: [dim]{slot.source.value}[/dim]",
            "",
            "[dim]top cosine matches:[/dim]",
        ]

        atoms = self._codebook_atoms(codebook)
        if not atoms:
            lines.append(_DIM_EM_DASH)
        else:
            scored = self._top_cosine(slot.vector, atoms, k=5)
            for name, cos in scored:
                lines.append(
                    f"  [{BRAND_SOFT}]{name[:18]:<18}[/{BRAND_SOFT}] {_bar(max(0.0, cos), width=8)} {cos:+.3f}"
                )

        self.set_lines(lines)

    @staticmethod
    def _codebook_atoms(codebook: Any) -> dict[str, torch.Tensor]:
        atoms = getattr(codebook, "_atoms", None)
        if not isinstance(atoms, dict):
            return {}
        return dict(atoms)

    @staticmethod
    def _top_cosine(query: torch.Tensor, atoms: dict[str, torch.Tensor], *, k: int) -> list[tuple[str, float]]:
        q = query.detach().to(torch.float32).view(-1)
        qn = q.norm().clamp_min(1e-12)
        scored: list[tuple[str, float]] = []
        for name, atom in atoms.items():
            a = atom.detach().to(torch.float32).view(-1)
            an = a.norm().clamp_min(1e-12)
            cos = float((q @ a / (qn * an)).item())
            scored.append((name, cos))
        scored.sort(key=lambda kv: -abs(kv[1]))
        return scored[:k]


class MRSActivityPanel(_BasePanel):
    """High-detail MRS event log (most recent first)."""

    DEFAULT_CSS = f"""
    MRSActivityPanel {{
{_CSS_BRAND_PANEL_BODY}        min-height: 20;
    }}
    """

    _MAX_EVENTS = 200

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(title="MRS event stream", **kwargs)
        self._events: list[str] = []
        self._counter: int = 0

    def append_event(self, line: str) -> None:
        self._counter += 1
        self._events.append(line)
        if len(self._events) > self._MAX_EVENTS:
            self._events = self._events[-self._MAX_EVENTS:]
        # Most-recent first; cap displayed lines so the panel stays readable.
        head = [f"[dim]events seen: {self._counter}  buffered: {len(self._events)}[/dim]"]
        body = list(reversed(self._events[-50:]))
        self.set_lines(head + body)
