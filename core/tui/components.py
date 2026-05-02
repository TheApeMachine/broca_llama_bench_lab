from typing import Any

from core.infra.constants import BRAND_SOFT

from .styles import _ACTIVITY_LOG_LEVEL_MARKUP, _DIM_EM_DASH

def _rich_section_title(label: str) -> str:
    return f"[b][{BRAND_SOFT}]{label}[/{BRAND_SOFT}][/b]"


def _titled_placeholder(title_rich: str) -> str:
    return f"{title_rich}\n{_DIM_EM_DASH}"


def _fmt_float(v: Any, prec: int = 3) -> str:
    if v is None:
        return "—"

    try:
        return f"{float(v):.{prec}f}"
    except (TypeError, ValueError):
        return str(v)


def _fmt_intent(intent: str | None) -> str:
    return intent or "—"


def _frame_summary_lines(src: dict[str, Any]) -> list[str]:
    """Intent, subject, answer, confidence — shared by last_chat and latest_frame."""
    return [
        f"intent: [b]{_fmt_intent(src.get('intent'))}[/b]",
        f"subject: {src.get('subject') or '—'}",
        f"answer: {src.get('answer') or '—'}",
        f"confidence: {_fmt_float(src.get('confidence'))}",
    ]


def _rich_yes_no_strong(yes: bool) -> str:
    return "[green]yes[/green]" if yes else "[red]no[/red]"


def _rich_yes_no_soft(yes: bool) -> str:
    return "[green]yes[/green]" if yes else "[dim]no[/dim]"


def _activity_line_frame_comprehend(ts: str, payload: dict[str, Any]) -> str:
    return (
        f"[cyan]{ts}[/cyan] frame [b]{_fmt_intent(payload.get('intent'))}[/b]"
        f"  conf={_fmt_float(payload.get('confidence'))}"
        f"  subject={payload.get('subject') or '—'}"
    )


def _activity_line_intrinsic_cue(ts: str, payload: dict[str, Any]) -> str:
    return (
        f"[yellow]{ts}[/yellow] cue [b]{payload.get('faculty')}[/b]"
        f"  urgency={_fmt_float(payload.get('urgency'))}"
    )


def _activity_line_consolidation(ts: str, payload: dict[str, Any]) -> str:
    return f"[green]{ts}[/green] consolidation reflections={payload.get('reflections', 0)}"


def _activity_line_dmn_tick(ts: str, payload: dict[str, Any], duration_ms: float) -> str:
    return (
        f"[magenta]{ts}[/magenta] dmn tick {payload.get('iteration')}"
        f"  {duration_ms:.0f}ms  reflections={payload.get('reflections', 0)}"
    )


def _activity_line_self_improve_start(ts: str, payload: dict[str, Any]) -> str:
    return f"[blue]{ts}[/blue] self-improve start run={payload.get('run_id', '')[:8]}"


def _activity_line_self_improve_complete(ts: str, payload: dict[str, Any]) -> str:
    err = payload.get("error")
    run_id = payload.get("run_id", "")[:8]

    if err:
        return f"[red]{ts}[/red] self-improve fail run={run_id}  {err[:80]}"

    return f"[blue]{ts}[/blue] self-improve done run={run_id}  {payload.get('summary') or ''}"


def _activity_line_log(ts: str, payload: dict[str, Any]) -> str:
    level = payload.get("level", "INFO")
    msg = payload.get("msg", "")
    color = _ACTIVITY_LOG_LEVEL_MARKUP.get(level, "white")
    name = payload.get("name", "")

    return f"[{color}]{ts} {level:7} {name}[/{color}]  {msg}"
