from __future__ import annotations

from typing import Any

from textual.widgets import Static

from core.infra.constants import OFFLINE, ONLINE, WARNING

from .components import _rich_section_title, _titled_placeholder
from .styles import _CSS_BRAND_PANEL_BODY


class SystemsMatrix(Static):
    """A compact label matrix showing which substrate systems are online.

    Each row is one subsystem with a status dot (● online, ○ offline,
    ◐ degraded) followed by a name and a tiny detail string. The matrix is
    refreshed from the SubstrateController snapshot on every TUI tick.
    """

    DEFAULT_CSS = f"""
    SystemsMatrix {{
{_CSS_BRAND_PANEL_BODY}    }}
    """

    _COLORS = {
        "on": ONLINE,
        "off": OFFLINE,
        "warn": WARNING,
    }
    _GLYPHS = {
        "on": "●",
        "off": "○",
        "warn": "◐",
    }

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self._entries: list[tuple[str, str, str]] = []

    def render(self) -> str:
        head = _rich_section_title("Systems")

        if not self._entries:
            return _titled_placeholder(head)

        lines = [head]

        for status, name, detail in self._entries:
            color = self._COLORS.get(status, OFFLINE)
            glyph = self._GLYPHS.get(status, self._GLYPHS["off"])
            label = f"{name:<14}"
            tail = f" [dim]{detail}[/dim]" if detail else ""

            lines.append(f"[{color}]{glyph}[/{color}] {label}{tail}")

        return "\n".join(lines)

    def set_entries(self, entries: list[tuple[str, str, str]]) -> None:
        self._entries = list(entries)
        self.refresh()
