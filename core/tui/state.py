from __future__ import annotations

from typing import Any

from textual.widgets import Static

from core.infra.constants import BRAND_SOFT

from .components import _rich_section_title, _titled_placeholder
from .styles import _CSS_BRAND_PANEL_BODY


class StatePanel(Static):
    """A titled panel that renders a list of string lines under the header."""

    DEFAULT_CSS = f"""
    StatePanel {{
{_CSS_BRAND_PANEL_BODY}    }}
    StatePanel > .title {{
        text-style: bold;
        color: {BRAND_SOFT};
    }}
    """

    def __init__(self, title: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self._title = title
        self._lines: list[str] = []

    def render(self) -> str:
        head = _rich_section_title(self._title)

        if not self._lines:
            return _titled_placeholder(head)

        return head + "\n" + "\n".join(self._lines)

    def set_lines(self, lines: list[str]) -> None:
        self._lines = list(lines)
        self.refresh()
