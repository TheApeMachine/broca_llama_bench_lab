"""Re-export of :class:`StatePanel` so ``research_lab.tui.bench`` works.

The bench dashboard was moved out of ``core.tui`` into ``research_lab.tui`` but
its widget vocabulary is shared with the chat dashboard, so we just re-export
the canonical :class:`StatePanel` from :mod:`core.tui.state`.
"""

from __future__ import annotations

from core.tui.state import StatePanel

__all__ = ["StatePanel"]
