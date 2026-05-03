"""Plain-terminal chat entrypoints and substrate-biased decode orchestration."""

from __future__ import annotations

from typing import Any

from .orchestrator import ChatOrchestrator

__all__ = ["ChatOrchestrator", "main", "run_chat_repl"]


def __getattr__(name: str) -> Any:
    if name == "main":
        from .repl import main as m

        return m
    if name == "run_chat_repl":
        from .repl import run_chat_repl as r

        return r
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
