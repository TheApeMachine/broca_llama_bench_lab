"""Ensure ``core.tui.chat`` imports every ``_`` helper it calls from ``components``.

Static guard: avoids ``NameError`` at runtime when a panel refresh path first
exercises a missing import (Textual only surfaces one error by default).
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _function_names_starting_with_underscore(module_path: Path) -> set[str]:
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_") and not node.name.startswith("__"):
            names.add(node.name)
    return names


def _components_imports_from_chat(chat_path: Path) -> set[str]:
    tree = ast.parse(chat_path.read_text(encoding="utf-8"))
    for node in tree.body:
        # `from .components import ...` → module="components", level=1
        if isinstance(node, ast.ImportFrom) and node.module == "components" and node.level == 1:
            return {alias.name for alias in node.names}
    return set()


def _bare_underscore_calls(tree: ast.AST) -> set[str]:
    """``func`` is a simple Name (not ``self._m``, not ``super()``)."""
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            n = node.func.id
            if n.startswith("_") and not n.startswith("__"):
                out.add(n)
    return out


def test_tui_chat_imports_match_component_calls() -> None:
    components = _REPO_ROOT / "core" / "tui" / "components.py"
    chat = _REPO_ROOT / "core" / "tui" / "chat.py"
    helpers = _function_names_starting_with_underscore(components)
    imported = _components_imports_from_chat(chat)
    called = _bare_underscore_calls(ast.parse(chat.read_text(encoding="utf-8")))
    needed = called & helpers
    missing = sorted(needed - imported)
    assert not missing, (
        "core/tui/chat.py calls these helpers from core/tui/components.py but does not "
        f"import them in the `from .components import (...)` block: {', '.join(missing)}"
    )
