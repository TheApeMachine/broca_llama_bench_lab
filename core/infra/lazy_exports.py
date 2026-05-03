"""Lazy package export registry."""

from __future__ import annotations

import importlib
import os
from types import ModuleType
from typing import Any


class LazyExportRegistry:
    """Resolve public package exports without eager subsystem imports."""

    def __init__(self, *, package: str, exports: dict[str, tuple[str, str]]) -> None:
        self.package = package
        self.exports = dict(exports)

    def names(self) -> list[str]:
        return list(self.exports)

    def resolve(self, module_globals: dict[str, Any], name: str) -> Any:
        spec = self.exports.get(name)

        if spec is None:
            raise AttributeError(f"module {self.package!r} has no attribute {name!r}")

        module_name, attr = spec
        module = importlib.import_module(module_name, self.package)
        value = getattr(module, attr)
        module_globals[name] = value

        return value

    def dir_entries(self, module_globals: dict[str, Any]) -> list[str]:
        return sorted(set(module_globals) | set(self.exports))

    def auto_configure_logging(self, module: ModuleType, *, env_var: str) -> None:
        requested = str(os.environ.get(env_var, "")).strip().lower()

        if requested in {"1", "true"}:
            self.resolve(module.__dict__, "configure_lab_logging")()
