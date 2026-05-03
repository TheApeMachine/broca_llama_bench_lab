"""Runtime façade — mirrors orchestration attributes wired onto the controller."""

from __future__ import annotations

from typing import Any


class SubstrateRuntime:
    """Thin view over :meth:`~core.substrate.orchestration_linker.OrchestrationLinker.wire` results."""

    def __init__(self, mind: Any) -> None:
        for name in (
            "chat",
            "graft_frame",
            "comprehension",
            "preference",
            "algebra",
            "workers",
            "native_tools",
            "macros",
            "claims",
            "deferred_relations",
            "inspector",
            "speaker",
        ):
            if not hasattr(mind, name):
                raise RuntimeError(
                    f"SubstrateRuntime: controller missing {name!r}; "
                    "OrchestrationLinker.wire must run first"
                )
            setattr(self, name, getattr(mind, name))
