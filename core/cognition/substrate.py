"""Compatibility re-export for the substrate controller."""

from __future__ import annotations

from typing import Any

from core.substrate.controller import SubstrateController


def load_llama_broca_host(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
    from core.host.llama_broca_host import load_llama_broca_host as loader

    return loader(*args, **kwargs)


__all__ = ["SubstrateController", "load_llama_broca_host"]
