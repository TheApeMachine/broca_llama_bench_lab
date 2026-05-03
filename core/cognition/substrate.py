"""Compatibility re-export — controller implementation is :mod:`core.substrate.controller`."""

from __future__ import annotations

from core.host.llama_broca_host import load_llama_broca_host
from core.substrate.controller import SubstrateController

__all__ = ["SubstrateController", "load_llama_broca_host"]
