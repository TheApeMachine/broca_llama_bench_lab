"""Shared runtime setup for every Mosaic CLI (one consistent substrate stack).

Interactive paths build :class:`core.cognition.substrate.SubstrateController` through
this module so device resolution, SQLite paths, Hugging Face token handling,
tokenizers env, and background/self-improve services match everywhere.
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Any

from .cognition.substrate import SubstrateController
from .system.device import pick_torch_device
from .system.event_bus import EventBus, LogToBusHandler, get_default_bus
from .host.llama_broca_host import quiet_transformers_benchmark_log_warnings, resolve_hf_hub_token
from .infra.logging_setup import configure_lab_logging
from .substrate.runtime import (
    BROCA_BACKGROUND_INTERVAL_S,
    CHAT_NAMESPACE,
    default_model_id,
    default_substrate_sqlite_path,
    ensure_parent_dir,
)


def parse_device_env() -> str | None:
    """``M_DEVICE`` override, with deprecated ``ASI_DEVICE`` fallback."""

    raw_m = os.environ.get("M_DEVICE")

    if raw_m is not None and raw_m.strip() != "":
        return raw_m.strip()

    legacy = os.environ.get("ASI_DEVICE")

    if legacy is not None and legacy.strip() != "":
        warnings.warn(
            "ASI_DEVICE is deprecated; set M_DEVICE for the default torch device override.",
            DeprecationWarning,
            stacklevel=2,
        )

        return legacy.strip()

    return None


def configure_lab_session(*, silent_stderr_default: bool = False) -> None:
    """Apply logging defaults, then :func:`configure_lab_logging`.

    TUIs default to ``LOG_SILENT=1`` so stderr does not fight the screen; REPL
    and headless harnesses keep normal stderr behavior unless the user sets env.
    """

    if silent_stderr_default:
        os.environ.setdefault("LOG_SILENT", "1")

    configure_lab_logging()


def ensure_tokenizers_env() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def quiet_host_log_noise() -> None:
    quiet_transformers_benchmark_log_warnings()


def prepare_model_runtime() -> None:
    """Before loading transformers / SubstrateController: env + log noise."""

    ensure_tokenizers_env()
    quiet_host_log_noise()


def self_improve_enabled_from_env() -> bool:
    return os.environ.get("BROCA_SELF_IMPROVE", "").strip().lower() in {"1", "true", "yes", "on"}


def start_self_improve_if_enabled(controller: SubstrateController) -> None:
    if self_improve_enabled_from_env():
        controller.start_self_improve_worker(interval_s=None, enabled=True)


def start_background_stack(controller: SubstrateController) -> None:
    controller.start_background(interval_s=BROCA_BACKGROUND_INTERVAL_S)
    start_self_improve_if_enabled(controller)


def stop_background_stack(controller: SubstrateController) -> None:
    controller.stop_background()
    controller.stop_self_improve_worker()


_DEFAULT_HF_TOKEN = object()


def build_substrate_controller(
    *,
    bus: EventBus | None = None,
    seed: int = 0,
    db_path: str | Path | None = None,
    namespace: str | None = None,
    llama_model_id: str | None = None,
    device: Any = None,
    hf_token: str | bool | None | object = _DEFAULT_HF_TOKEN,
    preload_host_tokenizer: tuple[Any, Any] | None = None,
) -> SubstrateController:
    """Construct ``SubstrateController`` through the sole supported wiring surface.

    Chat REPL/TUI, benchmarks, Docker workers, and tests should instantiate the
    live stack only here (optionally overriding paths and model handles) so
    device handling, SQLite location, namespaces, tokenizer env setup, and token
    resolution stay aligned with AGENTS \"one system fully wired\".
    """

    prepare_model_runtime()

    rp = Path(db_path) if db_path is not None else default_substrate_sqlite_path()
    ensure_parent_dir(rp)
    ns = namespace if namespace is not None else CHAT_NAMESPACE
    mid = llama_model_id if llama_model_id is not None else default_model_id()
    resolved_device = pick_torch_device(parse_device_env()) if device is None else device

    if hf_token is _DEFAULT_HF_TOKEN:
        token_kw: str | bool | None = resolve_hf_hub_token(None)
    else:
        token_kw = hf_token

    controller = SubstrateController(
        seed=int(seed),
        db_path=rp,
        namespace=ns,
        llama_model_id=mid,
        device=resolved_device,
        hf_token=token_kw,
        preload_host_tokenizer=preload_host_tokenizer,
    )

    if bus is not None:
        controller.event_bus = bus

    return controller


def build_broca_mind(*, bus: EventBus | None = None) -> SubstrateController:
    """Deprecated name for :func:`build_substrate_controller`."""

    warnings.warn(
        "build_broca_mind is deprecated; use build_substrate_controller",
        DeprecationWarning,
        stacklevel=2,
    )

    return build_substrate_controller(bus=bus)


def attach_core_logs_to_bus(bus: EventBus, *, env_var: str = "TUI_LOG_LEVEL") -> logging.Handler:
    log_level = getattr(logging, str(os.environ.get(env_var, "INFO")).upper(), logging.INFO)
    handler = LogToBusHandler(bus, level=log_level)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger("core").addHandler(handler)

    return handler


def detach_core_log_handler(handler: logging.Handler) -> None:
    try:
        logging.getLogger("core").removeHandler(handler)
    except Exception as e:
        logging.getLogger("core").debug("Failed to remove handler %s: %s", handler, e)


def default_bus() -> EventBus:
    """Explicit singleton for call sites that need the bus before the controller exists."""

    return get_default_bus()
