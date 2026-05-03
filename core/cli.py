"""Shared runtime setup for every Mosaic CLI (one consistent substrate stack).

Interactive paths build :class:`core.substrate.controller.SubstrateController`
through this module so device resolution, SQLite paths, Hugging Face token
handling, tokenizer environment, and background services match everywhere.
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Any

from .host.llama_broca_host import quiet_transformers_benchmark_log_warnings, resolve_hf_hub_token
from .infra.logging_setup import configure_lab_logging
from .substrate.controller import SubstrateController
from .substrate.runtime import (
    BROCA_BACKGROUND_INTERVAL_S,
    CHAT_NAMESPACE,
    default_model_id,
    default_substrate_sqlite_path,
    ensure_parent_dir,
)
from .system.device import pick_torch_device
from .workspace import BaseWorkspace, WorkspaceBuilder
from .workspace.log_handler import LogToBusHandler

_DEFAULT_HF_TOKEN = object()


class LabSessionConfigurator:
    """Applies process-level logging defaults for CLI sessions."""

    def configure(self, *, silent_stderr_default: bool = False) -> None:
        if silent_stderr_default:
            os.environ.setdefault("LOG_SILENT", "1")

        configure_lab_logging()


class ModelRuntimePreparer:
    """Prepares tokenizer and host logging settings before model construction."""

    def ensure_tokenizers_env(self) -> None:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    def quiet_host_log_noise(self) -> None:
        quiet_transformers_benchmark_log_warnings()

    def prepare(self) -> None:
        self.ensure_tokenizers_env()
        self.quiet_host_log_noise()


class SelfImproveEnvironmentPolicy:
    """Reads opt-in background self-improvement settings from the environment."""

    def enabled(self) -> bool:
        return os.environ.get("BROCA_SELF_IMPROVE", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }


class BackgroundStackManager:
    """Starts and stops controller background services as one concern."""

    def __init__(self, self_improve_policy: SelfImproveEnvironmentPolicy | None = None) -> None:
        self.self_improve_policy = self_improve_policy or SelfImproveEnvironmentPolicy()

    def start_self_improve_if_enabled(self, controller: SubstrateController) -> None:
        if self.self_improve_policy.enabled():
            controller.start_self_improve_worker(interval_s=None, enabled=True)

    def start(self, controller: SubstrateController) -> None:
        controller.start_background(interval_s=BROCA_BACKGROUND_INTERVAL_S)
        self.start_self_improve_if_enabled(controller)

    def stop(self, controller: SubstrateController) -> None:
        controller.stop_background()
        controller.stop_self_improve_worker()


class SubstrateControllerFactory:
    """Constructs the fully wired controller through the canonical path."""

    def __init__(self, runtime: ModelRuntimePreparer | None = None) -> None:
        self.runtime = runtime or ModelRuntimePreparer()

    def build(
        self,
        *,
        bus: BaseWorkspace | None = None,
        seed: int = 0,
        db_path: str | Path | None = None,
        namespace: str | None = None,
        llama_model_id: str | None = None,
        device: Any = None,
        hf_token: str | bool | None | object = _DEFAULT_HF_TOKEN,
        preload_host_tokenizer: tuple[Any, Any] | None = None,
    ) -> SubstrateController:
        self.runtime.prepare()

        resolved_path = Path(db_path) if db_path is not None else default_substrate_sqlite_path()
        ensure_parent_dir(resolved_path)
        resolved_namespace = namespace if namespace is not None else CHAT_NAMESPACE
        model_id = llama_model_id if llama_model_id is not None else default_model_id()
        resolved_device = pick_torch_device(device)
        token_kw: str | bool | None

        if hf_token is _DEFAULT_HF_TOKEN:
            token_kw = resolve_hf_hub_token(None)
        else:
            token_kw = hf_token

        controller = SubstrateController(
            seed=int(seed),
            db_path=resolved_path,
            namespace=resolved_namespace,
            llama_model_id=model_id,
            device=resolved_device,
            hf_token=token_kw,
            preload_host_tokenizer=preload_host_tokenizer,
        )

        if bus is not None:
            controller.event_bus = bus

        return controller

    def build_broca_mind(self, *, bus: BaseWorkspace | None = None) -> SubstrateController:
        warnings.warn(
            "build_broca_mind is deprecated; use build_substrate_controller",
            DeprecationWarning,
            stacklevel=3,
        )

        return self.build(bus=bus)


class CoreLogBridge:
    """Attaches and detaches core logs from an event bus."""

    def attach(self, bus: BaseWorkspace, *, env_var: str = "TUI_LOG_LEVEL") -> logging.Handler:
        log_level = getattr(logging, str(os.environ.get(env_var, "INFO")).upper(), logging.INFO)
        handler = LogToBusHandler(bus, level=log_level)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logging.getLogger("core").addHandler(handler)

        return handler

    def detach(self, handler: logging.Handler) -> None:
        try:
            logging.getLogger("core").removeHandler(handler)
        except Exception as exc:
            logging.getLogger("core").debug("Failed to remove handler %s: %s", handler, exc)


class WorkspaceProvider:
    """Owns workspace construction for callers that need a bus first."""

    def default_bus(self) -> BaseWorkspace:
        return WorkspaceBuilder().process_default()


class CliRuntimeFacade:
    """Compatibility surface composed from small runtime services."""

    def __init__(
        self,
        *,
        session: LabSessionConfigurator | None = None,
        model_runtime: ModelRuntimePreparer | None = None,
        background: BackgroundStackManager | None = None,
        factory: SubstrateControllerFactory | None = None,
        logs: CoreLogBridge | None = None,
        workspaces: WorkspaceProvider | None = None,
    ) -> None:
        runtime = model_runtime or ModelRuntimePreparer()
        self.session = session or LabSessionConfigurator()
        self.model_runtime = runtime
        self.background = background or BackgroundStackManager()
        self.factory = factory or SubstrateControllerFactory(runtime=runtime)
        self.logs = logs or CoreLogBridge()
        self.workspaces = workspaces or WorkspaceProvider()

    def configure_lab_session(self, *, silent_stderr_default: bool = False) -> None:
        self.session.configure(silent_stderr_default=silent_stderr_default)

    def ensure_tokenizers_env(self) -> None:
        self.model_runtime.ensure_tokenizers_env()

    def quiet_host_log_noise(self) -> None:
        self.model_runtime.quiet_host_log_noise()

    def prepare_model_runtime(self) -> None:
        self.model_runtime.prepare()

    def self_improve_enabled_from_env(self) -> bool:
        return self.background.self_improve_policy.enabled()

    def start_self_improve_if_enabled(self, controller: SubstrateController) -> None:
        self.background.start_self_improve_if_enabled(controller)

    def start_background_stack(self, controller: SubstrateController) -> None:
        self.background.start(controller)

    def stop_background_stack(self, controller: SubstrateController) -> None:
        self.background.stop(controller)

    def build_substrate_controller(self, **kwargs: Any) -> SubstrateController:
        return self.factory.build(**kwargs)

    def build_broca_mind(self, *, bus: BaseWorkspace | None = None) -> SubstrateController:
        return self.factory.build_broca_mind(bus=bus)

    def attach_core_logs_to_bus(
        self,
        bus: BaseWorkspace,
        *,
        env_var: str = "TUI_LOG_LEVEL",
    ) -> logging.Handler:
        return self.logs.attach(bus, env_var=env_var)

    def detach_core_log_handler(self, handler: logging.Handler) -> None:
        self.logs.detach(handler)

    def default_bus(self) -> BaseWorkspace:
        return self.workspaces.default_bus()


_FACADE = CliRuntimeFacade()

configure_lab_session = _FACADE.configure_lab_session
ensure_tokenizers_env = _FACADE.ensure_tokenizers_env
quiet_host_log_noise = _FACADE.quiet_host_log_noise
prepare_model_runtime = _FACADE.prepare_model_runtime
self_improve_enabled_from_env = _FACADE.self_improve_enabled_from_env
start_self_improve_if_enabled = _FACADE.start_self_improve_if_enabled
start_background_stack = _FACADE.start_background_stack
stop_background_stack = _FACADE.stop_background_stack
build_substrate_controller = _FACADE.build_substrate_controller
build_broca_mind = _FACADE.build_broca_mind
attach_core_logs_to_bus = _FACADE.attach_core_logs_to_bus
detach_core_log_handler = _FACADE.detach_core_log_handler
default_bus = _FACADE.default_bus

__all__ = [
    "BackgroundStackManager",
    "CliRuntimeFacade",
    "CoreLogBridge",
    "LabSessionConfigurator",
    "ModelRuntimePreparer",
    "SelfImproveEnvironmentPolicy",
    "SubstrateControllerFactory",
    "WorkspaceProvider",
    "attach_core_logs_to_bus",
    "build_broca_mind",
    "build_substrate_controller",
    "configure_lab_session",
    "default_bus",
    "detach_core_log_handler",
    "ensure_tokenizers_env",
    "prepare_model_runtime",
    "quiet_host_log_noise",
    "self_improve_enabled_from_env",
    "start_background_stack",
    "start_self_improve_if_enabled",
    "stop_background_stack",
]
