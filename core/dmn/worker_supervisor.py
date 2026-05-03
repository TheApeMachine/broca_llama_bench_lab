"""WorkerSupervisor — start/stop the substrate's two background daemons.

The substrate runs two independent background loops:

* :class:`CognitiveBackgroundWorker` — the DMN, ticking through consolidation
  / separation / latent discovery / chunking / REM phases.
* :class:`SelfImproveDockerWorker` — the Docker-isolated self-improve loop
  that proposes patches and opens PRs.

Each is opt-in. The supervisor owns their lifecycle so the controller's
public surface stops carrying ``start_X`` / ``stop_X`` methods.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ..dmn import CognitiveBackgroundWorker, DMNConfig


if TYPE_CHECKING:
    from ..substrate.controller import SubstrateController


logger = logging.getLogger(__name__)


class WorkerSupervisor:
    """Lifecycle controller for the DMN and self-improve daemons."""

    def __init__(self, mind: "SubstrateController") -> None:
        self._mind = mind

    def start_background(
        self,
        *,
        interval_s: float = 5.0,
        config: DMNConfig | None = None,
    ) -> CognitiveBackgroundWorker:
        mind = self._mind
        sess = mind.session
        if sess.background_worker is None:
            sess.background_worker = CognitiveBackgroundWorker(
                mind,
                interval_s=interval_s,
                config=config,
                motor_trainer=mind.motor_trainer,
            )
        else:
            sess.background_worker.interval_s = max(0.1, float(interval_s))
            if config is not None:
                sess.background_worker.config = config
        sess.background_worker.start()
        return sess.background_worker

    def stop_background(self) -> None:
        sess = self._mind.session
        if sess.background_worker is not None:
            sess.background_worker.stop()

    def start_self_improve(
        self,
        *,
        interval_s: float | None = None,
        enabled: bool | None = None,
    ) -> Any:
        """Start Docker-backed self-improve loop (separate from DMN background).

        See :mod:`core.workers.docker_self_improve_worker` for environment
        variables and prerequisites (``GITHUB_TOKEN``, Docker, and ``repo``
        scope).
        """

        try:
            from ..workers.docker_self_improve_worker import (
                SelfImproveConfig,
                SelfImproveDockerWorker,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "Could not import core.workers.docker_self_improve_worker (self-improve worker). "
                "Ensure project dependencies are installed and Docker is available on the host."
            ) from exc

        mind = self._mind
        sess = mind.session
        cfg = SelfImproveConfig()
        if enabled is not None:
            cfg.enabled = bool(enabled)
        if interval_s is not None:
            cfg.interval_s = max(60.0, float(interval_s))
        if sess.self_improve_worker is None:
            sess.self_improve_worker = SelfImproveDockerWorker(mind, config=cfg)
        else:
            sess.self_improve_worker.config = cfg
        sess.self_improve_worker.start()
        return sess.self_improve_worker

    def stop_self_improve(self, timeout: float = 5.0) -> None:
        sw = self._mind.session.self_improve_worker
        if sw is not None:
            sw.stop(timeout=timeout)
