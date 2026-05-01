"""Default logging for the lab package.

``asi_broca_core`` installs a ``StreamHandler`` on the ``asi_broca_core`` logger
so ``DEBUG`` messages from substrate code are visible without extra setup.

Override with environment variables:

  ASI_BROCA_LOG_LEVEL   — DEBUG, INFO, WARNING, ERROR (default: DEBUG)
  ASI_BROCA_LOG_SILENT — if set to 1/true/yes, skip installing the handler

This is safe to call multiple times (only the first call attaches a handler).
"""

from __future__ import annotations

import logging
import os

_CONFIGURED = False


def configure_lab_logging() -> None:
    """Attach stderr logging for the ``asi_broca_core`` namespace at DEBUG by default."""

    global _CONFIGURED
    if _CONFIGURED:
        return
    silent = os.environ.get("ASI_BROCA_LOG_SILENT", "").strip().lower() in {"1", "true", "yes", "on"}
    if silent:
        _CONFIGURED = True
        return

    level_name = os.environ.get("ASI_BROCA_LOG_LEVEL", "DEBUG").strip().upper()
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        level = logging.DEBUG

    pkg = logging.getLogger("asi_broca_core")
    pkg.setLevel(level)

    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    pkg.addHandler(handler)
    pkg.propagate = False
    _CONFIGURED = True
