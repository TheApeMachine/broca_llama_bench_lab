"""Default logging for the lab package.

By default ``configure_lab_logging`` attaches **both** a stderr stream handler
(so the chat CLI keeps its live debug feed) and a rotating file handler at
``runs/broca.log`` (so the full event stream survives a session and is grep-able
after the fact).

Override with environment variables:

  LOG_LEVEL   — DEBUG, INFO, WARNING, ERROR (default: DEBUG)
  LOG_SILENT  — if set to 1/true/yes, skip stderr handler (file still
                          attaches unless ``LOG_FILE_DISABLED`` is set)
  LOG_FILE    — override the log-file path (default: runs/broca.log).
                          Set to empty to use the default; set
                          ``LOG_FILE_DISABLED=1`` to skip the file
                          handler altogether.
  LOG_FILE_LEVEL — independent level for the file handler (default:
                          DEBUG; falls back to LOG_LEVEL).
  LOG_FILE_MAX_BYTES — rotating size cap (default: 16 MB).
  LOG_FILE_BACKUPS — number of rotated backups to keep (default: 4).

Idempotent: calling more than once is a no-op after the first configuration
(beginning ``configure_lab_logging`` sets the configured flag immediately so handler
installation cannot partially double-run across threads).
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path

_CONFIGURED = False
_CONFIG_LOCK = threading.Lock()
_DEFAULT_LOG_FILE = Path("runs") / "broca.log"
_DEFAULT_MAX_BYTES = 16 * 1024 * 1024
_DEFAULT_BACKUPS = 4

_FILE_FORMATTER = logging.Formatter(
    fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
_STREAM_FORMATTER = logging.Formatter(
    fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_level(name: str | None, default: int) -> int:
    if not name:
        return default
    cand = getattr(logging, name.strip().upper(), None)
    return cand if isinstance(cand, int) else default


def _resolve_int(name: str | None, default: int) -> int:
    if not name:
        return default
    try:
        return int(name)
    except ValueError:
        return default


def _resolve_log_file_path() -> Path | None:
    if _truthy(os.environ.get("LOG_FILE_DISABLED")):
        return None
    raw = os.environ.get("LOG_FILE")
    if raw is None or raw.strip() == "":
        return _DEFAULT_LOG_FILE
    return Path(raw).expanduser()


def configure_lab_logging() -> None:
    """Attach stderr + rotating-file logging for the ``core`` namespace."""

    global _CONFIGURED
    with _CONFIG_LOCK:
        if _CONFIGURED:
            return
        _CONFIGURED = True

        silent = _truthy(os.environ.get("LOG_SILENT"))
        base_level = _resolve_level(
            os.environ.get("LOG_LEVEL"), logging.DEBUG
        )
        file_level = _resolve_level(
            os.environ.get("LOG_FILE_LEVEL"), base_level
        )

        pkg = logging.getLogger("core")
        # Logger threshold must be the more permissive of the two handlers so each
        # handler can independently filter without the parent logger swallowing
        # records below its own level.
        pkg.setLevel(min(base_level, file_level))
        pkg.propagate = False

        if not silent:
            stream = logging.StreamHandler()
            stream.setLevel(base_level)
            stream.setFormatter(_STREAM_FORMATTER)
            pkg.addHandler(stream)

        log_path = _resolve_log_file_path()
        if log_path is not None:
            try:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                max_bytes = _resolve_int(
                    os.environ.get("LOG_FILE_MAX_BYTES"), _DEFAULT_MAX_BYTES
                )
                backups = _resolve_int(
                    os.environ.get("LOG_FILE_BACKUPS"), _DEFAULT_BACKUPS
                )
                file_handler = RotatingFileHandler(
                    log_path,
                    maxBytes=max(1024, max_bytes),
                    backupCount=max(0, backups),
                    encoding="utf-8",
                )
                file_handler.setLevel(file_level)
                file_handler.setFormatter(_FILE_FORMATTER)
                pkg.addHandler(file_handler)
                pkg.info(
                    "logging initialized log_file=%s level=%s file_level=%s",
                    log_path,
                    logging.getLevelName(base_level),
                    logging.getLevelName(file_level),
                )
            except OSError as exc:
                # Falling back silently here would hide misconfiguration; surface to stderr handler.
                pkg.warning("failed to attach file handler at %s: %s", log_path, exc)
                if silent or not pkg.handlers:
                    print(
                        f"[core] WARNING: failed to attach file handler at {log_path}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
