"""Unified runtime defaults for the single Mosaic substrate."""

from __future__ import annotations

import os
from pathlib import Path

# --- SQLite -----------------------------------------------------------------

CANONICAL_SUBSTRATE_SQLITE = Path("runs/broca_substrate.sqlite")


def default_substrate_sqlite_path() -> Path:
    """Return the SQLite path used by SubstrateController and benchmarks (non-tests).

    When ``MOSAIC_UNDER_TEST=1``, ``MOSAIC_TEST_DB`` must point at the writable
    per-test database file (set by pytest ``conftest``).
    """

    if os.environ.get("MOSAIC_UNDER_TEST", "").strip().casefold() in {"1", "true", "yes"}:
        raw = os.environ.get("MOSAIC_TEST_DB", "").strip()
        if not raw:
            raise RuntimeError(
                "MOSAIC_UNDER_TEST is set but MOSAIC_TEST_DB is missing; fix tests/conftest wiring."
            )
        return Path(raw)
    return CANONICAL_SUBSTRATE_SQLITE


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# --- Model / device (infra via env until values are dynamically derived) ----


def default_model_id() -> str:
    for key in ("MODEL_ID", "BENCHMARK_MODEL"):
        raw = os.environ.get(key)
        if raw is None:
            continue
        s = raw.strip()
        if s:
            return s
    return "meta-llama/Llama-3.2-1B-Instruct"


def benchmark_output_root() -> Path:
    return Path(os.environ.get("BENCHMARK_OUTPUT_DIR", "runs/benchmarks"))

# --- Chat (generation knobs fixed here; callers should not expose CLI tuning) -

CHAT_MAX_NEW_TOKENS = 512
CHAT_DO_SAMPLE = False
CHAT_TEMPERATURE = 0.7
CHAT_TOP_P = 0.9
CHAT_NAMESPACE = "chat"
BROCA_BACKGROUND_INTERVAL_S = 5.0

# --- Benchmark suite (fixed product configuration) ---------------------------

BENCHMARK_ENGINE = "both"
BENCHMARK_NATIVE_PRESET = "standard"
BENCHMARK_LM_EVAL_PRESET = "standard"
BENCHMARK_LIMIT = 250
BENCHMARK_FIXED_SEED = 0
BENCHMARK_GEN_MAX_NEW_TOKENS = 128
