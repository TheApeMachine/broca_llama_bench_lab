"""Defaults for the cognitive substrate stack (SQLite + hosted LLM)."""

from __future__ import annotations

import os

DEFAULT_CHAT_MODEL_ID = os.environ.get("MODEL_ID", "meta-llama/Llama-3.2-1B-Instruct")
SEMANTIC_CONFIDENCE_FLOOR = 0.5
BELIEF_REVISION_LOG_ODDS_THRESHOLD = 0.5
BELIEF_REVISION_MIN_CLAIMS = 1
