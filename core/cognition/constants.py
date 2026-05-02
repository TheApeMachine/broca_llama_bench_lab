"""Defaults for the cognitive substrate stack (SQLite + hosted LLM)."""

import os

# Default Hugging Face model id when ``MODEL_ID`` is unset (informative string, not numeric).
DEFAULT_CHAT_MODEL_ID: str = os.environ.get("MODEL_ID", "meta-llama/Llama-3.2-1B-Instruct")

# Minimum semantic confidence treated as usable; typically in [0.0, 1.0].
SEMANTIC_CONFIDENCE_FLOOR: float = 0.5

# Threshold on candidate-vs-current log-score gap (nats) before revising a belief;
# tune in roughly [0.0, 1.0] with ``consolidate_claims_once``.
BELIEF_REVISION_LOG_ODDS_THRESHOLD: float = 0.5

# Minimum distinct supporting claims needed before a belief revision is considered; must be >= 1.
BELIEF_REVISION_MIN_CLAIMS: int = 2
