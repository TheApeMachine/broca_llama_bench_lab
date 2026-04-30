from __future__ import annotations

import os

import pytest


def _hf_token_available() -> bool:
    if os.environ.get("HF_TOKEN", "").strip():
        return True
    try:
        from huggingface_hub import HfFolder

        return bool(HfFolder.get_token())
    except Exception:
        return False


@pytest.fixture
def llama_broca_loaded() -> None:
    """Gate tests that download/load Hugging Face Llama checkpoints."""

    pytest.importorskip("transformers")
    if not _hf_token_available():
        pytest.skip("Need Hugging Face auth: set HF_TOKEN or run `huggingface-cli login` for Llama-backed tests.")
