
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Callable

# Keep the CPU test suite deterministic and prevent OpenMP/PyTorch worker pools
# from lingering at interpreter shutdown on constrained CI sandboxes.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pytest
import torch

try:
    torch.set_num_threads(1)
except RuntimeError:
    pass
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    # PyTorch may reject interop-thread changes after a backend initialized; the
    # environment variables above still keep fresh test processes bounded.
    pass


@pytest.fixture(autouse=True)
def _mosaic_test_sqlite(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Isolate substrate SQLite so unit tests never touch ``runs/``."""

    monkeypatch.setenv("MOSAIC_UNDER_TEST", "1")
    monkeypatch.setenv("MOSAIC_TEST_DB", str(tmp_path / "mosaic_test.sqlite"))


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


# ---------------------------------------------------------------------------
# Stub LLM + tokenizer for relation-extraction tests.
#
# Production code requires a real LLM with no heuristic fallback. These stubs
# mimic the HuggingFace surface that ``LLMRelationExtractor`` calls into so the
# extractor can be exercised in unit tests without loading model weights. The
# default extraction rule (a thin SVO heuristic) lives only in test code; it
# represents what a competent real LLM would return on the test inputs.
# ---------------------------------------------------------------------------


def _default_stub_extract(sentence: str) -> tuple[str, str, str] | None:
    words = re.findall(r"[A-Za-z0-9_]+", sentence.lower())
    while words and words[0] in ("the", "a", "an"):
        words.pop(0)
    if len(words) < 3:
        return None
    return words[0], "is in", words[-1]


class StubGenerationTokenizer:
    """Pretends to be an HF tokenizer; captures the prompt and primes the LLM stub."""

    def __init__(self, llm: "StubGenerationLLM", extractor: Callable[[str], tuple[str, str, str] | None]):
        self._llm = llm
        self._extractor = extractor
        self.pad_token_id = 0
        self.eos_token_id = 0

    def __call__(self, prompt: str, return_tensors: str = "pt"):
        sentence_marker = "Sentence: "
        json_marker = "\nJSON:"
        idx = prompt.rfind(sentence_marker)
        if idx < 0:
            sentence = ""
        else:
            tail = prompt[idx + len(sentence_marker):]
            sentence = tail.split(json_marker, 1)[0].strip()
        triple = self._extractor(sentence)
        self._llm._next_response = (
            json.dumps({"subject": triple[0], "relation": triple[1], "object": triple[2]})
            if triple is not None
            else "no triple"
        )
        return {
            "input_ids": torch.zeros((1, 4), dtype=torch.long),
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
        }

    def decode(self, ids, skip_special_tokens: bool = True):
        return self._llm._next_response


class StubGenerationLLM:
    """Pretends to be an HF causal LM. The decode after generate returns whatever the tokenizer primed."""

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self._next_response: str = ""

    def parameters(self):
        yield torch.zeros(1, device=self.device)

    def generate(
        self,
        *,
        input_ids,
        attention_mask=None,
        max_new_tokens=64,
        do_sample=False,
        pad_token_id=None,
        temperature=None,
        top_p=None,
        **kwargs,
    ):
        _ = attention_mask, max_new_tokens, do_sample, pad_token_id, temperature, top_p, kwargs
        return torch.zeros((1, input_ids.shape[1] + 4), dtype=torch.long, device=self.device)


def make_stub_llm_pair(extractor: Callable[[str], tuple[str, str, str] | None] | None = None) -> tuple[StubGenerationLLM, StubGenerationTokenizer]:
    """Construct a paired stub LLM and HF tokenizer wired to a deterministic extractor."""

    llm = StubGenerationLLM()
    tok = StubGenerationTokenizer(llm, extractor or _default_stub_extract)
    return llm, tok
