
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


@pytest.fixture(autouse=True)
def _autostub_substrate_organs(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace organs with canned stubs whenever a test builds a ``SubstrateController``.

    ``SubstrateController.__init__`` instantiates :class:`ExtractionOrgan`
    and :class:`AffectOrgan`, which lazy-load HuggingFace weights on first
    use. The first ``comprehend`` call therefore tries to download
    ``fastino/gliner2-base-v1`` and SamLowe's GoEmotions model, neither of
    which the unit suite should depend on. We wrap ``__init__`` with a
    post-step that swaps the freshly-built organs out for canned stubs so
    every test gets a substrate that *functions* without network access.

    Tests that genuinely want the real organs (e.g. ``test_organ_integration``)
    can opt out by adding the ``real_organs`` marker.
    """

    if request.node.get_closest_marker("real_organs"):
        return

    import core.cognition.substrate as substrate_mod

    real_init = substrate_mod.SubstrateController.__init__

    def patched_init(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        stub_substrate_organs(self)

    monkeypatch.setattr(substrate_mod.SubstrateController, "__init__", patched_init)


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


# ---------------------------------------------------------------------------
# Substrate organ stubbing.
#
# Every ``SubstrateController.comprehend`` call now runs through the intent
# gate (``ExtractionOrgan.classify``) and the affect organ
# (``AffectOrgan.detect``), which lazy-load model weights from HuggingFace on
# first use. Tests that exercise memory, journals, or grafts do not care
# about the gate's accuracy — they only need a substrate that *functions*.
# ``stub_substrate_organs`` swaps in tiny canned implementations so those
# tests stay fast and deterministic.
#
# Tests that DO want to exercise the real organs (``test_organ_integration``,
# ``test_substrate_intent_gating`` opting into stubs explicitly) should not
# call this helper.
# ---------------------------------------------------------------------------


class _CannedExtractionOrgan:
    """Minimal stand-in for :class:`core.organs.extraction.ExtractionOrgan`.

    Defaults ``classify`` to "statement" so the substrate's intent gate
    routes everything as actionable, which matches the pre-organ behavior
    that legacy tests expect. Tests can pass per-fragment overrides for
    either ``classify`` or ``extract_relations`` results.
    """

    def __init__(
        self,
        *,
        intent_responses: "dict[str, list[tuple[str, float]]] | None" = None,
        relation_responses: "dict[str, list] | None" = None,
        default_intent_label: str = "statement",
        default_intent_score: float = 0.95,
    ):
        self._intent = intent_responses or {}
        self._relations = relation_responses or {}
        self._default_intent_label = default_intent_label
        self._default_intent_score = float(default_intent_score)
        self.classify_calls: list[str] = []
        self.relation_calls: list[str] = []

    def classify(self, text: str, *, labels, multi_label: bool = True, threshold: float = 0.0):
        self.classify_calls.append(text)
        for fragment, scores in self._intent.items():
            if fragment in text.lower():
                return list(scores)
        # Match the smallest set of pragmatic features the legacy substrate
        # relied on: a trailing ``?`` is a question; otherwise the canned
        # default applies. Tests that need finer behavior pass explicit
        # ``intent_responses``.
        if "?" in text:
            return [("question", 0.95)]
        return [(self._default_intent_label, self._default_intent_score)]

    def extract_relations(self, text: str, *, entity_labels=None, relation_labels=None):
        _ = entity_labels, relation_labels
        self.relation_calls.append(text)
        for fragment, rels in self._relations.items():
            if fragment in text.lower():
                return list(rels)
        if "?" in text:
            return []
        return _heuristic_extract_relations(text)


class _CannedAffectOrgan:
    """Returns a fixed neutral :class:`core.organs.affect.AffectState`."""

    def __init__(self, state=None):
        from core.organs.affect import AffectState

        self._state = state if state is not None else AffectState(
            dominant_emotion="neutral",
            dominant_score=0.5,
            valence=0.0,
            arousal=0.0,
        )
        self.calls: list[str] = []

    def detect(self, text: str, *, threshold=None):
        _ = threshold
        self.calls.append(text)
        return self._state


def _heuristic_extract_relations(text: str):
    """Tiny SVO heuristic — ``"X is in Y"`` → triple, otherwise empty.

    This mirrors the ``_default_stub_extract`` behavior used by the legacy
    LLM extractor stubs in this conftest, so memory-layer tests that send
    sentences like ``"ada is in rome ."`` continue to produce a triple
    after we route extraction through the organ.
    """

    from core.organs.extraction import ExtractedRelation

    import re

    words = re.findall(r"[A-Za-z0-9_]+", text.lower())
    while words and words[0] in ("the", "a", "an"):
        words.pop(0)
    if len(words) < 3:
        return []
    return [
        ExtractedRelation(
            subject=words[0],
            predicate="is_in",
            object=words[-1],
            confidence=0.9,
        )
    ]


def stub_substrate_organs(
    mind,
    *,
    intent_responses: "dict[str, list[tuple[str, float]]] | None" = None,
    relation_responses: "dict[str, list] | None" = None,
    affect_state=None,
    default_intent_label: str = "statement",
    default_intent_score: float = 0.95,
) -> _CannedExtractionOrgan:
    """Replace a substrate's organs with deterministic canned stubs.

    Returns the canned extraction organ so tests can inspect ``classify_calls``
    or ``relation_calls`` after the fact.
    """

    from core.cognition.intent_gate import IntentGate
    from core.cognition.organ_relation_extractor import OrganRelationExtractor

    extraction = _CannedExtractionOrgan(
        intent_responses=intent_responses,
        relation_responses=relation_responses,
        default_intent_label=default_intent_label,
        default_intent_score=default_intent_score,
    )
    mind.extraction_organ = extraction
    mind.affect_organ = _CannedAffectOrgan(affect_state)
    mind.intent_gate = IntentGate(extraction)
    mind.router.extractor = OrganRelationExtractor(
        intent_gate=mind.intent_gate,
        organ=extraction,
    )
    return extraction
