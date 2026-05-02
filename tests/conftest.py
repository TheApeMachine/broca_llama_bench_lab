
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
def _autostub_substrate_encoders(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace heavy encoders with canned stubs whenever a test builds a ``SubstrateController``.

    ``SubstrateController.__init__`` instantiates :class:`ExtractionEncoder`
    and :class:`AffectEncoder`, which lazy-load HuggingFace weights on first
    use. The first ``comprehend`` call therefore tries to download
    ``fastino/gliner2-base-v1`` and SamLowe's GoEmotions model, neither of
    which the unit suite should depend on. We wrap ``__init__`` with a
    post-step that swaps the freshly-built encoders out for canned stubs so
    every test gets a substrate that *functions* without network access.

    Tests that genuinely want the real weights (e.g. ``test_encoder_integration``)
    can opt out by adding the ``real_encoders`` marker.
    """

    if request.node.get_closest_marker("real_encoders"):
        return

    import core.cognition.substrate as substrate_mod

    real_init = substrate_mod.SubstrateController.__init__

    def patched_init(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        stub_substrate_encoders(self)

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

    def apply_chat_template(self, messages, add_generation_prompt: bool = True, return_tensors: str | None = "pt"):
        _ = messages, add_generation_prompt, return_tensors
        return torch.tensor([[1, 2, 3]], dtype=torch.long)


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
# Substrate encoder stubbing.
#
# Every ``SubstrateController.comprehend`` call now runs through the semantic
# cascade, the extraction encoder, and the affect encoder. Those load model
# weights from HuggingFace on first use. Tests that exercise memory, journals,
# or grafts do not care about classifier accuracy — they only need a substrate
# that functions. ``stub_substrate_encoders`` swaps in tiny canned
# implementations so those tests stay fast and deterministic.
#
# Tests that DO want to exercise the real weights (``test_encoder_integration``,
# ``test_substrate_intent_gating`` opting into stubs explicitly) should not
# call this helper.
# ---------------------------------------------------------------------------


class _CannedExtractionEncoder:
    """Minimal stand-in for :class:`core.encoders.extraction.ExtractionEncoder`.

    Defaults ``classify`` to "statement" so the substrate's intent gate
    routes everything as actionable, which matches the pre-extractor behavior
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
        self.identity_calls: list[str] = []

    def extract_identity_relations(self, text: str):
        self.identity_calls.append(text)
        return []

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


class _CannedAffectEncoder:
    """Returns a fixed neutral :class:`core.encoders.affect.AffectState`."""

    def __init__(self, state=None):
        from core.encoders.affect import AffectState

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


class _CannedSemanticCascade:
    def __init__(self, extraction: _CannedExtractionEncoder):
        self.extraction = extraction

    def intent_scores(self, text: str):
        from core.cognition.intent_gate import INTENT_LABELS

        ranked = self.extraction.classify(text, labels=INTENT_LABELS, multi_label=False, threshold=0.0)
        if not ranked:
            return {
                "label": "",
                "confidence": 0.0,
                "scores": {},
                "allows_storage": False,
                "evidence": {},
            }
        scores = {label: 0.0 for label in INTENT_LABELS}
        for label, score in ranked:
            scores[label] = float(score)
        top_label, top_score = ranked[0]
        return {
            "label": top_label,
            "confidence": float(top_score),
            "scores": scores,
            "allows_storage": top_label == "statement",
            "evidence": {"stub": True},
        }


def _heuristic_extract_relations(text: str):
    """Tiny SVO heuristic — ``"X is in Y"`` → triple, otherwise empty.

    This mirrors the ``_default_stub_extract`` behavior used by the legacy
    LLM extractor stubs in this conftest, so memory-layer tests that send
    sentences like ``"ada is in rome ."`` continue to produce a triple
    after we route extraction through the encoder.
    """

    from core.encoders.extraction import ExtractedRelation

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


def stub_substrate_encoders(
    mind,
    *,
    intent_responses: "dict[str, list[tuple[str, float]]] | None" = None,
    relation_responses: "dict[str, list] | None" = None,
    affect_state=None,
    default_intent_label: str = "statement",
    default_intent_score: float = 0.95,
) -> _CannedExtractionEncoder:
    """Replace a substrate's encoders with deterministic canned stubs.

    Returns the canned extraction encoder so tests can inspect ``classify_calls``
    or ``relation_calls`` after the fact.
    """

    from core.cognition.intent_gate import IntentGate
    from core.cognition.encoder_relation_extractor import EncoderRelationExtractor

    extraction = _CannedExtractionEncoder(
        intent_responses=intent_responses,
        relation_responses=relation_responses,
        default_intent_label=default_intent_label,
        default_intent_score=default_intent_score,
    )
    mind.extraction_encoder = extraction
    mind.affect_encoder = _CannedAffectEncoder(affect_state)
    mind.semantic_cascade = _CannedSemanticCascade(extraction)
    mind.intent_gate = IntentGate(mind.semantic_cascade)
    mind.router.extractor = EncoderRelationExtractor(
        intent_gate=mind.intent_gate,
        extraction=extraction,
    )
    return extraction
