
from __future__ import annotations

import json
import hashlib
import logging
import math
import os
import random
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .active_inference import (
    ActiveInferenceAgent,
    CoupledEFEAgent,
    ToolForagingAgent,
    build_causal_epistemic_pomdp,
    build_tiger_pomdp,
    entropy as belief_entropy,
)
from .causal import build_simpson_scm
from .chunking import (
    ChunkingDetectionConfig,
    CompiledMacro,
    DMNChunkingCompiler,
    MacroChunkRegistry,
    macro_frame_features,
)
from .continuous_frame import (
    COGNITIVE_FRAME_DIM,
    SKETCH_DIM,
    TextEncoder,
    frozen_subword_projector_from_model,
    pack_cognitive_frame,
    stable_sketch,
)
from .device_utils import pick_torch_device
from .grafts import BaseGraft, DEFAULT_GRAFT_TARGET_SNR, snr_magnitude, _state_confidence, _state_inertia
from .hf_tokenizer_compat import HuggingFaceBrocaTokenizer
from .llama_broca_host import LlamaBrocaHost, load_llama_broca_host
from .predictive_coding import lexical_surprise_gap
from .substrate_graph import EpisodeAssociationGraph, merge_epistemic_evidence_dict
from .tokenizer import speech_seed_ids, utterance_words
from .vsa import VSACodebook
from .hopfield import HopfieldAssociativeMemory
from .conformal import ConformalPredictor, PersistentConformalCalibration
from .hawkes import MultivariateHawkesProcess, PersistentHawkes, fit_excitation_em
from .preference_learning import DirichletPreference, PersistentPreference, feedback_polarity_from_text
from .ontological_expansion import OntologicalRegistry, PersistentOntologicalRegistry
from .causal_discovery import pc_algorithm, build_scm_from_skeleton
from .native_tools import NativeTool, NativeToolRegistry, ToolSandbox, ToolSynthesisError
from .dynamic_grafts import DynamicGraftSynthesizer, CapturedActivationMode, ACTIVATION_MODE_KIND
from .memory import SQLiteActivationMemory

logger = logging.getLogger(__name__)

DEFAULT_BROCA_MODEL_ID = os.environ.get("MODEL_ID", "meta-llama/Llama-3.2-1B-Instruct")
SEMANTIC_CONFIDENCE_FLOOR = 0.5
BELIEF_REVISION_LOG_ODDS_THRESHOLD = 0.5
BELIEF_REVISION_MIN_CLAIMS = 1


def _gap_population_stats(claims: Sequence[dict]) -> tuple[float, float] | None:
    """Population mean and std of prediction gaps across the claim corpus.

    Surprise is a moving target — what counts as a high-gap outlier depends on
    how the LLM scored every other observation. Returns ``None`` when there is
    not enough variance to anchor a Z-score (single sample or constant gaps),
    which leaves :func:`_claim_trust_weight` to fall back to its standard
    Gaussian prior with σ = 1.
    """

    gaps: list[float] = []
    for claim in claims:
        ev = claim.get("evidence")
        if not isinstance(ev, dict):
            continue
        raw = ev.get("prediction_gap")
        try:
            gap = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(gap) and gap > 0.0:
            gaps.append(gap)
    if len(gaps) < 2:
        return None
    mu = sum(gaps) / len(gaps)
    var = sum((g - mu) ** 2 for g in gaps) / len(gaps)
    sigma = math.sqrt(var)
    if sigma <= 1e-6:
        return None
    return mu, sigma


def _claim_trust_weight(claim: dict, *, gap_stats: tuple[float, float] | None = None) -> float:
    """Likelihood-ratio weight for prediction-error-weighted consolidation.

    The weight is the Gaussian likelihood ratio of the claim's surprise under
    the population's distribution of past surprises: ``exp(-0.5 * z^2)`` where
    ``z = (gap - μ) / σ``. Surprises smaller than μ collapse to a Z of zero
    (the claim is at-or-below typical), so a low-gap claim earns full trust.
    Without a population (``gap_stats is None``) the formula falls back to a
    unit-Gaussian baseline anchored at zero, which preserves the previous
    1/(1+gap)-style decay for callers that pass a single claim with no
    historical context.
    """

    ev = claim.get("evidence")
    if not isinstance(ev, dict):
        return 1.0
    raw = ev.get("prediction_gap")
    try:
        gap = float(raw)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(gap) or gap <= 0.0:
        return 1.0
    if gap_stats is None:
        mu, sigma = 0.0, 1.0
    else:
        mu, sigma = gap_stats
        sigma = max(1e-3, float(sigma))
    z = max(0.0, (gap - mu) / sigma)
    return math.exp(-0.5 * z * z)


@dataclass(frozen=True)
class ParsedClaim:
    subject: str
    predicate: str
    obj: str
    confidence: float
    evidence: dict[str, Any]


@dataclass(frozen=True)
class ParsedQuery:
    subject: str
    predicate: str
    confidence: float
    evidence: dict[str, Any]


def _word_tokens(toks: Sequence[str]) -> list[str]:
    return [t for t in toks if any(ch.isalnum() for ch in t)]


def _lexical_tokens(value: Any) -> list[str]:
    text = str(value).replace("_", " ").strip().lower()
    return _word_tokens(utterance_words(text))


def _is_question(toks: Sequence[str]) -> bool:
    return any(t == "?" for t in toks)


class RelationExtractor:
    """Pluggable subject/predicate/object extractor for declarative utterances.

    Implementations return ``None`` for non-declarative input (questions,
    fragments, or anything the LLM cannot resolve into a triple) and a
    ``ParsedClaim`` otherwise. Extractors are called per routing decision.
    """

    def extract_claim(self, utterance: str, toks: Sequence[str]) -> "ParsedClaim | None":  # pragma: no cover - protocol
        logger.debug(f"extract_claim: {utterance} {toks}")
        raise NotImplementedError


class LLMRelationExtractor(RelationExtractor):
    """Few-shot SVO extraction via the host LLM. There is no heuristic fallback.

    Uses ``host.llm.generate`` with a JSON few-shot prompt so subordinate
    clauses, passive voice, and stripped determiners are resolved by the
    language model rather than by a brittle regex. Out-of-distribution
    sentences that the LLM cannot turn into a clean ``{subject, relation,
    object}`` triple yield ``None``; the router then routes the utterance to
    the remaining faculties (active inference, causal effect) instead of
    fabricating a triple from string-splitting.

    The host must expose ``host.llm.generate`` and the tokenizer must expose
    ``tokenizer.inner`` with the standard HuggingFace surface (``__call__``
    returning ``input_ids``/``attention_mask``, plus ``decode``); production
    use is wired up by :class:`BrocaMind`. Tests should provide an HF-shaped
    stub LLM with the desired generation behavior.
    """

    PROMPT_TEMPLATE = (
        "Extract the subject, relation, and object of each sentence as JSON. "
        "Use lowercase. Strip articles. The relation is the verb phrase.\n"
        "Sentence: ada is in rome .\n"
        'JSON: {"subject":"ada","relation":"is in","object":"rome"}\n'
        "Sentence: the apple, which was incredibly red, fell from the tree .\n"
        'JSON: {"subject":"apple","relation":"fell from","object":"tree"}\n'
        "Sentence: the dog chased the ball .\n"
        'JSON: {"subject":"dog","relation":"chased","object":"ball"}\n'
        "Sentence: <SENTENCE>\n"
        "JSON: "
    )

    def __init__(self, host: Any, tokenizer: Any, *, max_new_tokens: int = 64, cache_size: int = 256):
        self.host = host
        self.tokenizer = tokenizer
        self.max_new_tokens = int(max_new_tokens)
        self._cache: dict[str, tuple[str, str, str] | None] = {}
        self._cache_size = max(0, int(cache_size))

    def extract_claim(self, utterance: str, toks: Sequence[str]) -> "ParsedClaim | None":
        if _is_question(toks):
            logger.debug(f"extract_claim: {utterance} {toks} is question")
            return None
        
        words = list(_word_tokens(toks))
        
        if len(words) < 3:
            logger.debug(f"extract_claim: {utterance} {toks} too few words")
            return None
        
        triple = self._llm_extract(utterance)
        
        if triple is None:
            logger.debug(f"extract_claim: {utterance} {toks} no triple")
            return None
        
        subject, predicate, obj = triple
        
        logger.debug(f"extract_claim: {utterance} {toks} {triple}")
        
        return ParsedClaim(
            subject=subject.lower(),
            predicate=predicate.lower(),
            obj=obj.lower(),
            confidence=1.0,
            evidence={
                "parser": "llm_relation_extractor",
                "predicate_surface": predicate,
                "source_words": words,
                "utterance": utterance,
            },
        )

    def _llm_extract(self, utterance: str) -> tuple[str, str, str] | None:
        key = utterance.strip()
        
        if key in self._cache:
            logger.debug(f"_llm_extract: {utterance} {key} in cache")
            return self._cache[key]
        
        result = self._llm_extract_uncached(key)

        if self._cache_size > 0:
            if len(self._cache) >= self._cache_size:
                self._cache.pop(next(iter(self._cache)))
        
            self._cache[key] = result
        
        logger.debug(f"_llm_extract: {utterance} {key} {result}")
        return result

    def _llm_extract_uncached(self, utterance: str) -> tuple[str, str, str] | None:
        llm = self.host.llm
        hf_tok = self.tokenizer.inner
        prompt = self.PROMPT_TEMPLATE.replace("<SENTENCE>", utterance)
        device = next(llm.parameters()).device
        encoded = hf_tok(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask")
        
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        pad_id = getattr(hf_tok, "pad_token_id", None) or getattr(hf_tok, "eos_token_id", None)
        
        with torch.no_grad():
            output = llm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
            )
        generated = output[0, input_ids.shape[1]:]
        new_text = hf_tok.decode(generated, skip_special_tokens=True)
        
        logger.debug("_llm_extract_uncached: utterance=%r raw_json_fragment=%r", utterance, new_text[:512] if len(new_text) > 512 else new_text)
        return self._parse_json_triple(new_text)

    @staticmethod
    def _parse_json_triple(text: str) -> tuple[str, str, str] | None:
        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        end = -1
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end < 0:
            return None
        try:
            obj = json.loads(text[start:end])
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(obj, dict):
            return None
        s = obj.get("subject")
        p = obj.get("relation")
        o = obj.get("object")
        if not isinstance(s, str) or not isinstance(p, str) or not isinstance(o, str):
            return None
        s, p, o = s.strip(), p.strip(), o.strip()
        if not s or not p or not o:
            return None
        return s, p, o


def _choose_subject(words: Sequence[str], known_subjects: Sequence[str]) -> str | None:
    if not words:
        return None
    known = {s.lower(): s.lower() for s in known_subjects}
    for word in words:
        got = known.get(word.lower())
        if got is not None:
            return got
    if known:
        return None
    return words[-1].lower()


def _choose_predicate(
    utterance: str,
    records: Sequence[tuple[str, str, float, dict]],
    text_encoder: TextEncoder | None,
) -> str:
    if not records:
        return ""
    if len(records) == 1:
        return records[0][0]
    query_vec = _text_vector(utterance, text_encoder)
    scored: list[tuple[float, str]] = []
    for pred, obj, conf, ev in records:
        evidence_text = " ".join(str(x) for x in (pred, obj, ev.get("predicate_surface", ""), ev.get("parser", "")))
        score = _cosine(query_vec, _text_vector(evidence_text, text_encoder)) + 0.05 * float(conf)
        scored.append((score, pred))
    return max(scored, key=lambda item: item[0])[1]


def _query_from_tokens(
    toks: Sequence[str],
    *,
    utterance: str,
    known_subjects: Sequence[str],
    records_for_subject: Callable[[str], Sequence[tuple[str, str, float, dict]]],
    text_encoder: TextEncoder | None,
) -> ParsedQuery | None:
    """Resolve a question into an existing subject/predicate memory lookup."""

    if not _is_question(toks):
        return None
    words = _word_tokens(toks)
    if not words:
        logger.debug("_query_from_tokens: empty words utterance=%r", utterance)
        return None
    subject = _choose_subject(words, known_subjects)
    if subject is None or not str(subject).strip():
        logger.debug("_query_from_tokens: no subject utterance=%r words=%s", utterance, words)
        return None
    records = list(records_for_subject(subject))
    predicate = _choose_predicate(utterance, records, text_encoder)
    if not predicate:
        logger.debug("_query_from_tokens: no predicate utterance=%r subject=%r n_records=%d", utterance, subject, len(records))
        return None
    logger.debug("_query_from_tokens: utterance=%r subject=%r predicate=%r", utterance, subject, predicate)
    return ParsedQuery(
        subject=subject,
        predicate=predicate,
        confidence=1.0,
        evidence={"parser": "open_memory_query", "source_words": words, "predicate_candidates": [r[0] for r in records]},
    )


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    den = (a.norm() * b.norm()).clamp_min(1e-12)
    return float(torch.dot(a.view(-1), b.view(-1)) / den)


def _text_vector(text: str, text_encoder: TextEncoder | None) -> torch.Tensor:
    if text_encoder is None:
        return stable_sketch(text)
    try:
        v = text_encoder(text)
    except (RuntimeError, ValueError):
        logger.error("text_encoder failed in _text_vector; falling back to stable_sketch", exc_info=True)
        v = stable_sketch(text)
    return v.detach().float().cpu().view(-1)


def _claim_prediction_gap(mind: "BrocaMind", utterance: str, claim: ParsedClaim) -> float | None:
    """Mean per-token surprise (graft - plain CE) of the host on a contradicting claim.

    Returns ``None`` when the host cannot run a graft-aware forward pass (e.g.
    test fakes). A larger positive gap means the LLM finds the claim less
    plausible; consolidation uses this as a trust attenuator so a low-prior
    statement repeated by an attacker requires more corroboration to flip a
    belief than a low-surprise statement.
    """

    try:
        plan_words = [claim.subject, claim.predicate, claim.obj, "."]
        broca_features = pack_cognitive_frame(
            "memory_write", claim.subject, claim.obj, float(claim.confidence), claim.evidence,
            text_encoder=mind.text_encoder,
        )
        _ce_g, _ce_p, gap = lexical_surprise_gap(
            mind.host,
            mind.tokenizer,
            utterance=utterance,
            plan_words=plan_words,
            broca_features=broca_features,
        )
        logger.debug("_claim_prediction_gap: gap=%s subject=%r pred=%r obj=%r", gap, claim.subject, claim.predicate, claim.obj)
        return float(gap)
    except (AttributeError, RuntimeError, TypeError, ValueError, StopIteration, IndexError):
        logger.debug("_claim_prediction_gap: unavailable host path utterance=%r", utterance[:200])
        return None


def _route_relevance(utterance: str, toks: Sequence[str], descriptors: Sequence[str], text_encoder: TextEncoder | None) -> float:
    """Continuous intent relevance plus lexical coverage from a route manifest."""

    descriptor_text = " ".join(descriptors)
    sem = max(0.0, _cosine(_text_vector(utterance, text_encoder), _text_vector(descriptor_text, text_encoder)))
    words = set(_word_tokens(toks))
    desc_words = set(_word_tokens(utterance_words(descriptor_text)))
    hits = words & desc_words
    overlap = len(hits) / max(1, min(len(words), len(desc_words)))
    hit_gate = 1.0 if hits else 0.0
    # Coefficients (0.50 + 0.35 + 0.25) sum above 1.0; normalize before clamping so the score stays in [0, 1].
    combined = 0.50 * sem + 0.35 * overlap + 0.25 * hit_gate
    return max(0.0, min(1.0, combined / 1.10))


def _frame_descriptor_tokens(frame: "CognitiveFrame") -> list[str]:
    parts: list[str] = []
    for value in (frame.intent, frame.subject, frame.answer):
        parts.extend(_lexical_tokens(value))
    for key, value in sorted((frame.evidence or {}).items()):
        parts.extend(_lexical_tokens(key))
        if isinstance(value, (str, int, float, bool)):
            parts.extend(_lexical_tokens(value))
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                parts.extend(_lexical_tokens(sub_key))
                if isinstance(sub_value, (str, int, float, bool)):
                    parts.extend(_lexical_tokens(sub_value))
    return parts


def _frame_relevance(utterance: str, toks: Sequence[str], frame: "CognitiveFrame", text_encoder: TextEncoder | None) -> float:
    return _route_relevance(utterance, toks, _frame_descriptor_tokens(frame), text_encoder)


def _frame_speech_plan(frame: "CognitiveFrame") -> list[str]:
    tokens: list[str] = []
    tokens.extend(_lexical_tokens(frame.intent))
    if frame.subject:
        tokens.extend(_lexical_tokens(frame.subject))
    predicate = (frame.evidence or {}).get("predicate") or (frame.evidence or {}).get("predicate_surface")
    if predicate:
        tokens.extend(_lexical_tokens(predicate))
    if frame.answer and frame.answer != "unknown":
        tokens.extend(_lexical_tokens(frame.answer))
    claimed = (frame.evidence or {}).get("claimed_answer")
    if claimed:
        tokens.extend(_lexical_tokens(claimed))
    if not tokens:
        tokens.extend(_lexical_tokens(frame.answer))
    return tokens + ["."]


@dataclass
class CognitiveFrame:
    """A non-linguistic content packet for the Broca interface to express.

    ``intent`` is an open vocabulary routing label; substrates may emit labels
    such as ``spatial_navigation`` or ``einstein_bio`` without changing feature shape.

    ``to_features()`` maps arbitrary intent/subject/answer strings through frozen subword sketches;
    ``speech_plan()`` accepts ``evidence[\"speech_plan_words\"]`` overrides and otherwise
    lexicalizes the frame's own intent/subject/predicate/answer fields.
    """

    intent: str
    subject: str = ""
    answer: str = "unknown"
    confidence: float = 1.0
    evidence: dict = field(default_factory=dict)

    def speech_plan(self) -> list[str]:
        raw_override = self.evidence.get("speech_plan_words")
        if isinstance(raw_override, list) and raw_override and all(isinstance(x, str) for x in raw_override):
            return list(raw_override)

        return _frame_speech_plan(self)

    def to_features(self, text_encoder: TextEncoder | None = None) -> torch.Tensor:
        """Semantic subword bottleneck over intent/subject/answer + numeric faculty scalars."""

        return pack_cognitive_frame(self.intent, self.subject, self.answer, float(self.confidence), self.evidence, text_encoder=text_encoder)


class PersistentSemanticMemory:
    """SQLite-backed symbolic/semantic memory for the cognitive substrate.

    This is deliberately separate from prompt context. The language module asks
    the substrate for a memory result; it does not receive a pasted fact list.
    """

    def __init__(self, path: str | Path, *, namespace: str = "main"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.namespace = namespace
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path)
        con.execute("PRAGMA journal_mode=WAL")
        return con

    def _init_schema(self) -> None:
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS semantic_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    namespace TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    evidence_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    UNIQUE(namespace, subject, predicate)
                )
                """
            )
            con.execute("CREATE INDEX IF NOT EXISTS idx_semantic_lookup ON semantic_memory(namespace, subject, predicate)")
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS semantic_claims (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    namespace TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    status TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            con.execute("CREATE INDEX IF NOT EXISTS idx_claim_lookup ON semantic_claims(namespace, subject, predicate, status)")
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_reflections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    namespace TEXT NOT NULL,
                    dedupe_key TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    UNIQUE(namespace, dedupe_key)
                )
                """
            )
            con.execute("CREATE INDEX IF NOT EXISTS idx_reflection_lookup ON memory_reflections(namespace, kind, subject, predicate)")

    def upsert(
        self,
        subject: str,
        predicate: str,
        obj: str,
        *,
        confidence: float = 1.0,
        evidence: dict | None = None,
        con: sqlite3.Connection | None = None,
    ) -> None:
        now = time.time()
        row = (
            self.namespace,
            subject.lower(),
            predicate.lower(),
            obj.lower(),
            float(confidence),
            json.dumps(evidence or {}, sort_keys=True),
            now,
            now,
        )
        if con is not None:
            con.execute(
                """
                INSERT INTO semantic_memory(namespace, subject, predicate, object, confidence, evidence_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(namespace, subject, predicate)
                DO UPDATE SET object=excluded.object, confidence=excluded.confidence,
                              evidence_json=excluded.evidence_json, updated_at=excluded.updated_at
                """,
                row,
            )
            logger.debug("PersistentSemanticMemory.upsert: ns=%s %s.%s -> %s conf=%s", self.namespace, subject.lower(), predicate.lower(), obj.lower(), confidence)
            return
        with self._connect() as c:
            c.execute(
                """
                INSERT INTO semantic_memory(namespace, subject, predicate, object, confidence, evidence_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(namespace, subject, predicate)
                DO UPDATE SET object=excluded.object, confidence=excluded.confidence,
                              evidence_json=excluded.evidence_json, updated_at=excluded.updated_at
                """,
                row,
            )
            logger.debug("PersistentSemanticMemory.upsert: ns=%s %s.%s -> %s conf=%s", self.namespace, subject.lower(), predicate.lower(), obj.lower(), confidence)

    def record_claim(
        self,
        subject: str,
        predicate: str,
        obj: str,
        *,
        confidence: float = 1.0,
        status: str = "observed",
        evidence: dict | None = None,
    ) -> int:
        now = time.time()
        with self._connect() as con:
            cur = con.execute(
                """
                INSERT INTO semantic_claims(namespace, subject, predicate, object, confidence, status, evidence_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.namespace,
                    subject.lower(),
                    predicate.lower(),
                    obj.lower(),
                    float(confidence),
                    str(status),
                    json.dumps(evidence or {}, sort_keys=True),
                    now,
                ),
            )
            cid = int(cur.lastrowid)
            logger.debug("PersistentSemanticMemory.record_claim: id=%s %s.%s=%s status=%s", cid, subject.lower(), predicate.lower(), obj.lower(), status)
            return cid

    def claims(self, subject: str | None = None, predicate: str | None = None, *, status: str | None = None) -> list[dict]:
        clauses = ["namespace=?"]
        args: list[Any] = [self.namespace]
        if subject is not None:
            clauses.append("subject=?")
            args.append(subject.lower())
        if predicate is not None:
            clauses.append("predicate=?")
            args.append(predicate.lower())
        if status is not None:
            clauses.append("status=?")
            args.append(status)
        sql = (
            "SELECT id, subject, predicate, object, confidence, status, evidence_json, created_at "
            f"FROM semantic_claims WHERE {' AND '.join(clauses)} ORDER BY id"
        )
        with self._connect() as con:
            rows = con.execute(sql, args).fetchall()
        return [
            {
                "id": int(r[0]),
                "subject": str(r[1]),
                "predicate": str(r[2]),
                "object": str(r[3]),
                "confidence": float(r[4]),
                "status": str(r[5]),
                "evidence": json.loads(r[6]),
                "created_at": float(r[7]),
            }
            for r in rows
        ]

    def record_reflection(
        self,
        kind: str,
        subject: str,
        predicate: str,
        summary: str,
        evidence: dict,
        *,
        dedupe_key: str,
        con: sqlite3.Connection | None = None,
    ) -> int | None:
        now = time.time()
        params = (
            self.namespace,
            dedupe_key,
            kind,
            subject.lower(),
            predicate.lower(),
            summary,
            json.dumps(evidence, sort_keys=True),
            now,
        )
        if con is not None:
            cur = con.execute(
                """
                INSERT OR IGNORE INTO memory_reflections(namespace, dedupe_key, kind, subject, predicate, summary, evidence_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                params,
            )
            if cur.rowcount == 0:
                return None
            return int(cur.lastrowid)
        with self._connect() as c:
            cur = c.execute(
                """
                INSERT OR IGNORE INTO memory_reflections(namespace, dedupe_key, kind, subject, predicate, summary, evidence_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                params,
            )
            if cur.rowcount == 0:
                return None
            return int(cur.lastrowid)

    def reflections(self, *, kind: str | None = None) -> list[dict]:
        clauses = ["namespace=?"]
        args: list[Any] = [self.namespace]
        if kind is not None:
            clauses.append("kind=?")
            args.append(kind)
        sql = (
            "SELECT id, dedupe_key, kind, subject, predicate, summary, evidence_json, created_at "
            f"FROM memory_reflections WHERE {' AND '.join(clauses)} ORDER BY id"
        )
        with self._connect() as con:
            rows = con.execute(sql, args).fetchall()
        return [
            {
                "id": int(r[0]),
                "dedupe_key": str(r[1]),
                "kind": str(r[2]),
                "subject": str(r[3]),
                "predicate": str(r[4]),
                "summary": str(r[5]),
                "evidence": json.loads(r[6]),
                "created_at": float(r[7]),
            }
            for r in rows
        ]

    def consolidate_claims_once(
        self,
        *,
        log_odds_threshold: float = BELIEF_REVISION_LOG_ODDS_THRESHOLD,
        min_claims: int = BELIEF_REVISION_MIN_CLAIMS,
    ) -> list[dict]:
        claims = self.claims()
        grouped: dict[tuple[str, str], list[dict]] = {}
        for claim in claims:
            grouped.setdefault((claim["subject"], claim["predicate"]), []).append(claim)

        gap_stats = _gap_population_stats(claims)
        reflections: list[dict] = []
        for (subject, predicate), rows in grouped.items():
            if len({r["object"] for r in rows}) < 2:
                continue
            support: dict[str, dict[str, Any]] = {}
            for row in rows:
                entry = support.setdefault(row["object"], {"score": 0.0, "count": 0, "claim_ids": [], "trust_weights": []})
                trust = _claim_trust_weight(row, gap_stats=gap_stats)
                entry["score"] += float(row["confidence"]) * trust
                entry["count"] += 1
                entry["claim_ids"].append(int(row["id"]))
                entry["trust_weights"].append(float(trust))

            current = self.get(subject, predicate)
            current_obj = current[0] if current is not None else ""
            current_score = float(support.get(current_obj, {}).get("score", 0.0))
            best_obj, best = max(support.items(), key=lambda item: (float(item[1]["score"]), int(item[1]["count"])))
            best_score = float(best["score"])
            best_count = int(best["count"])
            # Log-odds of the candidate vs. the current belief, in nats. With
            # adversarial high-surprise claims the candidate's score collapses
            # under the EMA Z-score Bayes factor, so the log-odds stay
            # negative; with low-surprise corroborating evidence the candidate
            # accumulates above the threshold.
            log_odds = math.log(max(best_score, 1e-12)) - math.log(max(current_score, 1e-12))
            evidence = {
                "support": support,
                "current_object": current_obj,
                "candidate_object": best_obj,
                "log_odds": float(log_odds),
                "log_odds_threshold": float(log_odds_threshold),
                "min_claims": int(min_claims),
                "gap_stats": (
                    {"mu": float(gap_stats[0]), "sigma": float(gap_stats[1])} if gap_stats else None
                ),
                "instrument": "background_claim_consolidation",
            }

            if (
                current_obj
                and best_obj != current_obj
                and best_count >= int(min_claims)
                and log_odds >= float(log_odds_threshold)
            ):
                claim_ids_digest = hashlib.sha256(
                    json.dumps(sorted(int(i) for i in best["claim_ids"]), separators=(",", ":")).encode()
                ).hexdigest()
                dedupe = f"belief_revision:{subject}:{predicate}:{current_obj}->{best_obj}:{claim_ids_digest}"
                with self._connect() as con:
                    con.execute("BEGIN")
                    try:
                        reflection_id = self.record_reflection(
                            "belief_revision",
                            subject,
                            predicate,
                            f"revised {subject}.{predicate} from {current_obj} to {best_obj}",
                            evidence,
                            dedupe_key=dedupe,
                            con=con,
                        )
                        if reflection_id is not None:
                            self.upsert(
                                subject,
                                predicate,
                                best_obj,
                                confidence=min(1.0, best_score / max(1.0, sum(float(v["score"]) for v in support.values()))),
                                evidence={**evidence, "reflection_id": reflection_id},
                                con=con,
                            )
                            reflections.append({"id": reflection_id, "kind": "belief_revision", **evidence})
                            logger.debug(
                                "consolidate_claims_once: belief_revision reflection_id=%s %s.%s %s -> %s",
                                reflection_id,
                                subject,
                                predicate,
                                current_obj,
                                best_obj,
                            )
                        con.commit()
                    except Exception:
                        con.rollback()
                        raise
            else:
                dedupe = f"belief_conflict:{subject}:{predicate}:{','.join(str(r['id']) for r in rows)}"
                reflection_id = self.record_reflection(
                    "belief_conflict",
                    subject,
                    predicate,
                    f"unresolved conflict over {subject}.{predicate}",
                    evidence,
                    dedupe_key=dedupe,
                )
                if reflection_id is not None:
                    reflections.append({"id": reflection_id, "kind": "belief_conflict", **evidence})
                    logger.debug(
                        "consolidate_claims_once: belief_conflict reflection_id=%s %s.%s (unresolved)",
                        reflection_id,
                        subject,
                        predicate,
                    )
        logger.debug("consolidate_claims_once: reflections_emitted=%d", len(reflections))
        return reflections

    def observe_claim(self, subject: str, predicate: str, obj: str, *, confidence: float = 1.0, evidence: dict | None = None) -> dict:
        subj = subject.lower()
        pred = predicate.lower()
        observed_obj = obj.lower()
        ev = dict(evidence or {})
        current = self.get(subj, pred)
        if current is None:
            claim_id = self.record_claim(subj, pred, observed_obj, confidence=confidence, status="accepted", evidence=ev)
            self.upsert(
                subj,
                pred,
                observed_obj,
                confidence=confidence,
                evidence={**ev, "claim_id": claim_id, "claim_status": "accepted"},
            )
            logger.debug(
                "PersistentSemanticMemory.observe_claim: accepted new triple %s.%s=%s claim_id=%s",
                subj,
                pred,
                observed_obj,
                claim_id,
            )
            return {"status": "accepted", "claim_id": claim_id, "current_object": observed_obj, "observed_object": observed_obj}

        current_obj, current_conf, current_ev = current
        if current_obj == observed_obj:
            claim_id = self.record_claim(subj, pred, observed_obj, confidence=confidence, status="corroborated", evidence=ev)
            merged_ev = merge_epistemic_evidence_dict(dict(current_ev), {**ev, "claim_id": claim_id, "claim_status": "corroborated"})
            self.upsert(subj, pred, observed_obj, confidence=max(float(current_conf), float(confidence)), evidence=merged_ev)
            logger.debug(
                "PersistentSemanticMemory.observe_claim: corroborated %s.%s=%s claim_id=%s",
                subj,
                pred,
                observed_obj,
                claim_id,
            )
            return {"status": "corroborated", "claim_id": claim_id, "current_object": current_obj, "observed_object": observed_obj}

        conflict_ev = {
            **ev,
            "conflicts_with": {
                "subject": subj,
                "predicate": pred,
                "object": current_obj,
                "confidence": current_conf,
                "evidence": current_ev,
            },
            "counterfactual": {
                "if_accepted": {"subject": subj, "predicate": pred, "object": observed_obj},
                "would_change_answer_from": current_obj,
                "would_change_answer_to": observed_obj,
            },
        }
        claim_id = self.record_claim(subj, pred, observed_obj, confidence=confidence, status="conflict", evidence=conflict_ev)
        logger.debug(
            "PersistentSemanticMemory.observe_claim: conflict %s.%s observed=%s current=%s claim_id=%s",
            subj,
            pred,
            observed_obj,
            current_obj,
            claim_id,
        )
        return {
            "status": "conflict",
            "claim_id": claim_id,
            "current_object": current_obj,
            "current_confidence": current_conf,
            "observed_object": observed_obj,
            "counterfactual": conflict_ev["counterfactual"],
        }

    def get(self, subject: str, predicate: str) -> tuple[str, float, dict] | None:
        with self._connect() as con:
            row = con.execute(
                "SELECT object, confidence, evidence_json FROM semantic_memory WHERE namespace=? AND subject=? AND predicate=?",
                (self.namespace, subject.lower(), predicate.lower()),
            ).fetchone()
        if row is None:
            return None
        return str(row[0]), float(row[1]), json.loads(row[2])

    def subjects_for_predicate(self, predicate: str) -> list[str]:
        """All subjects with this predicate — drives intrinsic cues without a fixed ENTITY list."""

        pred = predicate.lower()
        with self._connect() as con:
            rows = con.execute(
                "SELECT DISTINCT subject FROM semantic_memory WHERE namespace=? AND predicate=? ORDER BY subject",
                (self.namespace, pred),
            ).fetchall()
        return [str(r[0]) for r in rows]

    def subjects(self) -> list[str]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT DISTINCT subject FROM semantic_memory WHERE namespace=? ORDER BY subject",
                (self.namespace,),
            ).fetchall()
        return [str(r[0]) for r in rows]

    def records_for_subject(self, subject: str) -> list[tuple[str, str, float, dict]]:
        with self._connect() as con:
            rows = con.execute(
                """
                SELECT predicate, object, confidence, evidence_json
                FROM semantic_memory
                WHERE namespace=? AND subject=?
                ORDER BY updated_at DESC, id DESC
                """,
                (self.namespace, subject.lower()),
            ).fetchall()
        return [(str(r[0]), str(r[1]), float(r[2]), json.loads(r[3])) for r in rows]

    def distinct_objects_for_predicate(self, predicate: str) -> frozenset[str]:
        """Known objects already stored for a predicate; used for conflict detection."""

        pred = predicate.lower()
        with self._connect() as con:
            rows = con.execute(
                "SELECT DISTINCT object FROM semantic_memory WHERE namespace=? AND predicate=?",
                (self.namespace, pred),
            ).fetchall()
        return frozenset(str(r[0]).lower() for r in rows)

    def count(self) -> int:
        with self._connect() as con:
            row = con.execute("SELECT COUNT(*) FROM semantic_memory WHERE namespace=?", (self.namespace,)).fetchone()
        return int(row[0])

    def mean_confidence(self) -> float | None:
        with self._connect() as con:
            row = con.execute(
                "SELECT AVG(confidence), COUNT(*) FROM semantic_memory WHERE namespace=?",
                (self.namespace,),
            ).fetchone()
        if row is None or int(row[1]) == 0:
            return None
        return float(row[0])

    def merge_epistemic_evidence(self, subject: str, predicate: str, incoming: dict) -> None:
        got = self.get(subject, predicate)
        if got is None:
            return
        obj, conf, ev_old = got
        merged_ev = merge_epistemic_evidence_dict(dict(ev_old), incoming)
        self.upsert(subject, predicate, obj, confidence=conf, evidence=merged_ev)

    def all_facts(self) -> list[tuple[str, str, str, float, dict]]:
        """Every (subject, predicate, object, confidence, evidence) row in this namespace."""

        with self._connect() as con:
            rows = con.execute(
                """
                SELECT subject, predicate, object, confidence, evidence_json
                FROM semantic_memory WHERE namespace=?
                """,
                (self.namespace,),
            ).fetchall()
        return [(str(r[0]), str(r[1]), str(r[2]), float(r[3]), json.loads(r[4])) for r in rows]

    def boost_confidence(self, subject: str, predicate: str, *, factor: float, cap: float = 1.0, reason: str | None = None) -> tuple[str, float, float] | None:
        """Multiplicatively raise the confidence of a stored fact (DMN reinforcement).

        Returns ``(object, old_conf, new_conf)`` so callers can log the move,
        or ``None`` if the row doesn't exist. Evidence is updated with a
        ``dmn_consolidation`` note recording the boost factor and reason so
        the provenance trail survives across runs.
        """

        got = self.get(subject, predicate)
        if got is None:
            return None
        obj, conf, ev_old = got
        new_conf = float(min(float(cap), max(0.0, conf * float(factor))))
        ev_new = dict(ev_old)
        notes = list(ev_new.get("dmn_consolidation") or [])
        notes.append({
            "ts": time.time(),
            "factor": float(factor),
            "old_confidence": float(conf),
            "new_confidence": float(new_conf),
            "reason": str(reason or "centrality_boost"),
        })
        ev_new["dmn_consolidation"] = notes[-32:]
        self.upsert(subject, predicate, obj, confidence=new_conf, evidence=ev_new)
        return obj, float(conf), float(new_conf)

    def overlapping_subject_pairs(self, *, min_shared: int = 2) -> list[dict]:
        """Pairs of distinct subjects that share ``>= min_shared`` (predicate, object) tuples.

        Used by the DMN's separation phase to flag entities the substrate
        cannot discriminate from observation alone — high overlap means
        Fristonian ambiguity is high (observations don't separate the two
        hypotheses), so the substrate raises an intrinsic cue asking the LLM
        to pose a disambiguating question on the next turn.
        """

        max_bucket = 256
        threshold = max(1, int(min_shared))
        # Group subjects by the (predicate, object) tuples they share.
        bucket: dict[tuple[str, str], list[str]] = {}
        with self._connect() as con:
            rows = con.execute(
                """
                SELECT subject, predicate, object FROM semantic_memory WHERE namespace=?
                """,
                (self.namespace,),
            ).fetchall()
        per_subject: dict[str, set[tuple[str, str]]] = {}
        for subj, pred, obj in rows:
            subj = str(subj)
            key = (str(pred), str(obj))
            bucket.setdefault(key, []).append(subj)
            per_subject.setdefault(subj, set()).add(key)
        pair_overlap: dict[tuple[str, str], set[tuple[str, str]]] = {}
        for key, subjects in bucket.items():
            if len(subjects) < 2:
                continue
            unique = sorted(set(subjects))
            if len(unique) > max_bucket:
                logger.info(
                    "PersistentSemanticMemory.overlapping_subject_pairs: capped bucket key=%r n_unique=%d max=%d",
                    key,
                    len(unique),
                    max_bucket,
                )
                unique = unique[:max_bucket]
            for i in range(len(unique)):
                for j in range(i + 1, len(unique)):
                    pair_overlap.setdefault((unique[i], unique[j]), set()).add(key)
        out: list[dict] = []
        for (a, b), shared in pair_overlap.items():
            if len(shared) < threshold:
                continue
            size_a = len(per_subject.get(a, set()))
            size_b = len(per_subject.get(b, set()))
            denom = max(1, min(size_a, size_b))
            ratio = len(shared) / float(denom)
            out.append(
                {
                    "subject_a": a,
                    "subject_b": b,
                    "shared": sorted(shared),
                    "shared_count": len(shared),
                    "size_a": size_a,
                    "size_b": size_b,
                    "overlap_ratio": float(ratio),
                }
            )
        out.sort(key=lambda d: (-d["overlap_ratio"], -d["shared_count"], d["subject_a"], d["subject_b"]))
        return out


class WorkspaceJournal:
    """Episodic log of workspace frames paired with raw utterances (same SQLite file as semantic memory)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path)
        con.execute("PRAGMA journal_mode=WAL")
        return con

    def _init_schema(self) -> None:
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS workspace_journal (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    utterance TEXT NOT NULL,
                    intent TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    evidence_json TEXT NOT NULL
                )
                """
            )

    def append(self, utterance: str, frame: CognitiveFrame) -> int:
        now = time.time()
        with self._connect() as con:
            cur = con.execute(
                """
                INSERT INTO workspace_journal(ts, utterance, intent, subject, answer, confidence, evidence_json)
                VALUES (?,?,?,?,?,?,?)
                """,
                (
                    now,
                    utterance,
                    frame.intent,
                    frame.subject,
                    frame.answer,
                    float(frame.confidence),
                    json.dumps(frame.evidence or {}, sort_keys=True),
                ),
            )
            jid = int(cur.lastrowid)
            logger.debug(
                "WorkspaceJournal.append: id=%s intent=%s subject=%r answer=%r utterance_preview=%r",
                jid,
                frame.intent,
                frame.subject,
                frame.answer,
                (utterance[:120] + "…") if len(utterance) > 120 else utterance,
            )
            return jid

    def fetch(self, episode_id: int) -> dict | None:
        with self._connect() as con:
            row = con.execute(
                """
                SELECT id, ts, utterance, intent, subject, answer, confidence, evidence_json
                FROM workspace_journal WHERE id=?
                """,
                (int(episode_id),),
            ).fetchone()
        if row is None:
            return None
        return {
            "id": int(row[0]),
            "ts": float(row[1]),
            "utterance": str(row[2]),
            "intent": str(row[3]),
            "subject": str(row[4]),
            "answer": str(row[5]),
            "confidence": float(row[6]),
            "evidence": json.loads(row[7]),
        }

    def recent(self, limit: int) -> list[dict]:
        lim = max(1, int(limit))
        with self._connect() as con:
            rows = con.execute(
                """
                SELECT id, ts, utterance, intent, subject, answer, confidence, evidence_json
                FROM workspace_journal ORDER BY id DESC LIMIT ?
                """,
                (lim,),
            ).fetchall()
        out: list[dict] = []
        for row in reversed(rows):
            out.append(
                {
                    "id": int(row[0]),
                    "ts": float(row[1]),
                    "utterance": str(row[2]),
                    "intent": str(row[3]),
                    "subject": str(row[4]),
                    "answer": str(row[5]),
                    "confidence": float(row[6]),
                    "evidence": json.loads(row[7]),
                }
            )
        return out

    def count(self) -> int:
        with self._connect() as con:
            row = con.execute("SELECT COUNT(*) FROM workspace_journal").fetchone()
        return int(row[0])


def cognitive_frame_from_episode_row(row: dict) -> CognitiveFrame:
    ev = dict(row["evidence"])
    ev["retrieved_episode_id"] = row["id"]
    ev["episode_original_ts"] = row["ts"]
    inst = list(ev.get("instruments") or [])
    if "episodic_retrieval" not in inst:
        inst.append("episodic_retrieval")
    ev["instruments"] = inst
    return CognitiveFrame(
        row["intent"],
        subject=row["subject"],
        answer=row["answer"],
        confidence=float(row["confidence"]),
        evidence=ev,
    )


def working_memory_synthesize(working: list[CognitiveFrame]) -> CognitiveFrame | None:
    """When working memory simultaneously carries memory and causal readouts, bind provenance."""

    if len(working) < 2:
        return None
    if any(f.intent == "synthesis_bundle" for f in working[-6:]):
        return None
    mem = None
    ce = None
    for f in reversed(working):
        if mem is None and f.intent.startswith("memory_") and f.intent not in {"memory_write", "memory_conflict"}:
            mem = f
        if ce is None and f.intent == "causal_effect":
            ce = f
        if mem is not None and ce is not None:
            break
    if mem is None or ce is None:
        return None
    jids: list[int] = []
    for f in (mem, ce):
        jid = (f.evidence or {}).get("journal_id")
        if jid is not None:
            jids.append(int(jid))
    geo = math.sqrt(max(1e-12, float(mem.confidence)) * max(1e-12, float(ce.confidence)))
    return CognitiveFrame(
        "synthesis_bundle",
        subject=mem.subject,
        answer=mem.answer,
        confidence=min(1.0, geo),
        evidence={
            "episode_ids": jids,
            "instruments": ["working_memory_synthesis", "semantic_memory", "scm_do_readout"],
            "predicate": (mem.evidence or {}).get("predicate", ""),
            "causal_ate": ce.evidence.get("ate"),
            "source_intents": [mem.intent, ce.intent],
        },
    )


@dataclass
class IntrinsicCue:
    urgency: float
    faculty: str
    evidence: dict = field(default_factory=dict)


class GlobalWorkspace:
    """A tiny blackboard for non-language faculty frames."""

    def __init__(self):
        self.frames: list[CognitiveFrame] = []
        self.intrinsic_cues: list[IntrinsicCue] = []
        self.working: list[CognitiveFrame] = []

    def _trim_working(self) -> None:
        cap = max(8, int(math.ceil(math.sqrt(max(1, len(self.frames)))) * 4))
        self.working = self.frames[-cap:]

    def publish(self, frame: CognitiveFrame) -> CognitiveFrame:
        self.frames.append(frame)
        self._trim_working()
        syn = working_memory_synthesize(self.working)
        if syn is not None:
            logger.debug("GlobalWorkspace.publish: synthesized intent=%s from working tail", syn.intent)
            self.frames.append(syn)
            self._trim_working()
        logger.debug("GlobalWorkspace.publish: intent=%s journal_id=%s frames_total=%d", frame.intent, (frame.evidence or {}).get("journal_id"), len(self.frames))
        return frame

    @property
    def latest(self) -> CognitiveFrame | None:
        return self.frames[-1] if self.frames else None

    def snapshot(self) -> list[dict]:
        return [asdict(f) for f in self.frames]


@dataclass
class DMNConfig:
    """Tunable thresholds for the Default Mode Network's three idle phases.

    These knobs are deliberately exposed: the DMN's behavior is a function of
    how aggressively the user wants the substrate to forget, disambiguate, and
    speculate during idle cycles. All have safe defaults; the chat CLI can
    override them with environment variables or constructor arguments.
    """

    # Phase 1 — consolidation
    decay_gamma: float = 0.99
    decay_prune_below: float = 0.01
    centrality_iterations: int = 20
    centrality_min_weight: float = 0.0
    centrality_boost_floor: float = 0.05  # only boost facts whose central episode beats this PageRank mass
    centrality_boost_factor: float = 1.05
    centrality_boost_cap: float = 0.999

    # Phase 2 — separation
    overlap_min_shared: int = 2
    overlap_ratio_floor: float = 0.66
    overlap_max_cues: int = 4

    # Phase 3 — latent discovery
    dream_attempts_per_tick: int = 3
    dream_ate_insight_threshold: float = 0.4
    transitive_min_pair_weight: float = 0.5
    transitive_cosine_threshold: float = 0.55
    transitive_max_new_edges: int = 4

    # REM sleep — trigger when the user has been quiet this long.
    sleep_idle_seconds: float = 600.0
    sleep_max_replay: int = 32
    sleep_min_observations_for_pc: int = 24
    sleep_pc_alpha: float = 0.05
    sleep_hawkes_min_events: int = 6


class CognitiveBackgroundWorker:
    """Default Mode Network for the cognitive substrate.

    Runs three thermodynamic phases on every tick — even when the user has
    been silent — so the substrate physically reorganizes itself between
    turns:

      1. **Consolidation.** Episodic association edges decay multiplicatively
         and weak ones are pruned. A PageRank pass over the survivors finds
         the central episodes; the confidence of any semantic fact whose
         provenance cites a central episode is boosted, so frequently-used
         knowledge becomes harder to forget.
      2. **Separation.** Subjects that share suspiciously many predicate /
         object pairs are scored for Fristonian ambiguity (binary entropy of
         the disambiguation distribution) and emit an intrinsic cue so the
         next reply tends toward a clarifying question rather than committing
         to a coin-flip identity.
      3. **Latent discovery.** Two stochastic subroutines: the SCM is
         "dreamt" — random treatment / outcome pairs are selected, do(·)
         interventions are run, and large average treatment effects are
         persisted as ``latent_causal_insight`` reflections; and the episode
         graph is walked for transitive closure: a strongly-connected (A, B)
         and (B, C) trigger a cosine-similarity test on A and C's frames,
         creating a new (A, C) edge if the test fires.

    Each phase emits one or more reflection-shaped dicts so the chat CLI and
    debugger can replay what the DMN did between turns.
    """

    def __init__(
        self,
        mind: "BrocaMind",
        *,
        interval_s: float = 5.0,
        config: DMNConfig | None = None,
        rng: random.Random | None = None,
        motor_trainer: Any | None = None,
    ):
        self.mind = mind
        self.interval_s = max(0.1, float(interval_s))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.iterations = 0
        self.last_error: str | None = None
        self.config = config if config is not None else DMNConfig()
        self._rng = rng or random.Random(0xB0CA1)
        self.last_phase_summary: dict[str, dict[str, Any]] = {}
        self.last_user_activity_at: float = time.time()
        self.motor_trainer = motor_trainer
        self.last_rem_summary: dict[str, Any] = {}

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="broca-dmn", daemon=True)
        self._thread.start()
        logger.info("CognitiveBackgroundWorker.start: interval=%.3fs config=%s", self.interval_s, asdict(self.config))

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.0, float(timeout)))
        logger.info("CognitiveBackgroundWorker.stop: iterations=%d last_error=%s", self.iterations, self.last_error)

    def run_once(self) -> list[dict]:
        """Run one DMN tick: phases 1 → 2 → 3, then claim consolidation."""

        tick_started = time.time()
        reflections: list[dict] = []
        phase_summary: dict[str, dict[str, Any]] = {}

        for name, fn in (
            ("consolidation", self._phase1_consolidation),
            ("separation", self._phase2_separation),
            ("latent_discovery", self._phase3_latent_discovery),
            ("chunk_compilation", self._phase4_chunk_compilation),
            ("tool_foraging", self._phase5_tool_foraging),
        ):
            phase_started = time.time()
            try:
                phase_reflections, summary = fn()
            except Exception:
                logger.exception("DMN phase %s failed", name)
                phase_summary[name] = {"error": True}
                continue
            reflections.extend(phase_reflections)
            summary["duration_ms"] = int(round((time.time() - phase_started) * 1000))
            summary["reflections"] = len(phase_reflections)
            phase_summary[name] = summary
            logger.info("DMN.phase=%s %s", name, summary)

        # Reactive belief consolidation runs after the autonomous phases so
        # any DMN-promoted edges or boosted confidences are visible to it.
        try:
            claim_reflections = self.mind.consolidate_once()
        except Exception:
            logger.exception("DMN claim consolidation failed")
            claim_reflections = []
        reflections.extend(claim_reflections)
        phase_summary["claim_consolidation"] = {"reflections": len(claim_reflections)}

        # REM sleep — only if the user has been quiet long enough.
        idle = max(0.0, time.time() - self.last_user_activity_at)
        if idle >= self.config.sleep_idle_seconds:
            phase_started = time.time()
            try:
                rem_reflections, rem_summary = self._rem_sleep()
            except Exception:
                logger.exception("DMN REM sleep failed")
                rem_summary = {"error": True}
                rem_reflections = []
            rem_summary["duration_ms"] = int(round((time.time() - phase_started) * 1000))
            rem_summary["idle_seconds"] = float(idle)
            rem_summary["reflections"] = len(rem_reflections)
            phase_summary["rem_sleep"] = rem_summary
            self.last_rem_summary = rem_summary
            reflections.extend(rem_reflections)
            logger.info("DMN.phase=rem_sleep %s", rem_summary)

        self.iterations += 1
        self.last_error = None
        self.last_phase_summary = phase_summary
        logger.debug(
            "CognitiveBackgroundWorker.run_once: iteration=%d total_reflections=%d duration_ms=%d idle=%.1fs",
            self.iterations,
            len(reflections),
            int(round((time.time() - tick_started) * 1000)),
            idle,
        )
        return reflections

    def mark_user_active(self) -> None:
        """Reset the idle clock when the user types something."""

        self.last_user_activity_at = time.time()

    # ------------------------------------------------------------------ Phase 1

    def _phase1_consolidation(self) -> tuple[list[dict], dict[str, Any]]:
        cfg = self.config
        graph = self.mind.episode_graph
        memory = self.mind.memory

        decayed, pruned = graph.decay_all(gamma=cfg.decay_gamma, prune_below=cfg.decay_prune_below)
        centrality = graph.centrality(
            damping=0.85,
            iterations=cfg.centrality_iterations,
            min_weight=cfg.centrality_min_weight,
        )

        boosts: list[dict[str, Any]] = []
        if centrality:
            top_episodes = {ep for ep, score in centrality.items() if score >= cfg.centrality_boost_floor}
            if top_episodes:
                # Boost any fact whose recorded episode_ids intersect the central set.
                for subj, pred, _obj, _conf, evidence in memory.all_facts():
                    eps = evidence.get("episode_ids") if isinstance(evidence, dict) else None
                    if not isinstance(eps, list):
                        continue
                    parsed: list[int] = []
                    for e in eps:
                        try:
                            ei = int(e)
                        except (TypeError, ValueError):
                            continue
                        parsed.append(ei)
                    intersection = [ei for ei in parsed if ei in top_episodes]
                    if not intersection:
                        continue
                    # Scale the boost by the maximum centrality mass that touched this fact.
                    mass = max(centrality.get(ei, 0.0) for ei in intersection)
                    factor = 1.0 + (cfg.centrality_boost_factor - 1.0) * min(1.0, mass / max(cfg.centrality_boost_floor, 1e-6))
                    result = memory.boost_confidence(
                        subj,
                        pred,
                        factor=factor,
                        cap=cfg.centrality_boost_cap,
                        reason=f"pagerank_central(mass={mass:.4f})",
                    )
                    if result is None:
                        continue
                    obj, old_conf, new_conf = result
                    boosts.append({
                        "subject": subj,
                        "predicate": pred,
                        "object": obj,
                        "old_confidence": old_conf,
                        "new_confidence": new_conf,
                        "factor": float(factor),
                        "central_episode_mass": float(mass),
                        "central_episodes": intersection,
                    })
                    logger.debug(
                        "DMN.phase1.boost: subj=%r pred=%r %.4f -> %.4f factor=%.4f mass=%.4f episodes=%s",
                        subj,
                        pred,
                        old_conf,
                        new_conf,
                        float(factor),
                        float(mass),
                        intersection,
                    )

        reflections: list[dict] = []
        if boosts:
            reflections.append({
                "kind": "consolidation_boost",
                "boosts": boosts,
            })
        summary = {
            "decayed_edges": int(decayed),
            "pruned_edges": int(pruned),
            "central_nodes": len(centrality),
            "boosted_facts": len(boosts),
        }
        return reflections, summary

    # ------------------------------------------------------------------ Phase 2

    def _phase2_separation(self) -> tuple[list[dict], dict[str, Any]]:
        cfg = self.config
        memory = self.mind.memory
        # Clear any prior DMN-flagged ambiguity cues so we don't accumulate stale ones across ticks.
        ws = self.mind.workspace
        ws.intrinsic_cues = [c for c in ws.intrinsic_cues if c.faculty != "entity_ambiguity"]

        pairs = memory.overlapping_subject_pairs(min_shared=cfg.overlap_min_shared)
        emitted: list[dict[str, Any]] = []
        for pair in pairs[: max(0, cfg.overlap_max_cues)]:
            ratio = float(pair["overlap_ratio"])
            if ratio < cfg.overlap_ratio_floor:
                continue
            # Fristonian ambiguity ≈ binary entropy of the maximum-entropy
            # disambiguation distribution under the observed overlap. Total
            # overlap → 50/50 hypothesis posterior → ambiguity = log(2).
            p = 0.5 + 0.5 * (1.0 - ratio)  # slide from 0.5 (full overlap) toward 1.0 (none)
            p = max(1e-6, min(1 - 1e-6, p))
            ambiguity = -(p * math.log(p) + (1 - p) * math.log(1 - p))
            urgency = float(min(1.0, ambiguity / math.log(2)))
            cue_evidence = {
                "subject_a": pair["subject_a"],
                "subject_b": pair["subject_b"],
                "shared_count": pair["shared_count"],
                "overlap_ratio": ratio,
                "ambiguity_nats": float(ambiguity),
                "shared_predicates": [list(t) for t in pair["shared"]],
            }
            ws.intrinsic_cues.append(IntrinsicCue(urgency=urgency, faculty="entity_ambiguity", evidence=cue_evidence))
            emitted.append(cue_evidence | {"urgency": urgency})
            logger.info(
                "DMN.phase2.cue: %r ↔ %r ratio=%.3f ambiguity=%.4f nats urgency=%.3f",
                pair["subject_a"],
                pair["subject_b"],
                ratio,
                ambiguity,
                urgency,
            )

        reflections: list[dict] = []
        if emitted:
            reflections.append({"kind": "separation_cue", "cues": emitted})

        summary = {
            "candidate_pairs": len(pairs),
            "cues_emitted": len(emitted),
        }
        return reflections, summary

    # ------------------------------------------------------------------ Phase 3

    def _phase3_latent_discovery(self) -> tuple[list[dict], dict[str, Any]]:
        reflections: list[dict] = []
        causal = self._causal_dreaming()
        reflections.extend(causal["reflections"])
        transitive = self._transitive_episode_closure()
        reflections.extend(transitive["reflections"])
        summary = {
            "causal_attempts": causal["attempts"],
            "causal_insights": causal["insights"],
            "transitive_pairs_examined": transitive["pairs_examined"],
            "transitive_edges_added": transitive["edges_added"],
        }
        return reflections, summary

    def _causal_dreaming(self) -> dict[str, Any]:
        cfg = self.config
        scm = getattr(self.mind, "scm", None)
        if scm is None:
            return {"reflections": [], "attempts": 0, "insights": 0}
        endogenous = list(scm.endogenous_names)
        if len(endogenous) < 2:
            return {"reflections": [], "attempts": 0, "insights": 0}

        attempts = 0
        insights: list[dict[str, Any]] = []
        for _ in range(max(0, int(cfg.dream_attempts_per_tick))):
            attempts += 1
            treatment, outcome = self._rng.sample(endogenous, 2)
            try:
                t_dom = scm.domains.get(treatment)
                o_dom = scm.domains.get(outcome)
                if not t_dom or not o_dom or len(t_dom) < 2 or len(o_dom) < 2:
                    continue
                t_pos, t_neg = t_dom[0], t_dom[1]
                outcome_value = o_dom[0]
                p_pos = scm.probability({outcome: outcome_value}, interventions={treatment: t_pos})
                p_neg = scm.probability({outcome: outcome_value}, interventions={treatment: t_neg})
            except (KeyError, ValueError, RuntimeError):
                logger.debug("DMN.phase3.dream: failed treatment=%s outcome=%s", treatment, outcome, exc_info=True)
                continue
            ate = float(p_pos - p_neg)
            logger.debug(
                "DMN.phase3.dream: do(%s=%s)→P(%s=%s)=%.4f vs do(%s=%s)→%.4f ate=%.4f",
                treatment,
                t_pos,
                outcome,
                outcome_value,
                p_pos,
                treatment,
                t_neg,
                p_neg,
                ate,
            )
            if abs(ate) < cfg.dream_ate_insight_threshold:
                continue
            relation_label = scm.labels.get("positive_effect" if ate >= 0 else "negative_effect")
            relation = relation_label or ("causes_increase" if ate >= 0 else "causes_decrease")
            evidence = {
                "treatment": treatment,
                "outcome": outcome,
                "outcome_value": outcome_value,
                "treatment_values": [t_pos, t_neg],
                "p_do_positive": float(p_pos),
                "p_do_negative": float(p_neg),
                "ate": ate,
                "instrument": "dmn_causal_dream",
            }
            dedupe = f"latent_causal_insight:{treatment}->{outcome}:{relation}"
            reflection_id = self.mind.memory.record_reflection(
                "latent_causal_insight",
                treatment,
                relation,
                f"dreamt that intervening on {treatment} {relation} {outcome} (ATE={ate:+.2f})",
                evidence,
                dedupe_key=dedupe,
            )
            if reflection_id is None:
                continue
            insights.append({"id": reflection_id, "kind": "latent_causal_insight", **evidence})
            logger.info(
                "DMN.phase3.dream.insight: id=%d %s %s %s ate=%+.3f",
                reflection_id,
                treatment,
                relation,
                outcome,
                ate,
            )

        return {"reflections": insights, "attempts": attempts, "insights": len(insights)}

    def _transitive_episode_closure(self) -> dict[str, Any]:
        cfg = self.config
        graph = self.mind.episode_graph
        edges = graph.edges(min_weight=cfg.transitive_min_pair_weight)
        if not edges:
            return {"reflections": [], "pairs_examined": 0, "edges_added": 0}

        # Index neighbors by node so we can spot A–B–C chains efficiently.
        neighbors: dict[int, list[tuple[int, float]]] = {}
        for lo, hi, w in edges:
            neighbors.setdefault(lo, []).append((hi, w))
            neighbors.setdefault(hi, []).append((lo, w))

        pairs_examined = 0
        added: list[dict[str, Any]] = []
        text_encoder = getattr(self.mind, "text_encoder", None)
        for hub, hub_edges in neighbors.items():
            if len(hub_edges) < 2 or len(added) >= cfg.transitive_max_new_edges:
                continue
            # Sample a pair of distinct neighbors; the random thermal kick is
            # what lets the system jump between local minima rather than
            # rediscovering the same closures every tick.
            sampled = self._rng.sample(hub_edges, k=min(len(hub_edges), 4))
            for i in range(len(sampled)):
                for j in range(i + 1, len(sampled)):
                    a, _w_a = sampled[i]
                    c, _w_c = sampled[j]
                    if a == c:
                        continue
                    pairs_examined += 1
                    if graph.weight(a, c) > 0.0:
                        continue
                    cosine = self._episode_frame_similarity(a, c, text_encoder=text_encoder)
                    logger.debug(
                        "DMN.phase3.transitive: hub=%d a=%d c=%d cosine=%s",
                        hub,
                        a,
                        c,
                        ("%.4f" % cosine) if cosine is not None else "n/a",
                    )
                    if cosine is None or cosine < cfg.transitive_cosine_threshold:
                        continue
                    delta = float(cosine)
                    graph.bump(a, c, delta=delta)
                    added.append({
                        "lo": min(a, c),
                        "hi": max(a, c),
                        "via_hub": hub,
                        "cosine": float(cosine),
                        "weight_added": delta,
                    })
                    logger.info("DMN.phase3.transitive.edge: %d↔%d via %d cosine=%.4f", a, c, hub, cosine)
                    if len(added) >= cfg.transitive_max_new_edges:
                        break
                if len(added) >= cfg.transitive_max_new_edges:
                    break

        reflections: list[dict] = []
        if added:
            reflections.append({"kind": "transitive_episode_closure", "edges": added})
        return {"reflections": reflections, "pairs_examined": pairs_examined, "edges_added": len(added)}

    def _episode_frame_similarity(self, a: int, b: int, *, text_encoder) -> float | None:
        row_a = self.mind.journal.fetch(a)
        row_b = self.mind.journal.fetch(b)
        if row_a is None or row_b is None:
            return None
        frame_a = cognitive_frame_from_episode_row(row_a)
        frame_b = cognitive_frame_from_episode_row(row_b)
        text_a = " ".join(_frame_descriptor_tokens(frame_a))
        text_b = " ".join(_frame_descriptor_tokens(frame_b))
        if not text_a.strip() or not text_b.strip():
            return None
        try:
            return float(_cosine(_text_vector(text_a, text_encoder), _text_vector(text_b, text_encoder)))
        except (RuntimeError, ValueError):
            logger.debug("DMN.phase3.transitive.similarity_failed a=%d b=%d", a, b, exc_info=True)
            return None

    # ------------------------------------------------------------------ Phase 4

    def _phase4_chunk_compilation(self) -> tuple[list[dict], dict[str, Any]]:
        """Detect repeated motifs in the workspace journal and compile them into macros.

        Implements the proceduralization side of the System-2 → System-1
        transition: every repeated reasoning trajectory becomes a single
        ``CompiledMacro`` whose mean feature vector is what the
        :class:`TrainableBrocaGraft` injects when the substrate next sees the
        macro's prefix.
        """

        compiler = getattr(self.mind, "chunking_compiler", None)
        if compiler is None:
            return [], {"compiled": 0, "candidates": 0, "scanned": 0}
        result = compiler.run_once()
        return list(result.get("reflections") or []), {
            "compiled": int(result.get("compiled", 0)),
            "candidates": int(result.get("candidates", 0)),
            "scanned": int(result.get("scanned", 0)),
        }

    # ------------------------------------------------------------------ Phase 5

    def _phase5_tool_foraging(self) -> tuple[list[dict], dict[str, Any]]:
        """Decide whether the substrate should synthesize a new native tool.

        We do **not** synthesize the tool here — that requires an LLM call to
        produce candidate Python source and is therefore an external,
        user-or-agent-driven step.  What we *do* run during DMN time is the
        active-inference math itself: when the unified faculty's posterior is
        confused (high entropy) and there are few existing tools, the EFE of
        ``synthesize_tool`` collapses below the alternatives, and we emit a
        ``tool_synthesis_recommended`` reflection so a downstream agent
        knows to act.
        """

        agent = getattr(self.mind, "tool_foraging_agent", None)
        unified = getattr(self.mind, "unified_agent", None)
        registry = getattr(self.mind, "tool_registry", None)
        if agent is None or unified is None or registry is None:
            return [], {"ran": False}

        try:
            coupled = unified.decide()
        except Exception:
            logger.exception("DMN.phase5.tool_foraging: unified_agent.decide failed")
            return [], {"ran": False, "error": True}

        if coupled.faculty == "spatial":
            posterior = list(coupled.spatial_decision.posterior_over_policies)
        else:
            posterior = list(coupled.causal_decision.posterior_over_policies)

        n = len(posterior)
        if n < 2:
            insufficient_prior = 0.5
        else:
            h = belief_entropy(posterior)
            h_max = math.log(n)
            insufficient_prior = max(1e-6, min(1 - 1e-6, h / max(h_max, 1e-9)))

        agent.update_belief(insufficient_prior=float(insufficient_prior))
        decision = agent.decide()
        recommended = decision.action_name == "synthesize_tool"

        reflections: list[dict] = []
        if recommended:
            evidence = {
                "action": decision.action_name,
                "insufficient_prior": float(insufficient_prior),
                "n_existing_tools": int(registry.count()),
                "policy_efe": [
                    {
                        "policy": list(int(a) for a in p.policy),
                        "expected_free_energy": float(p.expected_free_energy),
                    }
                    for p in decision.policies
                ],
                "instrument": "dmn_tool_foraging",
                "coupled_faculty": coupled.faculty,
            }
            reflection_id = self.mind.memory.record_reflection(
                "tool_synthesis_recommended",
                "tool_foraging",
                "synthesize_tool",
                f"EFE math recommends synthesizing a new tool (insufficient_prior={insufficient_prior:.3f})",
                evidence,
                dedupe_key=f"tool_synthesis_recommended:{int(time.time() // 60)}",
            )
            if reflection_id is not None:
                reflections.append({"id": reflection_id, "kind": "tool_synthesis_recommended", **evidence})
                logger.info(
                    "DMN.phase5.tool_foraging.recommend: id=%d insufficient_prior=%.3f n_tools=%d",
                    reflection_id,
                    insufficient_prior,
                    registry.count(),
                )

        summary = {
            "ran": True,
            "recommended": recommended,
            "insufficient_prior": float(insufficient_prior),
            "n_existing_tools": int(registry.count()),
            "chosen_action": decision.action_name,
        }
        return reflections, summary

    # ------------------------------------------------------------------ REM sleep

    def _rem_sleep(self) -> tuple[list[dict], dict[str, Any]]:
        """REM-style consolidation: motor learning + causal discovery + Hawkes refit.

        Runs only when the user has been idle long enough that a multi-second
        compute spike won't affect interactive latency. Each subroutine is
        wrapped so a failure in one doesn't block the others.
        """

        cfg = self.config
        summary: dict[str, Any] = {}
        reflections: list[dict] = []

        # 1. Motor learning — re-train the Broca grafts on recent journals.
        motor = {"ran": False}
        if self.motor_trainer is not None and getattr(self.mind, "motor_replay", None):
            replay = list(self.mind.motor_replay)[-cfg.sleep_max_replay :]
            try:
                step = self.motor_trainer.step(replay)
            except Exception:
                logger.exception("REM.motor: step failed")
                step = {"skipped": True, "reason": "exception"}
            motor.update(step)
            motor["ran"] = True
            if step.get("skipped") is False:
                reflections.append({"kind": "rem_motor_learning", **step})
        summary["motor"] = motor

        # 2. Hawkes refit — relearn excitation matrix from recent journal events.
        hawkes_summary: dict[str, Any] = {"ran": False}
        try:
            recent = self.mind.journal.recent(limit=128)
        except Exception:
            logger.exception("REM.hawkes: journal recent failed")
            recent = []
        events: list[tuple[str, float]] = []
        for row in recent:
            channel = str(row.get("intent", "") or "unknown")
            ts = float(row.get("ts", 0.0))
            events.append((channel, ts))
        if len(events) >= cfg.sleep_hawkes_min_events:
            channels = sorted({c for c, _ in events})
            try:
                mu, alpha = fit_excitation_em(events, channels, beta=self.mind.hawkes.beta)
            except Exception:
                logger.exception("REM.hawkes: EM fit failed")
                mu, alpha = None, None
            if mu is not None and alpha is not None:
                self.mind.hawkes.refit(channels, mu, alpha)
                try:
                    self.mind.hawkes_persistence.save(self.mind.hawkes)
                except Exception:
                    logger.exception("REM.hawkes: persistence save failed")
                hawkes_summary = {
                    "ran": True,
                    "channels": channels,
                    "events": len(events),
                    "mu_max": float(max(mu)) if mu else 0.0,
                    "alpha_norm": float(sum(sum(abs(x) for x in row) for row in alpha)),
                }
                reflections.append({"kind": "rem_hawkes_refit", **hawkes_summary})
        summary["hawkes"] = hawkes_summary

        # 3. Causal discovery — re-run PC on observation memory and rebuild SCM.
        cd_summary: dict[str, Any] = {"ran": False}
        try:
            observations = self._collect_observations_for_pc()
        except Exception:
            logger.exception("REM.causal_discovery: observation collection failed")
            observations = []
        if len(observations) >= cfg.sleep_min_observations_for_pc:
            try:
                graph = pc_algorithm(observations, alpha=cfg.sleep_pc_alpha)
                if graph.directed_edges or graph.undirected_edges:
                    new_scm = build_scm_from_skeleton(graph, observations)
                    self.mind.discovered_scm = new_scm
                    cd_summary = {
                        "ran": True,
                        "n_observations": len(observations),
                        "directed_edges": [list(e) for e in sorted(graph.directed_edges)],
                        "undirected_edges": [sorted(list(e)) for e in graph.undirected_edges],
                        "variables": list(graph.variables),
                    }
                    reflections.append({"kind": "rem_causal_discovery", **cd_summary})
            except Exception:
                logger.exception("REM.causal_discovery: PC algorithm failed")
        summary["causal_discovery"] = cd_summary

        # 4. Persist preference + ontology to disk.
        try:
            self.mind.preference_persistence.save("spatial", self.mind.spatial_preference)
            self.mind.preference_persistence.save("causal", self.mind.causal_preference)
            self.mind.ontology_persistence.save(self.mind.ontology)
            self.mind.conformal_calibration.persist(self.mind.relation_conformal, "relation_extraction")
        except Exception:
            logger.exception("REM.persist: save failed")

        return reflections, summary

    def _collect_observations_for_pc(self) -> list[dict[str, object]]:
        """Build a row-per-subject observation table for PC discovery.

        Each row is one subject; columns are predicates; cells are the stored
        objects. Rows missing a column are dropped per-pair by the CI test, so
        sparse coverage isn't fatal.
        """

        rows: list[dict[str, object]] = []
        for subject in self.mind.memory.subjects():
            record = {pred: obj for pred, obj, _conf, _ev in self.mind.memory.records_for_subject(subject)}
            if record:
                rows.append(record)
        return rows

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                self.run_once()
            except Exception as exc:  # pragma: no cover - background safety net
                logger.exception("Broca background DMN loop failed")
                self.last_error = repr(exc)


class LexicalPlanGraft(BaseGraft):
    """Writes a planned next word into the frozen host's residual stream.

    This is the cleanest Broca analogy in the lab: the cognitive substrate
    decides the lexical content; this graft turns the intended lexical sequence
    into hidden-state directions that the frozen language host can emit.
    """

    def __init__(self, *, target_snr: float = DEFAULT_GRAFT_TARGET_SNR):
        super().__init__()
        self.target_snr = float(target_snr)
        self.mixer_priority = 2.0
        self.last_token_id: int | None = None
        self.last_token: str | None = None

    def forward(self, x: torch.Tensor, state: dict) -> torch.Tensor:
        if not self.enabled or "broca_plan_token_ids" not in state:
            return x
        plan = state["broca_plan_token_ids"]
        if not isinstance(plan, torch.Tensor):
            plan = torch.tensor(plan, device=x.device, dtype=torch.long)
        plan = plan.to(x.device)
        if plan.ndim == 1:
            plan = plan.view(1, -1).expand(x.shape[0], -1)
        step = state.get("broca_step", 0)
        if not isinstance(step, torch.Tensor):
            step = torch.full((x.shape[0],), int(step), device=x.device, dtype=torch.long)
        step = step.to(x.device).long().view(-1)
        step = step.clamp_min(0).clamp_max(plan.shape[1] - 1)
        target_ids = plan[torch.arange(x.shape[0], device=x.device), step]
        directions = F.normalize(state["model"].lm_head.weight[target_ids].detach().to(x.device, x.dtype), dim=-1)
        last = state["last_indices"].to(x.device)
        rows = torch.arange(x.shape[0], device=x.device)
        host_at_last = x[rows, last]
        confidence = _state_confidence(state)
        inertia = _state_inertia(state)
        magnitude = snr_magnitude(host_at_last, target_snr=self.target_snr, confidence=confidence, inertia=inertia)
        out = x.clone()
        out[rows, last] += directions * magnitude
        self.last_token_id = int(target_ids[0].item())
        tok = getattr(state.get("tokenizer", None), "decode_id", None)
        self.last_token = tok(self.last_token_id) if callable(tok) else None
        return out


class TrainableBrocaGraft(BaseGraft):
    """Trainable bridge from latent cognitive frames to language tokens.

    The host transformer can remain frozen. This module learns how to project a
    faculty-state vector plus a production step into the residual stream.
    """

    def __init__(self, d_features: int, d_model: int, *, max_steps: int = 10, step_dim: int = 16, hidden: int = 160, target_snr: float = DEFAULT_GRAFT_TARGET_SNR):
        super().__init__()
        self.d_features = int(d_features)
        self.max_steps = int(max_steps)
        self.target_snr = float(target_snr)
        self.mixer_priority = 0.35
        self.norm = nn.LayerNorm(d_features)
        self.step_emb = nn.Embedding(max_steps, step_dim)
        self.net = nn.Sequential(
            nn.Linear(d_features + step_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        nn.init.normal_(self.net[0].weight, std=0.02)
        nn.init.zeros_(self.net[0].bias)
        nn.init.normal_(self.net[2].weight, std=0.02)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, x: torch.Tensor, state: dict) -> torch.Tensor:
        if not self.enabled or "broca_features" not in state:
            return x
        feats = state["broca_features"]
        if not isinstance(feats, torch.Tensor):
            feats = torch.tensor(feats, device=x.device, dtype=x.dtype)
        param_dtype = self.norm.weight.dtype
        feats = feats.to(x.device, param_dtype)
        if feats.ndim == 1:
            feats = feats.view(1, -1).expand(x.shape[0], -1)
        if feats.shape[-1] != self.d_features:
            raise ValueError(f"expected broca_features dim {self.d_features}, got {feats.shape[-1]}")
        step = state.get("broca_step", torch.zeros(x.shape[0], device=x.device, dtype=torch.long))
        if not isinstance(step, torch.Tensor):
            step = torch.full((x.shape[0],), int(step), device=x.device, dtype=torch.long)
        step = step.to(x.device).long().view(-1).clamp(0, self.max_steps - 1)
        z = torch.cat([self.norm(feats), self.step_emb(step).to(device=x.device, dtype=param_dtype)], dim=-1)
        last = state["last_indices"].to(x.device)
        rows = torch.arange(x.shape[0], device=x.device)
        host_at_last = x[rows, last]
        direction = F.normalize(self.net(z).to(device=x.device, dtype=x.dtype), dim=-1)
        confidence = _state_confidence(state)
        inertia = _state_inertia(state)
        magnitude = snr_magnitude(host_at_last, target_snr=self.target_snr, confidence=confidence, inertia=inertia)
        out = x.clone()
        out[rows, last] += direction * magnitude
        return out


class SubstrateLogitBiasGraft(BaseGraft):
    """Dynamic, context-aware logit bias on substrate-supplied vocabulary IDs.

    Wired into the host's ``logits`` slot. The cognitive substrate supplies the
    set of token ids it wants to surface; the graft itself derives the actual
    push at every step from the host's current logit distribution so the bias
    is meaningful regardless of how confident the LLM happens to be.

    State keys it reads (all optional except ``broca_logit_bias``):

      ``broca_logit_bias`` — ``Mapping[int, float]`` of base nat-scale bonuses.
      ``broca_logit_bias_decay`` — semantic multiplier (typically 1.0 until the
        target concept has appeared in the prefix, then collapses).
      ``substrate_confidence`` — scalar in [0, 1] from the substrate frame.
      ``substrate_inertia`` — log1p(prefix length); how much momentum the LLM
        has built up that the bias must shout over.

    The dynamic push has two parts:
      1. ``target_boost = max(0, max_logit - target_logit) * confidence``
         — drag the target up to (and past) the current top logit, scaled by
         how strongly the substrate believes in the answer.
      2. ``stubborn_push = stubbornness * base_bonus``
         — augment with a structural bonus where ``stubbornness`` is the
         normalized peakedness of the distribution (1.0 when the LLM is a
         delta, ≈0 when it is uniform). A confident-but-wrong LLM gets a
         bigger nudge than an indecisive one.
      The combined value is then scaled by ``decay`` (semantic) and
      ``inertia`` (sequence-length) so the bias keeps up with autoregressive
      momentum until the target concept actually appears.
    """

    def __init__(self):
        super().__init__()
        self.mixer_priority = 0.5

    def forward(self, x: torch.Tensor, state: dict) -> torch.Tensor:
        if not self.enabled:
            return x
        bias = state.get("broca_logit_bias")
        if not bias:
            return x
        decay_raw = state.get("broca_logit_bias_decay", 1.0)
        try:
            decay = float(decay_raw)
        except (TypeError, ValueError):
            decay = 1.0
        if decay <= 0.0:
            return x

        confidence = float(_state_confidence(state))
        confidence = max(0.0, min(1.0, confidence))
        inertia = float(_state_inertia(state))
        small_inertia = 1e-6
        inertia = max(inertia, small_inertia)

        out = x.clone()
        last = state["last_indices"].to(x.device)
        rows = torch.arange(x.shape[0], device=x.device)

        last_logits = out[rows, last].float()                           # [B, V]
        max_logit = last_logits.max(dim=-1, keepdim=True).values         # [B, 1]
        log_probs = F.log_softmax(last_logits, dim=-1)
        probs = log_probs.exp()
        entropy_val = -(probs * log_probs).sum(dim=-1)                   # [B]
        log_vocab = math.log(max(2, last_logits.shape[-1]))
        # peakedness ∈ [0, 1]: 1.0 when distribution is a delta, ~0 when uniform
        stubbornness = (1.0 - entropy_val / log_vocab).clamp(0.0, 1.0).unsqueeze(-1)

        for token_id, bonus in bias.items():
            tid = int(token_id)
            if tid < 0 or tid >= out.shape[-1]:
                continue
            cur = last_logits[:, tid:tid + 1]                            # [B, 1]
            target_boost = (max_logit - cur).clamp_min(0.0) * confidence
            stubborn_push = stubbornness * float(bonus)
            delta = ((target_boost + stubborn_push) * decay * inertia).to(out.dtype)
            out[rows, last, tid] = out[rows, last, tid] + delta.squeeze(-1)
        return out


def _batch_from_ids(rows: Sequence[Sequence[int]], pad_id: int, *, device: torch.device | str | None = None):
    max_len = max(1, max(len(r) for r in rows))
    ids = torch.full((len(rows), max_len), pad_id, dtype=torch.long)
    mask = torch.zeros((len(rows), max_len), dtype=torch.bool)
    lengths = torch.tensor([max(1, len(r)) for r in rows], dtype=torch.long)
    for i, row in enumerate(rows):
        if not row:
            continue
        ids[i, : len(row)] = torch.tensor(row, dtype=torch.long)
        mask[i, : len(row)] = True
    if device is not None:
        ids = ids.to(device)
        mask = mask.to(device)
        lengths = lengths.to(device)
    return ids, mask, lengths


def default_lexical_target_snr(model: nn.Module) -> float:
    """Target SNR for the lexical Broca graft.

    Geometry-independent: the graft injects ``target_snr`` × host RMS energy
    along the planned token direction, so the same fraction works regardless of
    ``d_model``. The argument is accepted for API compatibility with callers
    that still want to inspect the host's configuration.
    """

    _ = model
    return DEFAULT_GRAFT_TARGET_SNR


def decode_generation(tokenizer: Any, generated: Sequence[int]) -> str:
    dec = getattr(tokenizer, "decode_tokens", None)
    if callable(dec):
        return str(dec(list(generated))).strip()
    return " ".join(tokenizer.decode_id(int(i)) for i in generated)


def generate_from_plan(
    model: nn.Module,
    tokenizer: Any,
    plan_tokens: Sequence[str],
    *,
    prefix: str | None = None,
    max_new_tokens: int | None = None,
    broca_features: torch.Tensor | None = None,
) -> str:
    plan_ids = list(tokenizer.encode_plan_words(plan_tokens))
    max_new_tokens = max_new_tokens or len(plan_ids)
    ids = speech_seed_ids(tokenizer, prefix)
    generated: list[int] = []
    device = next(model.parameters()).device
    steps = range(min(max_new_tokens, len(plan_ids)))
    for step in steps:
        row = ids + generated
        batch_ids, mask, _ = _batch_from_ids([row], tokenizer.pad_id, device=device)
        logits = model(
            batch_ids,
            mask,
            extra_state={
                "broca_plan_token_ids": torch.tensor([plan_ids], device=device),
                "broca_step": torch.tensor([step], device=device),
                "tokenizer": tokenizer,
                **({"broca_features": broca_features.to(device)} if broca_features is not None else {}),
            },
        )
        pred = int(logits[0, mask.long().sum().item() - 1].argmax().item())
        generated.append(pred)
    return decode_generation(tokenizer, generated)


def generate_without_broca(model: nn.Module, tokenizer: Any, *, prefix: str | None = None, max_new_tokens: int = 5) -> str:
    ids = speech_seed_ids(tokenizer, prefix)
    generated: list[int] = []
    device = next(model.parameters()).device
    for _ in range(max_new_tokens):
        row = ids + generated
        batch_ids, mask, _ = _batch_from_ids([row], tokenizer.pad_id, device=device)
        logits = model(batch_ids, mask)
        pred = int(logits[0, mask.long().sum().item() - 1].argmax().item())
        generated.append(pred)
    return decode_generation(tokenizer, generated)


@dataclass
class FacultyCandidate:
    name: str
    score: float
    build: Callable[[], CognitiveFrame]
    evidence: dict[str, Any] = field(default_factory=dict)


class CognitiveRouter:
    """Always-on faculty router over memory, action, and causal candidates.

    Each faculty receives every utterance and emits a scored latent frame. The
    selected frame is the highest precision candidate above a low relevance
    floor; otherwise the workspace still receives an ``unknown`` frame with the
    candidate traces attached.
    """

    def __init__(self, *, extractor: RelationExtractor, relevance_floor: float = 0.28):
        self.relevance_floor = float(relevance_floor)
        self.extractor: RelationExtractor = extractor

    def route(self, mind: "BrocaMind", utterance: str, toks: Sequence[str]) -> CognitiveFrame:
        candidates: list[FacultyCandidate] = []
        claim = self.extractor.extract_claim(utterance, toks)
        query = _query_from_tokens(
            toks,
            utterance=utterance,
            known_subjects=mind.memory.subjects(),
            records_for_subject=mind.memory.records_for_subject,
            text_encoder=mind.text_encoder,
        )

        if claim is not None:
            candidates.append(FacultyCandidate("semantic_claim", 1.45, lambda claim=claim: self._memory_write(mind, utterance, claim)))
        if query is not None:
            candidates.append(FacultyCandidate("semantic_query", 1.35, lambda query=query: self._memory_query(mind, utterance, toks, query)))

        active_frame = self._active_action(mind)
        active_score = 0.1 + _frame_relevance(utterance, toks, active_frame, mind.text_encoder)
        candidates.append(FacultyCandidate("active_inference", active_score, lambda active_frame=active_frame: active_frame))

        causal_frame = self._causal_effect(mind)
        causal_score = 0.1 + _frame_relevance(utterance, toks, causal_frame, mind.text_encoder)
        candidates.append(FacultyCandidate("causal_effect", causal_score, lambda causal_frame=causal_frame: causal_frame))

        ranked = sorted(candidates, key=lambda c: c.score, reverse=True)
        selected = ranked[0] if ranked and ranked[0].score >= self.relevance_floor else None
        frame = selected.build() if selected is not None else CognitiveFrame("unknown", answer="unknown", confidence=0.0, evidence={"route": "none"})
        frame.evidence = {
            **dict(frame.evidence),
            "router_selected": selected.name if selected is not None else "unknown",
            "router_candidates": [
                {"name": c.name, "score": round(float(c.score), 6)}
                for c in ranked
            ],
            "intrinsic_cues": [asdict(cue) for cue in mind.workspace.intrinsic_cues],
        }
        logger.debug(
            "CognitiveRouter.route: selected=%s intent=%s scores=%s utterance_preview=%r",
            selected.name if selected is not None else "unknown",
            frame.intent,
            [(c.name, round(float(c.score), 4)) for c in ranked],
            (utterance[:160] + "…") if len(utterance) > 160 else utterance,
        )
        return frame

    def _memory_write(self, mind: "BrocaMind", utterance: str, claim: ParsedClaim) -> CognitiveFrame:
        ev = {
            "source": "observed_utterance",
            "utterance": utterance,
            "predicate": claim.predicate,
            "instruments": ["runtime_observation"],
            **dict(claim.evidence),
        }
        existing = mind.memory.get(claim.subject, claim.predicate)
        if existing is not None and existing[0] != claim.obj:
            gap = _claim_prediction_gap(mind, utterance, claim)
            if gap is not None:
                ev["prediction_gap"] = float(gap)
        observed = mind.memory.observe_claim(
            claim.subject,
            claim.predicate,
            claim.obj,
            confidence=float(claim.confidence),
            evidence=ev,
        )
        if observed["status"] == "conflict":
            return CognitiveFrame(
                "memory_conflict",
                subject=claim.subject,
                answer=str(observed["current_object"]),
                confidence=float(observed.get("current_confidence", 0.0)),
                evidence={
                    "predicate": claim.predicate,
                    "claimed_answer": claim.obj,
                    "claim_id": observed["claim_id"],
                    "claim_status": observed["status"],
                    "counterfactual": observed["counterfactual"],
                    "source": "observed_utterance",
                    "instruments": ["runtime_observation", "counterfactual_belief_update"],
                },
            )
        return CognitiveFrame(
            "memory_write",
            subject=claim.subject,
            answer=claim.obj,
            confidence=float(claim.confidence),
            evidence={
                "predicate": claim.predicate,
                "claim_id": observed["claim_id"],
                "claim_status": observed["status"],
                "source": "observed_utterance",
                "instruments": ["runtime_observation"],
            },
        )

    def _memory_query(self, mind: "BrocaMind", utterance: str, toks: Sequence[str], query: ParsedQuery) -> CognitiveFrame:
        if not query.subject or not str(query.subject).strip():
            return CognitiveFrame(
                "unknown",
                subject="",
                answer="unknown",
                confidence=0.0,
                evidence={"missing": "semantic_subject", "predicate": query.predicate, **dict(query.evidence)},
            )
        rec = mind.memory.get(query.subject, query.predicate)
        if rec is None:
            return CognitiveFrame(
                "unknown",
                subject=query.subject,
                answer="unknown",
                confidence=0.0,
                evidence={"missing": "semantic_memory", "predicate": query.predicate, **dict(query.evidence)},
            )

        obj, conf, ev = rec
        frame = CognitiveFrame("memory_lookup", subject=query.subject, answer=obj, confidence=conf, evidence=dict(ev))
        mu_pop = mind.memory.mean_confidence()
        frame.evidence["semantic_mean_confidence"] = max(SEMANTIC_CONFIDENCE_FLOOR, float(mu_pop or 0.0))
        frame.evidence["predicate"] = query.predicate

        known_objects = mind.memory.distinct_objects_for_predicate(query.predicate)
        mentioned_objects = [t for t in toks if t in known_objects]
        conflicting = bool(mentioned_objects and mentioned_objects[-1] != obj.lower())
        if not conflicting:
            return frame

        plan_words = frame.speech_plan()
        broca_features = frame.to_features(mind.text_encoder)
        ce_g, ce_p, gap = lexical_surprise_gap(
            mind.host,
            mind.tokenizer,
            utterance=utterance,
            plan_words=plan_words,
            broca_features=broca_features,
        )
        frame.evidence["prediction_ce_graft"] = ce_g
        frame.evidence["prediction_ce_plain"] = ce_p
        frame.evidence["prediction_gap"] = gap
        if gap <= 0.0:
            return frame

        coupled = mind.unified_agent.decide()
        return CognitiveFrame(
            "prediction_error",
            subject=query.subject,
            answer=obj,
            confidence=conf,
            evidence={
                **dict(frame.evidence),
                "delta_ce": gap,
                "coupled_faculty": coupled.faculty,
                "wake_action": coupled.action_name,
                "spatial_min_G": coupled.spatial_min_G,
                "causal_min_G": coupled.causal_min_G,
            },
        )

    def _active_action(self, mind: "BrocaMind") -> CognitiveFrame:
        coupled = mind.unified_agent.decide()
        posterior_spatial = {
            mind.pomdp.action_names[i]: float(p)
            for i, p in enumerate(coupled.spatial_decision.posterior_over_policies[: len(mind.pomdp.action_names)])
        }
        causal_names = mind.causal_pomdp.action_names
        posterior_causal = {
            causal_names[i]: float(p)
            for i, p in enumerate(coupled.causal_decision.posterior_over_policies[: len(causal_names)])
        }
        conf = (
            max(coupled.spatial_decision.posterior_over_policies)
            if coupled.faculty == "spatial"
            else max(coupled.causal_decision.posterior_over_policies)
        )
        return CognitiveFrame(
            "active_action",
            answer=coupled.action_name,
            confidence=float(conf),
            evidence={
                "coupled_faculty": coupled.faculty,
                "spatial_min_G": coupled.spatial_min_G,
                "causal_min_G": coupled.causal_min_G,
                "expected_free_energy_spatial": min(ev.expected_free_energy for ev in coupled.spatial_decision.policies),
                "expected_free_energy_causal": min(ev.expected_free_energy for ev in coupled.causal_decision.policies),
                "policy_posterior": posterior_spatial,
                "causal_policy_posterior": posterior_causal,
            },
        )

    def _causal_effect(self, mind: "BrocaMind") -> CognitiveFrame:
        p1 = mind.scm.probability({"Y": 1}, interventions={"T": 1})
        p0 = mind.scm.probability({"Y": 1}, interventions={"T": 0})
        ate = p1 - p0
        intervention_var = next((name for name in mind.scm.endogenous_names if name in mind.scm.domains and len(mind.scm.domains[name]) == 2), "intervention")
        labels = getattr(mind.scm, "labels", {}) or {}
        answer_key = "positive_effect" if ate >= 0 else "negative_effect"
        return CognitiveFrame(
            "causal_effect",
            subject=str(labels.get(intervention_var, intervention_var)),
            answer=str(labels.get(answer_key, answer_key)),
            confidence=float(min(1.0, abs(ate))),
            evidence={"p_do_positive": p1, "p_do_negative": p0, "ate": ate, "labels": labels},
        )


class BrocaMind:
    """Cognitive substrate with the language model demoted to speech interface."""

    host: LlamaBrocaHost
    tokenizer: HuggingFaceBrocaTokenizer

    def __init__(
        self,
        *,
        seed: int = 0,
        db_path: str | Path = "runs/broca_semantic_memory.sqlite",
        namespace: str = "main",
        llama_model_id: str | None = None,
        device: torch.device | str | None = None,
        hf_token: str | bool | None = None,
        lexical_target_snr: float | None = None,
    ):
        self.seed = seed
        resolved_device = device if isinstance(device, torch.device) else pick_torch_device(device)
        mid = llama_model_id or DEFAULT_BROCA_MODEL_ID
        self.memory = PersistentSemanticMemory(db_path, namespace=namespace)
        self.journal = WorkspaceJournal(Path(db_path))
        self.episode_graph = EpisodeAssociationGraph(Path(db_path))
        self._last_journal_id: int | None = None
        self.host, self.tokenizer = load_llama_broca_host(mid, device=resolved_device, token=hf_token)
        self.text_encoder = frozen_subword_projector_from_model(self.host, self.tokenizer)
        snr = lexical_target_snr if lexical_target_snr is not None else default_lexical_target_snr(self.host)
        self.lexical_graft = LexicalPlanGraft(target_snr=snr)
        self.host.add_graft("final_hidden", self.lexical_graft)
        self.feature_graft = TrainableBrocaGraft(
            COGNITIVE_FRAME_DIM,
            int(getattr(self.host.cfg, "d_model", 96)),
            target_snr=snr,
        )
        params = getattr(self.host, "parameters", None)
        if callable(params):
            host_param = next(params(), None)
            if host_param is not None:
                self.feature_graft.to(host_param.device)
        self.host.add_graft("final_hidden", self.feature_graft)
        self.logit_bias_graft = SubstrateLogitBiasGraft()
        self.host.add_graft("logits", self.logit_bias_graft)
        self.workspace = GlobalWorkspace()
        self.router = CognitiveRouter(extractor=LLMRelationExtractor(self.host, self.tokenizer))
        self.pomdp = build_tiger_pomdp()
        self.active_agent = ActiveInferenceAgent(self.pomdp, horizon=1, learn=False)
        self.scm = build_simpson_scm()
        self.causal_pomdp = build_causal_epistemic_pomdp(self.scm)
        self.causal_agent = ActiveInferenceAgent(self.causal_pomdp, horizon=1, learn=False)
        self.unified_agent = CoupledEFEAgent(self.active_agent, self.causal_agent)
        self._background_worker: CognitiveBackgroundWorker | None = None

        # New substrates ----------------------------------------------------
        d_model = int(getattr(self.host.cfg, "d_model", 96))
        self.vsa = VSACodebook(dim=10_000, base_seed=int(seed))
        self.hopfield_memory = HopfieldAssociativeMemory(d_model=d_model, max_items=65_536)
        self.conformal_calibration = PersistentConformalCalibration(Path(db_path), namespace=f"{namespace}__conformal")
        self.relation_conformal = ConformalPredictor(alpha=0.1, method="lac", min_calibration=8)
        self.conformal_calibration.hydrate(self.relation_conformal, channel="relation_extraction")
        # Hawkes channels are populated lazily by ``observe_event`` so the
        # excitation matrix grows with the user's vocabulary instead of being
        # hardcoded.
        self.hawkes_persistence = PersistentHawkes(Path(db_path), namespace=f"{namespace}__hawkes")
        loaded = self.hawkes_persistence.load()
        self.hawkes = loaded if loaded is not None else MultivariateHawkesProcess(beta=0.5, baseline=0.05)
        # One Dirichlet preference per active-inference faculty.
        self.preference_persistence = PersistentPreference(Path(db_path), namespace=f"{namespace}__pref")
        self.spatial_preference = self.preference_persistence.load("spatial") or DirichletPreference(
            len(self.pomdp.observation_names),
            initial_C=list(self.pomdp.C),
            prior_strength=4.0,
        )
        self.causal_preference = self.preference_persistence.load("causal") or DirichletPreference(
            len(self.causal_pomdp.observation_names),
            initial_C=list(self.causal_pomdp.C),
            prior_strength=4.0,
        )
        self._sync_preference_to_pomdp()
        # Hebbian-promoted ontology axes share the sketch dimension.
        self.ontology_persistence = PersistentOntologicalRegistry(Path(db_path), namespace=f"{namespace}__ontology")
        self.ontology = self.ontology_persistence.load(dim=SKETCH_DIM, frequency_threshold=8)
        # Causal-discovery learns a fresh SCM from observation data when DMN
        # decides the user has accumulated enough coherent variables to
        # justify rebuilding the model. The learned SCM is kept separate from
        # the bootstrap Simpson model so it is easy to A/B in benchmarks.
        self.discovered_scm: Any = None
        # Replay buffer for motor learning. Each item is one chat turn the
        # substrate produced; the trainer pulls items from here at REM time.
        self.motor_replay: list[dict] = []

        # Proceduralization (System 2 → System 1). The macro registry persists
        # compiled motifs across processes; the compiler runs on every DMN tick
        # and grows the registry as repeated reasoning patterns are detected.
        self.macro_registry = MacroChunkRegistry(Path(db_path), namespace=f"{namespace}__macros")
        self.chunking_compiler = DMNChunkingCompiler(self, registry=self.macro_registry)

        # Native tool synthesis. Tools live in the same SQLite file but in their
        # own namespace; ``attach_tools_to_scm`` rehydrates every persisted tool
        # into the live SCM as an endogenous equation.
        self.tool_registry = NativeToolRegistry(Path(db_path), namespace=f"{namespace}__tools")
        try:
            self.tool_registry.attach_to_scm(self.scm)
        except Exception:
            logger.exception("BrocaMind: initial tool attachment failed")

        # Activation-memory-backed dynamic graft synthesizer. The same SQLite
        # file backs the activation memory; modes are stored under their own
        # kind so they don't collide with other activation rows.
        self.activation_memory = SQLiteActivationMemory(
            Path(db_path), default_namespace=f"{namespace}__activation"
        )
        self.dynamic_graft_synth = DynamicGraftSynthesizer(
            self.activation_memory, namespace=f"{namespace}__activation"
        )

        # Tool foraging agent. The number of existing tools and the unified
        # agent's posterior entropy together drive when ``synthesize_tool``
        # wins on Expected Free Energy.
        self.tool_foraging_agent = ToolForagingAgent.build(
            n_existing_tools=self.tool_registry.count(),
            insufficient_prior=0.5,
        )

    @property
    def background_worker(self) -> CognitiveBackgroundWorker | None:
        return self._background_worker

    def consolidate_once(self) -> list[dict]:
        out = self.memory.consolidate_claims_once()
        logger.debug("BrocaMind.consolidate_once: reflections=%d", len(out))
        return out

    # -- New substrate plumbing -----------------------------------------------

    def _sync_preference_to_pomdp(self) -> None:
        """Push the Dirichlet means into the live POMDPs' C vectors."""

        try:
            self.pomdp.C = list(self.spatial_preference.expected_C())
        except (AttributeError, TypeError):
            logger.exception("BrocaMind._sync_preference_to_pomdp: spatial sync failed")
        try:
            self.causal_pomdp.C = list(self.causal_preference.expected_C())
        except (AttributeError, TypeError):
            logger.exception("BrocaMind._sync_preference_to_pomdp: causal sync failed")

    def observe_user_feedback(self, *, faculty: str, observation_index: int, polarity: float, weight: float = 1.0, reason: str = "") -> None:
        """Forward user feedback into the right Dirichlet preference and sync."""

        if faculty == "spatial":
            target = self.spatial_preference
        elif faculty == "causal":
            target = self.causal_preference
        else:
            raise ValueError(f"BrocaMind.observe_user_feedback: unsupported faculty {faculty!r}; expected 'spatial' or 'causal'")
        target.update(observation_index, polarity=polarity, weight=weight, reason=reason)
        self._sync_preference_to_pomdp()
        try:
            self.preference_persistence.save(faculty, target)
        except (sqlite3.Error, OSError):
            logger.exception("BrocaMind.observe_user_feedback: preference save failed")

    def observe_event(self, channel: str, *, t: float | None = None) -> None:
        """Record an event on the Hawkes layer (used by the conversational loop)."""

        self.hawkes.observe(channel, t=t)

    def encode_triple_vsa(self, subject: str, predicate: str, obj: str) -> torch.Tensor:
        """Compose a hypervector representation of (subject, predicate, object).

        The VSA bundle is independent of the LLM's tokenizer and lets the
        substrate do role-filler algebra on facts without round-tripping
        through subwords.
        """

        return self.vsa.encode_triple(subject, predicate, obj)

    # -- Native tool synthesis -------------------------------------------------

    def synthesize_native_tool(
        self,
        name: str,
        source: str,
        *,
        function_name: str | None = None,
        parents: Sequence[str],
        domain: Sequence[Any],
        sample_inputs: Sequence[dict],
        description: str = "",
        attach: bool = True,
        overwrite: bool = False,
    ) -> NativeTool:
        """Compile, sandbox, verify, persist, and (optionally) attach a synthesized tool.

        After synthesis the tool foraging agent's belief is updated to reflect
        the larger toolbox, so the next ``synthesize_tool`` decision factors in
        the additional coverage.
        """

        tool = self.tool_registry.synthesize(
            name,
            source,
            function_name=function_name,
            parents=parents,
            domain=domain,
            sample_inputs=sample_inputs,
            description=description,
            overwrite=overwrite,
        )
        if attach:
            try:
                self.tool_registry.attach_to_scm(self.scm)
            except Exception:
                logger.exception("BrocaMind.synthesize_native_tool: SCM re-attach failed")
        # Rebuild the tool foraging agent so its likelihoods reflect the new tool count.
        self.tool_foraging_agent = ToolForagingAgent.build(
            n_existing_tools=self.tool_registry.count(),
            insufficient_prior=0.5,
        )
        return tool

    def attach_tools_to_scm(self) -> int:
        """Re-attach every persisted native tool onto :attr:`scm`. Returns the count attached."""

        return self.tool_registry.attach_to_scm(self.scm)

    def should_synthesize_tool(self) -> bool:
        """Run the tool foraging agent against the current substrate state.

        The ``insufficient_prior`` is derived from the unified agent's
        normalized posterior entropy: when the substrate is genuinely
        confused (high entropy → high prior on ``knowledge_insufficient``)
        the EFE math will prefer ``synthesize_tool`` over the alternatives.
        """

        try:
            coupled = self.unified_agent.decide()
        except Exception:
            return False
        # Use whichever faculty currently wins on EFE; its posterior entropy is
        # the substrate's best self-estimate of confusion.
        if coupled.faculty == "spatial":
            posterior = list(coupled.spatial_decision.posterior_over_policies)
        else:
            posterior = list(coupled.causal_decision.posterior_over_policies)
        n = len(posterior)
        if n < 2:
            insufficient_prior = 0.5
        else:
            h = belief_entropy(posterior)
            h_max = math.log(n)
            insufficient_prior = max(1e-6, min(1 - 1e-6, h / max(h_max, 1e-9)))
        self.tool_foraging_agent.update_belief(insufficient_prior=float(insufficient_prior))
        return self.tool_foraging_agent.should_synthesize()

    # -- Proceduralization / macro lookup --------------------------------------

    def recent_intents(self, *, limit: int = 8) -> list[str]:
        try:
            rows = self.journal.recent(limit=int(limit))
        except Exception:
            return []
        return [str(r.get("intent", "") or "unknown") for r in rows]

    def find_matching_macro(self, *, recent_intents: Sequence[str] | None = None) -> CompiledMacro | None:
        """Return the most-observed macro whose prefix matches the recent intent tail."""

        recent = list(recent_intents) if recent_intents is not None else self.recent_intents()
        return self.macro_registry.find_macro_matching_prefix(recent)

    def macro_speech_features(self, macro: CompiledMacro) -> torch.Tensor:
        """Return the COGNITIVE_FRAME_DIM-shaped features the macro should inject via TrainableBrocaGraft."""

        return macro_frame_features(macro)

    # -- Dynamic graft synthesis -----------------------------------------------

    def synthesize_activation_mode(
        self,
        *,
        name: str,
        prompt: str,
        slot: str = "final_hidden",
        query_mode: str = "sequence_mean",
        value_mode: str = "mean_activation",
        target_token: str | None = None,
        confidence: float = 1.0,
    ) -> CapturedActivationMode:
        """Capture and persist an activation mode for the host (System-1 LLM tool).

        The captured mode lives in :attr:`activation_memory` and can be loaded
        into a :class:`KVMemoryGraft` via
        :meth:`load_activation_modes_into_graft`.
        """

        return self.dynamic_graft_synth.synthesize(
            self.host,
            self.tokenizer,
            name=name,
            prompt=prompt,
            slot=slot,
            query_mode=query_mode,
            value_mode=value_mode,
            target_token=target_token,
            confidence=float(confidence),
        )

    def load_activation_modes_into_graft(
        self,
        graft: Any,
        *,
        names: Optional[Sequence[str]] = None,
        clear_first: bool = True,
    ) -> int:
        return self.dynamic_graft_synth.load_modes(
            graft, names=names, clear_first=clear_first
        )

    def vector_for_concept(self, name: str, *, base_sketch: torch.Tensor | None = None) -> torch.Tensor:
        """Return the substrate's preferred vector for a concept name.

        Routes through the ontology registry so frequent concepts use their
        promoted orthogonal axis; less-frequent ones still use the hashed
        sketch. Always observes the access (so the next call can flip
        promotion).
        """

        self.ontology.observe(name)
        sketch = base_sketch if base_sketch is not None else stable_sketch(name, dim=SKETCH_DIM)
        promoted = self.ontology.maybe_promote(name, sketch)
        if promoted is not None:
            return promoted.axis
        return F.normalize(sketch.detach().to(torch.float32).flatten(), dim=0)

    def start_background(
        self,
        *,
        interval_s: float = 5.0,
        config: DMNConfig | None = None,
    ) -> CognitiveBackgroundWorker:
        if self._background_worker is None:
            self._background_worker = CognitiveBackgroundWorker(self, interval_s=interval_s, config=config)
        else:
            self._background_worker.interval_s = max(0.1, float(interval_s))
            if config is not None:
                self._background_worker.config = config
        self._background_worker.start()
        return self._background_worker

    def stop_background(self) -> None:
        if self._background_worker is not None:
            self._background_worker.stop()

    def _intrinsic_scan(self, toks: list[str]) -> None:
        self.workspace.intrinsic_cues.clear()
        mu_pop = self.memory.mean_confidence()
        confidence_floor = SEMANTIC_CONFIDENCE_FLOOR if mu_pop is None else max(SEMANTIC_CONFIDENCE_FLOOR, float(mu_pop))
        toks_set = set(toks)
        for ent in self.memory.subjects():
            if ent not in toks_set:
                continue
            records = self.memory.records_for_subject(ent)
            if not records:
                self.workspace.intrinsic_cues.append(IntrinsicCue(1.0, "memory_gap", {"subject": ent}))
                continue
            best_pred, _obj, best_conf, _ev = max(records, key=lambda row: row[2])
            if best_conf < confidence_floor:
                self.workspace.intrinsic_cues.append(
                    IntrinsicCue(
                        float(confidence_floor - best_conf),
                        "memory_low_confidence",
                        {"subject": ent, "predicate": best_pred, "confidence": best_conf},
                    )
                )
        cq = self.causal_agent.qs
        if cq is not None and len(cq) >= 2:
            max_ent = math.log(len(cq))
            h_q = belief_entropy(cq)
            if max_ent > 1e-9 and h_q > 0.5 * max_ent:
                self.workspace.intrinsic_cues.append(IntrinsicCue(float(h_q / max_ent), "causal_uncertain", {"entropy": h_q}))
        logger.debug("_intrinsic_scan: cues=%d toks=%d", len(self.workspace.intrinsic_cues), len(toks))

    def comprehend(self, utterance: str) -> CognitiveFrame:
        toks = utterance_words(utterance)
        self._intrinsic_scan(toks)
        frame = self.router.route(self, utterance, toks)
        out = self._commit_frame(utterance, toks, frame)
        # Tell the Hawkes layer that this intent fired so its working memory
        # heats up the relevant channels for the next few seconds.
        try:
            self.hawkes.observe(str(out.intent or "unknown"))
        except Exception:
            logger.exception("comprehend: hawkes observe failed")
        # Tell the DMN the user is active so REM doesn't fire mid-conversation.
        if self._background_worker is not None:
            self._background_worker.mark_user_active()
        # Promote frequent concepts to dedicated orthogonal axes.
        for concept in (out.subject, out.answer):
            if isinstance(concept, str) and concept and concept != "unknown":
                self.ontology.observe(concept)
                base = stable_sketch(concept, dim=SKETCH_DIM)
                self.ontology.maybe_promote(concept, base)
        # Bind the triple into the VSA codebook for algebraic recall.
        if out.subject and out.answer and out.intent in {"memory_write", "memory_lookup"}:
            try:
                self.vsa.encode_triple(out.subject, str((out.evidence or {}).get("predicate", out.intent)), out.answer)
            except Exception:
                logger.exception("comprehend: vsa encode failed")
        logger.debug(
            "comprehend: intent=%s confidence=%s journal_id=%s",
            out.intent,
            out.confidence,
            (out.evidence or {}).get("journal_id"),
        )
        return out

    def _commit_frame(self, utterance: str, toks: Sequence[str], frame: CognitiveFrame) -> CognitiveFrame:
        jid = self.journal.append(utterance, frame)
        frame.evidence = {**frame.evidence, "journal_id": jid}
        if self._last_journal_id is not None:
            self.episode_graph.bump(self._last_journal_id, jid)
        self._last_journal_id = jid
        logger.debug("_commit_frame: journal_id=%s intent=%s pred_error=%s", jid, frame.intent, frame.intent == "prediction_error")
        if frame.intent == "prediction_error":
            pred = str(frame.evidence.get("predicate", ""))
            if not pred and frame.subject:
                records = self.memory.records_for_subject(frame.subject)
                pred = records[0][0] if records else ""
            known_objects = self.memory.distinct_objects_for_predicate(pred)
            observed_objects = [t for t in toks if t in known_objects]
            if observed_objects and frame.subject and pred:
                self.memory.upsert(
                    frame.subject,
                    pred,
                    observed_objects[-1],
                    confidence=1.0,
                    evidence={"journal_id": jid, "resolver": "prediction_error"},
                )
        out = self.workspace.publish(frame)
        for tail in self.workspace.frames:
            pred = str((tail.evidence or {}).get("predicate", ""))
            if tail.intent == "synthesis_bundle" and tail.subject and pred:
                self.memory.merge_epistemic_evidence(tail.subject, pred, tail.evidence)
        logger.debug("_commit_frame: published intent=%s workspace_frames=%d", out.intent, len(self.workspace.frames))
        return out

    def retrieve_episode(self, episode_id: int) -> CognitiveFrame:
        """Reload a prior workspace episode into working memory (persistent episodic retrieval)."""

        row = self.journal.fetch(episode_id)
        if row is None:
            logger.debug("retrieve_episode: missing id=%s", episode_id)
            return CognitiveFrame(
                "unknown",
                answer="unknown",
                confidence=0.0,
                evidence={"missing_episode_id": int(episode_id)},
            )
        replay = cognitive_frame_from_episode_row(row)
        self.workspace.publish(replay)
        logger.debug("retrieve_episode: id=%s intent=%s", episode_id, replay.intent)
        return replay

    def speak(self, frame: CognitiveFrame) -> str:
        """Plan-forced surface generation via :class:`LexicalPlanGraft`.

        Retained for benchmark code that scores the substrate's ability to
        produce specific tokens. Conversational use should call
        :meth:`chat_reply` so the LLM speaks freely under soft graft bias.
        """

        return generate_from_plan(
            self.host,
            self.tokenizer,
            frame.speech_plan(),
            broca_features=frame.to_features(self.text_encoder),
        )

    def answer(self, utterance: str) -> tuple[CognitiveFrame, str]:
        """One-shot natural-language reply driven by substrate-biased decoding."""

        return self.chat_reply([{"role": "user", "content": utterance}])

    def chat_reply(
        self,
        messages: Sequence[dict[str, str]],
        *,
        max_new_tokens: int = 256,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9,
        on_token: Callable[[str], None] | None = None,
    ) -> tuple[CognitiveFrame, str]:
        """Substrate-biased free-form chat reply.

        The last user message routes through :meth:`comprehend` to obtain a
        cognitive frame. The frame's continuous features feed
        :class:`TrainableBrocaGraft` (residual-stream bias) and a derived
        logit-bias dict over the answer's content subwords feeds
        :class:`SubstrateLogitBiasGraft` (token-level bias). The LLM then
        decodes a free-form reply through its own chat template — surface
        form, fluency, and ordering are entirely the LLM's choice. The
        sampling temperature is annealed by the frame's confidence so
        high-confidence frames produce decisive replies and ``unknown`` /
        low-confidence frames let the LLM speak freely with no bias at all.
        """

        msgs = [dict(m) for m in messages]
        if not msgs or msgs[-1].get("role") != "user":
            raise ValueError("chat_reply expects messages ending with a user turn")
        user_text = str(msgs[-1].get("content", "")).strip()
        frame = self.comprehend(user_text)

        broca_features = frame.to_features(self.text_encoder) if frame.intent != "unknown" else None
        logit_bias = self._content_logit_bias(frame)
        confidence = max(0.0, min(1.0, float(frame.confidence)))
        eff_temperature = max(
            1e-3,
            float(temperature) * self._substrate_temperature_scale(frame, confidence),
        )
        logger.debug(
            "chat_reply: intent=%s bias_tokens=%d has_broca_features=%s confidence=%.3f eff_temperature=%.3f",
            frame.intent,
            len(logit_bias),
            broca_features is not None,
            confidence,
            eff_temperature,
        )

        text = self._stream_substrate_chat(
            msgs,
            broca_features=broca_features,
            logit_bias=logit_bias,
            max_new_tokens=int(max_new_tokens),
            do_sample=bool(do_sample),
            temperature=eff_temperature,
            top_p=float(top_p),
            on_token=on_token,
            substrate_confidence=confidence,
        )
        return frame, text

    def _substrate_temperature_scale(self, frame: CognitiveFrame, confidence: float) -> float:
        """Sampling temperature multiplier derived from substrate posterior entropy.

        Couples the LLM's decoding entropy to the active-inference faculty's
        posterior over policies: when the substrate is confused (high
        normalized entropy) the LLM is given headroom to explore; when the
        substrate has collapsed onto a single policy the LLM samples nearly
        greedily so it cannot drift away from the decided answer.
        """

        if frame.intent == "unknown":
            return 1.0
        try:
            coupled = self.unified_agent.decide()
        except (RuntimeError, ValueError, IndexError):
            logger.debug("_substrate_temperature_scale: unified_agent.decide() unavailable")
            return max(1e-3, 1.0 - 0.6 * float(confidence))
        if coupled.faculty == "spatial":
            posterior = list(coupled.spatial_decision.posterior_over_policies)
        else:
            posterior = list(coupled.causal_decision.posterior_over_policies)
        n = len(posterior)
        if n < 2:
            return max(1e-3, 1.0 - 0.6 * float(confidence))
        h_q = belief_entropy(posterior)
        h_max = math.log(n)
        if h_max <= 1e-9:
            return max(1e-3, 1.0 - 0.6 * float(confidence))
        normalized_uncertainty = max(0.0, min(1.0, h_q / h_max))
        # Multiplicatively combine the substrate's posterior entropy with the
        # frame's own confidence so both signals can pull temperature down.
        return max(1e-3, normalized_uncertainty * (1.0 - 0.6 * float(confidence)))

    def _content_logit_bias(self, frame: CognitiveFrame) -> dict[int, float]:
        """Map substrate content (subject / predicate / answer) to subword token ids.

        The numeric value attached to each token is a *base bonus* that the
        :class:`SubstrateLogitBiasGraft` interprets dynamically: it is scaled
        per step by the host's current peakedness, the substrate's confidence,
        and the autoregressive inertia, so callers do not need to guess a
        magnitude that wins against an arbitrary LLM. A unit base bonus is
        therefore the right choice — bias importance comes from the substrate
        frame, not from a hand-tuned scalar.
        """

        if frame.intent == "unknown":
            return {}
        targets: list[str] = []
        if frame.subject:
            targets.append(str(frame.subject))
        if frame.answer and frame.answer.lower() != "unknown":
            targets.append(str(frame.answer))
        pred = (frame.evidence or {}).get("predicate") or (frame.evidence or {}).get("predicate_surface")
        if isinstance(pred, str) and pred:
            targets.append(pred)
        if not targets:
            return {}
        hf_tok = getattr(self.tokenizer, "inner", None)
        bias: dict[int, float] = {}
        for surface in targets:
            surface = surface.strip()
            if not surface:
                continue
            ids: list[int] = []
            if hf_tok is not None and callable(getattr(hf_tok, "encode", None)):
                ids.extend(int(t) for t in hf_tok.encode(surface, add_special_tokens=False))
                ids.extend(int(t) for t in hf_tok.encode(" " + surface, add_special_tokens=False))
            else:
                ids.extend(int(t) for t in self.tokenizer.encode(surface))
            for tid in ids:
                if tid < 0:
                    continue
                bias[tid] = max(bias.get(tid, 0.0), 1.0)
        return bias

    def _stream_substrate_chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        broca_features: torch.Tensor | None,
        logit_bias: dict[int, float],
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        on_token: Callable[[str], None] | None,
        substrate_confidence: float = 1.0,
    ) -> str:
        hf_tok = getattr(self.tokenizer, "inner", None)
        if hf_tok is None or not callable(getattr(hf_tok, "apply_chat_template", None)):
            raise RuntimeError("chat_reply requires a HuggingFace chat-template tokenizer at .tokenizer.inner")

        device = next(self.host.parameters()).device
        prompt = hf_tok.apply_chat_template(list(messages), add_generation_prompt=True, return_tensors="pt")
        if not isinstance(prompt, torch.Tensor):
            prompt = prompt["input_ids"]
        prompt = prompt.to(device)
        if prompt.ndim == 1:
            prompt = prompt.view(1, -1)

        eos_id = getattr(hf_tok, "eos_token_id", None)
        current = prompt[0].tolist()
        generated: list[int] = []
        bias_active = bool(logit_bias)
        feature_tensor = broca_features.to(device) if broca_features is not None else None
        target_token_set = {int(t) for t in logit_bias.keys()} if bias_active else set()
        target_emitted = False

        logger.debug(
            "_stream_substrate_chat: prompt_len=%d max_new_tokens=%d bias_active=%s feature_active=%s confidence=%.3f",
            int(prompt.shape[1]),
            int(max_new_tokens),
            bias_active,
            feature_tensor is not None,
            float(substrate_confidence),
        )
        with torch.no_grad():
            for step in range(max(1, int(max_new_tokens))):
                row_t = torch.tensor([current], device=device, dtype=torch.long)
                mask_t = torch.ones_like(row_t, dtype=torch.bool)
                # Inertia grows with the autoregressive prefix so the bias and
                # SNR-targeted grafts can shout over a long babbling tail.
                inertia = math.log1p(float(len(current)))
                extra: dict[str, Any] = {
                    "tokenizer": self.tokenizer,
                    "substrate_confidence": float(substrate_confidence),
                    "substrate_inertia": float(inertia),
                }
                if feature_tensor is not None:
                    extra["broca_features"] = feature_tensor
                if bias_active:
                    # Semantic decay: full strength until any target subword is
                    # emitted, then fall away so the LLM is free to finish the
                    # reply naturally without being hammered into repeating it.
                    semantic_decay = 0.15 if target_emitted else 1.0
                    extra["broca_logit_bias"] = logit_bias
                    extra["broca_logit_bias_decay"] = semantic_decay
                logits = self.host(row_t, mask_t, extra_state=extra)
                last_pos = row_t.shape[1] - 1
                logits_row = logits[0, last_pos].float()
                if do_sample:
                    scaled = logits_row / max(temperature, 1e-5)
                    probs = torch.softmax(scaled, dim=-1)
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cdf = torch.cumsum(sorted_probs, dim=-1)
                    over = (cdf > top_p).nonzero(as_tuple=False)
                    keep = int(over[0, 0].item()) + 1 if over.numel() > 0 else int(probs.numel())
                    keep = max(1, keep)
                    kept_probs = sorted_probs[:keep]
                    kept_idx = sorted_idx[:keep]
                    kept_probs = kept_probs / kept_probs.sum().clamp_min(1e-12)
                    pick = int(torch.multinomial(kept_probs, num_samples=1).item())
                    pred = int(kept_idx[pick].item())
                else:
                    pred = int(logits_row.argmax().item())
                if eos_id is not None and pred == int(eos_id):
                    break
                generated.append(pred)
                current.append(pred)
                if bias_active and not target_emitted and pred in target_token_set:
                    target_emitted = True
                if on_token is not None:
                    piece = hf_tok.decode([pred], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    if piece:
                        on_token(piece)
        reply = hf_tok.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        logger.debug("_stream_substrate_chat: emitted_tokens=%d reply_preview=%r", len(generated), reply[:200] if len(reply) > 200 else reply)
        return reply
