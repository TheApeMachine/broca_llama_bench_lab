
from __future__ import annotations

import json
import math
import os
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .active_inference import (
    ActiveInferenceAgent,
    CoupledEFEAgent,
    build_causal_epistemic_pomdp,
    build_tiger_pomdp,
    entropy as belief_entropy,
)
from .causal import build_simpson_scm
from .continuous_frame import (
    COGNITIVE_FRAME_DIM,
    TextEncoder,
    frozen_subword_projector_from_model,
    pack_cognitive_frame,
    stable_sketch,
)
from .device_utils import pick_torch_device
from .grafts import BaseGraft
from .hf_tokenizer_compat import HuggingFaceBrocaTokenizer
from .llama_broca_host import LlamaBrocaHost, load_llama_broca_host
from .predictive_coding import lexical_surprise_gap
from .substrate_graph import EpisodeAssociationGraph, merge_epistemic_evidence_dict
from .tokenizer import speech_seed_ids, utterance_words


DEFAULT_BROCA_MODEL_ID = os.environ.get("ASI_BROCA_MODEL_ID", "meta-llama/Llama-3.2-1B-Instruct")
SEMANTIC_CONFIDENCE_FLOOR = 0.5
BELIEF_REVISION_MARGIN = 0.5
BELIEF_REVISION_MIN_CLAIMS = 2


def _speech_plan_active_action(answer: str) -> list[str]:
    # Keep action labels verbatim (e.g. ``open_left``) so checks that token-scan action names stay aligned.
    tok = answer.strip()
    return ["i", "should", tok, "."]


def _speech_plan_causal(answer: str) -> list[str]:
    lab = answer.strip().replace("_", " ")
    return ["intervention", "says", "treatment", lab, "."]


def _speech_plan_open_vocab(frame: "CognitiveFrame") -> list[str]:
    """Last resort — still emits content keyed by arbitrary intent / answer strings."""

    if frame.answer and frame.answer != "unknown":
        ans = frame.answer.replace("_", " ")
        if frame.subject:
            sub = frame.subject.replace("_", " ")
            return [sub, "maps", "to", ans, "."]
        return ["faculty", "reports", ans, "."]
    lab = frame.intent.replace("_", " ")
    return ["working", "memory", "holds", "intent", lab, "."]


QUESTION_WORDS = frozenset({"where", "what", "who", "when", "why", "how", "does", "do", "did", "is", "are", "can", "should"})
MEMORY_COMMAND_WORDS = frozenset({"remember", "note", "learn", "store"})
LOCATION_RELATORS = frozenset({"in", "at", "inside", "near", "on", "located"})


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


def _predicate_from_relator(relator: str) -> str:
    rel = relator.lower()
    return "location" if rel in LOCATION_RELATORS else rel


def _location_assertion_from_tokens(toks: Sequence[str]) -> tuple[str, str] | None:
    """Backward-compatible shim over the open vocabulary claim parser."""

    claim = _claim_from_tokens(toks)
    if claim is not None and claim.predicate == "location":
        return claim.subject, claim.obj
    return None


def _claim_from_tokens(toks: Sequence[str]) -> ParsedClaim | None:
    """Parse an observed relational claim without fixed entity/value slots."""

    words = _word_tokens(toks)
    if not words or words[0] in QUESTION_WORDS:
        return None
    source_words = words
    if words[0] in MEMORY_COMMAND_WORDS:
        words = words[1:]
    if len(words) >= 4 and words[1] == "is" and words[2] in LOCATION_RELATORS:
        return ParsedClaim(
            subject=words[0],
            predicate=_predicate_from_relator(words[2]),
            obj=words[3],
            confidence=1.0,
            evidence={"parser": "relational_claim", "source_words": source_words},
        )
    if len(words) >= 4 and words[1] in {"lives", "live"} and words[2] == "in":
        return ParsedClaim(
            subject=words[0],
            predicate="location",
            obj=words[3],
            confidence=1.0,
            evidence={"parser": "verb_relation_claim", "source_words": source_words, "verb": words[1]},
        )
    if len(words) >= 3 and words[1] in {"is", "are"}:
        return ParsedClaim(
            subject=words[0],
            predicate="attribute",
            obj=words[2],
            confidence=0.8,
            evidence={"parser": "attribute_claim", "source_words": source_words, "verb": words[1]},
        )
    return None


def _query_from_tokens(toks: Sequence[str]) -> ParsedQuery | None:
    """Parse a memory query into an open subject/predicate lookup."""

    words = _word_tokens(toks)
    if len(words) >= 3 and words[0] == "where" and words[1] in {"is", "are"}:
        return ParsedQuery(
            subject=words[2],
            predicate="location",
            confidence=1.0,
            evidence={"parser": "where_query", "source_words": words},
        )
    if len(words) >= 3 and words[0] == "what" and words[1] in {"is", "are"}:
        return ParsedQuery(
            subject=words[2],
            predicate="attribute",
            confidence=0.8,
            evidence={"parser": "what_query", "source_words": words},
        )
    return None


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    den = (a.norm() * b.norm()).clamp_min(1e-12)
    return float(torch.dot(a.view(-1), b.view(-1)) / den)


def _text_vector(text: str, text_encoder: TextEncoder | None) -> torch.Tensor:
    if text_encoder is None:
        return stable_sketch(text)
    try:
        v = text_encoder(text)
    except Exception:
        v = stable_sketch(text)
    return v.detach().float().cpu().view(-1)


def _route_relevance(utterance: str, toks: Sequence[str], descriptors: Sequence[str], text_encoder: TextEncoder | None) -> float:
    """Continuous intent relevance plus lexical coverage from a route manifest."""

    descriptor_text = " ".join(descriptors)
    sem = max(0.0, _cosine(_text_vector(utterance, text_encoder), _text_vector(descriptor_text, text_encoder)))
    words = set(_word_tokens(toks))
    desc_words = set(_word_tokens(utterance_words(descriptor_text)))
    overlap = len(words & desc_words) / math.sqrt(max(1, len(words)) * max(1, len(desc_words)))
    return max(0.0, min(1.0, 0.65 * sem + 0.55 * overlap))


@dataclass
class CognitiveFrame:
    """A non-linguistic content packet for the Broca interface to express.

    ``intent`` is an open vocabulary routing label (built-ins like ``memory_location``
    name bundled demos; substrates may emit ``spatial_navigation``, ``einstein_bio``, …).

    ``to_features()`` maps arbitrary intent/subject/answer strings through frozen subword sketches;
    ``speech_plan()`` accepts ``evidence[\"speech_plan_words\"]`` overrides and falls back to
    templates that work for strings outside the toy demos.
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

        if self.intent == "memory_conflict" and self.subject:
            claimed = str(self.evidence.get("claimed_answer", "unknown")).replace("_", " ")
            current = self.answer.replace("_", " ")
            return ["conflict", self.subject, "was", current, "but", "you", "said", claimed, "."]
        if self.intent == "memory_write" and self.subject and self.answer != "unknown":
            return ["noted", self.subject, "is", "in", self.answer, "."]
        if self.intent == "memory_location" and self.subject and self.answer != "unknown":
            mu_thr = float(self.evidence.get("semantic_mean_confidence", 1.0))
            if self.confidence < mu_thr:
                return ["i", "should", "verify", "where", self.subject, "is", "."]
            return [self.subject, "is", "in", self.answer, "."]
        if self.intent == "prediction_error":
            return ["semantic", "prediction", "clashes", "with", "input", "."]
        if self.intent == "active_action":
            return _speech_plan_active_action(self.answer)
        if self.intent == "causal_effect":
            return _speech_plan_causal(self.answer)
        if self.intent == "synthesis_bundle" and self.answer != "unknown":
            return ["situated", "memory", "matches", "causal", "readout", "for", self.answer, "."]
        return _speech_plan_open_vocab(self)

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

    def upsert(self, subject: str, predicate: str, obj: str, *, confidence: float = 1.0, evidence: dict | None = None) -> None:
        now = time.time()
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO semantic_memory(namespace, subject, predicate, object, confidence, evidence_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(namespace, subject, predicate)
                DO UPDATE SET object=excluded.object, confidence=excluded.confidence,
                              evidence_json=excluded.evidence_json, updated_at=excluded.updated_at
                """,
                (self.namespace, subject.lower(), predicate.lower(), obj.lower(), float(confidence), json.dumps(evidence or {}, sort_keys=True), now, now),
            )

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
            return int(cur.lastrowid)

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
    ) -> int | None:
        now = time.time()
        with self._connect() as con:
            cur = con.execute(
                """
                INSERT OR IGNORE INTO memory_reflections(namespace, dedupe_key, kind, subject, predicate, summary, evidence_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.namespace,
                    dedupe_key,
                    kind,
                    subject.lower(),
                    predicate.lower(),
                    summary,
                    json.dumps(evidence, sort_keys=True),
                    now,
                ),
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
        revision_margin: float = BELIEF_REVISION_MARGIN,
        min_claims: int = BELIEF_REVISION_MIN_CLAIMS,
    ) -> list[dict]:
        claims = self.claims()
        grouped: dict[tuple[str, str], list[dict]] = {}
        for claim in claims:
            grouped.setdefault((claim["subject"], claim["predicate"]), []).append(claim)

        reflections: list[dict] = []
        for (subject, predicate), rows in grouped.items():
            if len({r["object"] for r in rows}) < 2:
                continue
            support: dict[str, dict[str, Any]] = {}
            for row in rows:
                entry = support.setdefault(row["object"], {"score": 0.0, "count": 0, "claim_ids": []})
                entry["score"] += float(row["confidence"])
                entry["count"] += 1
                entry["claim_ids"].append(int(row["id"]))

            current = self.get(subject, predicate)
            current_obj = current[0] if current is not None else ""
            current_score = float(support.get(current_obj, {}).get("score", 0.0))
            best_obj, best = max(support.items(), key=lambda item: (float(item[1]["score"]), int(item[1]["count"])))
            best_score = float(best["score"])
            best_count = int(best["count"])
            evidence = {
                "support": support,
                "current_object": current_obj,
                "candidate_object": best_obj,
                "revision_margin": float(revision_margin),
                "min_claims": int(min_claims),
                "instrument": "background_claim_consolidation",
            }

            if current_obj and best_obj != current_obj and best_count >= int(min_claims) and best_score >= current_score + float(revision_margin):
                dedupe = f"belief_revision:{subject}:{predicate}:{current_obj}->{best_obj}:{','.join(map(str, best['claim_ids']))}"
                reflection_id = self.record_reflection(
                    "belief_revision",
                    subject,
                    predicate,
                    f"revised {subject}.{predicate} from {current_obj} to {best_obj}",
                    evidence,
                    dedupe_key=dedupe,
                )
                if reflection_id is not None:
                    self.upsert(
                        subject,
                        predicate,
                        best_obj,
                        confidence=min(1.0, best_score / max(1.0, sum(float(v["score"]) for v in support.values()))),
                        evidence={**evidence, "reflection_id": reflection_id},
                    )
                    reflections.append({"id": reflection_id, "kind": "belief_revision", **evidence})
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
            return {"status": "accepted", "claim_id": claim_id, "current_object": observed_obj, "observed_object": observed_obj}

        current_obj, current_conf, current_ev = current
        if current_obj == observed_obj:
            claim_id = self.record_claim(subj, pred, observed_obj, confidence=confidence, status="corroborated", evidence=ev)
            merged_ev = merge_epistemic_evidence_dict(dict(current_ev), {**ev, "claim_id": claim_id, "claim_status": "corroborated"})
            self.upsert(subj, pred, observed_obj, confidence=max(float(current_conf), float(confidence)), evidence=merged_ev)
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

    def distinct_objects_for_predicate(self, predicate: str) -> frozenset[str]:
        """Known objects (e.g. city names) already stored — used for conflict detection without CITIES."""

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
            return int(cur.lastrowid)

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
    """When working memory simultaneously carries locality and causal readouts, bind provenance."""

    if len(working) < 2:
        return None
    if any(f.intent == "synthesis_bundle" for f in working[-6:]):
        return None
    loc = None
    ce = None
    for f in reversed(working):
        if loc is None and f.intent == "memory_location":
            loc = f
        if ce is None and f.intent == "causal_effect":
            ce = f
        if loc is not None and ce is not None:
            break
    if loc is None or ce is None:
        return None
    jids: list[int] = []
    for f in (loc, ce):
        jid = (f.evidence or {}).get("journal_id")
        if jid is not None:
            jids.append(int(jid))
    geo = math.sqrt(max(1e-12, float(loc.confidence)) * max(1e-12, float(ce.confidence)))
    return CognitiveFrame(
        "synthesis_bundle",
        subject=loc.subject,
        answer=loc.answer,
        confidence=min(1.0, geo),
        evidence={
            "episode_ids": jids,
            "instruments": ["working_memory_synthesis", "semantic_location", "scm_do_readout"],
            "causal_ate": ce.evidence.get("ate"),
            "source_intents": [loc.intent, ce.intent],
        },
    )


@dataclass
class IntrinsicCue:
    urgency: float
    faculty: str
    evidence: dict = field(default_factory=dict)


class GlobalWorkspace:
    """A tiny blackboard where non-language faculties publish latent frames."""

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
            self.frames.append(syn)
            self._trim_working()
        return frame

    @property
    def latest(self) -> CognitiveFrame | None:
        return self.frames[-1] if self.frames else None

    def snapshot(self) -> list[dict]:
        return [asdict(f) for f in self.frames]


class CognitiveBackgroundWorker:
    """In-process consolidation loop for persisted cognitive substrate state."""

    def __init__(self, mind: "BrocaMind", *, interval_s: float = 5.0):
        self.mind = mind
        self.interval_s = max(0.1, float(interval_s))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.iterations = 0
        self.last_error: str | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="broca-consolidator", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.0, float(timeout)))

    def run_once(self) -> list[dict]:
        out = self.mind.consolidate_once()
        self.iterations += 1
        self.last_error = None
        return out

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                self.run_once()
            except Exception as exc:  # pragma: no cover - background safety net
                self.last_error = repr(exc)


class LexicalPlanGraft(BaseGraft):
    """Writes a planned next word into the frozen host's residual stream.

    This is the cleanest Broca analogy in the lab: the cognitive substrate
    decides what is to be said; this graft turns the intended lexical sequence
    into hidden-state directions that the frozen language host can emit.
    """

    def __init__(self, *, strength: float = 28.0):
        super().__init__()
        self.strength = float(strength)
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
        out = x.clone()
        last = state["last_indices"].to(x.device)
        rows = torch.arange(x.shape[0], device=x.device)
        out[rows, last] += self.strength * directions
        self.last_token_id = int(target_ids[0].item())
        tok = getattr(state.get("tokenizer", None), "decode_id", None)
        self.last_token = tok(self.last_token_id) if callable(tok) else None
        return out


class TrainableBrocaGraft(BaseGraft):
    """Trainable bridge from latent cognitive frames to language tokens.

    The host transformer can remain frozen. This module learns how to project a
    faculty-state vector plus a production step into the residual stream.
    """

    def __init__(self, d_features: int, d_model: int, *, max_steps: int = 10, step_dim: int = 16, hidden: int = 160, strength: float = 1.0):
        super().__init__()
        self.d_features = int(d_features)
        self.max_steps = int(max_steps)
        self.strength = float(strength)
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
        delta = (self.net(z) * self.strength).to(device=x.device, dtype=x.dtype)
        out = x.clone()
        last = state["last_indices"].to(x.device)
        rows = torch.arange(x.shape[0], device=x.device)
        out[rows, last] += delta
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


def default_lexical_strength(model: nn.Module) -> float:
    cfg = getattr(model, "cfg", None)
    d_model = float(getattr(cfg, "d_model", 96))
    return 30.0 * (d_model / 96.0) ** 0.5


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

    action_descriptors = (
        "action",
        "take",
        "choose",
        "decide",
        "should",
        "policy",
        "listen",
        "open",
        "expected free energy",
    )
    causal_descriptors = (
        "causal",
        "effect",
        "intervention",
        "treatment",
        "help",
        "helps",
        "hurt",
        "do calculus",
        "ate",
    )

    def __init__(self, *, relevance_floor: float = 0.28):
        self.relevance_floor = float(relevance_floor)

    def route(self, mind: "BrocaMind", utterance: str, toks: Sequence[str]) -> CognitiveFrame:
        candidates: list[FacultyCandidate] = []
        claim = _claim_from_tokens(toks)
        query = _query_from_tokens(toks)

        if claim is not None:
            candidates.append(FacultyCandidate("semantic_claim", 1.45, lambda claim=claim: self._memory_write(mind, utterance, claim)))
        if query is not None:
            candidates.append(FacultyCandidate("semantic_query", 1.35, lambda query=query: self._memory_query(mind, utterance, toks, query)))

        active_score = 0.1 + _route_relevance(utterance, toks, self.action_descriptors, mind.text_encoder)
        candidates.append(FacultyCandidate("active_inference", active_score, lambda: self._active_action(mind)))

        causal_score = 0.1 + _route_relevance(utterance, toks, self.causal_descriptors, mind.text_encoder)
        candidates.append(FacultyCandidate("causal_effect", causal_score, lambda: self._causal_effect(mind)))

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
        return frame

    def _memory_write(self, mind: "BrocaMind", utterance: str, claim: ParsedClaim) -> CognitiveFrame:
        ev = {
            "source": "observed_utterance",
            "utterance": utterance,
            "predicate": claim.predicate,
            "instruments": ["runtime_observation"],
            **dict(claim.evidence),
        }
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
        rec = mind.memory.get(query.subject, query.predicate) if query.subject else None
        if rec is None:
            return CognitiveFrame(
                "unknown",
                subject=query.subject,
                answer="unknown",
                confidence=0.0,
                evidence={"missing": "semantic_memory", "predicate": query.predicate, **dict(query.evidence)},
            )

        obj, conf, ev = rec
        frame = CognitiveFrame("memory_location" if query.predicate == "location" else "memory_lookup", subject=query.subject, answer=obj, confidence=conf, evidence=dict(ev))
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
        return CognitiveFrame(
            "causal_effect",
            subject="treatment",
            answer="helps" if ate >= 0 else "hurts",
            confidence=float(min(1.0, abs(ate))),
            evidence={"p_do_positive": p1, "p_do_negative": p0, "ate": ate},
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
        lexical_strength: float | None = None,
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
        graft_strength = lexical_strength if lexical_strength is not None else default_lexical_strength(self.host)
        self.lexical_graft = LexicalPlanGraft(strength=graft_strength)
        self.host.add_graft("final_hidden", self.lexical_graft)
        self.feature_graft = TrainableBrocaGraft(
            COGNITIVE_FRAME_DIM,
            int(getattr(self.host.cfg, "d_model", 96)),
            strength=0.1,
        )
        params = getattr(self.host, "parameters", None)
        if callable(params):
            host_param = next(params(), None)
            if host_param is not None:
                self.feature_graft.to(host_param.device)
        self.host.add_graft("final_hidden", self.feature_graft)
        self.workspace = GlobalWorkspace()
        self.router = CognitiveRouter()
        self.pomdp = build_tiger_pomdp()
        self.active_agent = ActiveInferenceAgent(self.pomdp, horizon=1, learn=False)
        self.scm = build_simpson_scm()
        self.causal_pomdp = build_causal_epistemic_pomdp(self.scm)
        self.causal_agent = ActiveInferenceAgent(self.causal_pomdp, horizon=1, learn=False)
        self.unified_agent = CoupledEFEAgent(self.active_agent, self.causal_agent)
        self._background_worker: CognitiveBackgroundWorker | None = None

    @property
    def background_worker(self) -> CognitiveBackgroundWorker | None:
        return self._background_worker

    def consolidate_once(self) -> list[dict]:
        return self.memory.consolidate_claims_once()

    def start_background(self, *, interval_s: float = 5.0) -> CognitiveBackgroundWorker:
        if self._background_worker is None:
            self._background_worker = CognitiveBackgroundWorker(self, interval_s=interval_s)
        else:
            self._background_worker.interval_s = max(0.1, float(interval_s))
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
        for ent in self.memory.subjects_for_predicate("location"):
            if ent not in toks_set:
                continue
            rec = self.memory.get(ent, "location")
            if rec is None:
                self.workspace.intrinsic_cues.append(IntrinsicCue(1.0, "memory_gap", {"subject": ent}))
            elif rec[1] < confidence_floor:
                self.workspace.intrinsic_cues.append(
                    IntrinsicCue(float(confidence_floor - rec[1]), "memory_low_confidence", {"subject": ent, "confidence": rec[1]})
                )
        cq = self.causal_agent.qs
        if cq is not None and len(cq) >= 2:
            max_ent = math.log(len(cq))
            h_q = belief_entropy(cq)
            if max_ent > 1e-9 and h_q > 0.5 * max_ent:
                self.workspace.intrinsic_cues.append(IntrinsicCue(float(h_q / max_ent), "causal_uncertain", {"entropy": h_q}))

    def comprehend(self, utterance: str) -> CognitiveFrame:
        toks = utterance_words(utterance)
        self._intrinsic_scan(toks)
        frame = self.router.route(self, utterance, toks)
        return self._commit_frame(utterance, toks, frame)

    def _commit_frame(self, utterance: str, toks: Sequence[str], frame: CognitiveFrame) -> CognitiveFrame:
        jid = self.journal.append(utterance, frame)
        frame.evidence = {**frame.evidence, "journal_id": jid}
        if self._last_journal_id is not None:
            self.episode_graph.bump(self._last_journal_id, jid)
        self._last_journal_id = jid
        if frame.intent == "prediction_error":
            pred = str(frame.evidence.get("predicate", "location"))
            known_objects = self.memory.distinct_objects_for_predicate(pred)
            observed_objects = [t for t in toks if t in known_objects]
            if observed_objects and frame.subject:
                self.memory.upsert(
                    frame.subject,
                    pred,
                    observed_objects[-1],
                    confidence=1.0,
                    evidence={"journal_id": jid, "resolver": "prediction_error"},
                )
        out = self.workspace.publish(frame)
        for tail in self.workspace.frames:
            if tail.intent == "synthesis_bundle" and tail.subject:
                self.memory.merge_epistemic_evidence(tail.subject, "location", tail.evidence)
        return out

    def retrieve_episode(self, episode_id: int) -> CognitiveFrame:
        """Reload a prior workspace episode into working memory (persistent episodic retrieval)."""

        row = self.journal.fetch(episode_id)
        if row is None:
            return CognitiveFrame(
                "unknown",
                answer="unknown",
                confidence=0.0,
                evidence={"missing_episode_id": int(episode_id)},
            )
        replay = cognitive_frame_from_episode_row(row)
        self.workspace.publish(replay)
        return replay

    def speak(self, frame: CognitiveFrame) -> str:
        return generate_from_plan(
            self.host,
            self.tokenizer,
            frame.speech_plan(),
            broca_features=frame.to_features(self.text_encoder),
        )

    def answer(self, utterance: str) -> tuple[CognitiveFrame, str]:
        frame = self.comprehend(utterance)
        return frame, self.speak(frame)
