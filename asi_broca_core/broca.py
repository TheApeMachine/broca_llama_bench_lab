
from __future__ import annotations

import json
import math
import os
import random
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

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
from .continuous_frame import COGNITIVE_FRAME_DIM, pack_cognitive_frame
from .device_utils import pick_torch_device
from .grafts import BaseGraft
from .hf_tokenizer_compat import HuggingFaceBrocaTokenizer
from .host import count_parameters
from .llama_broca_host import LlamaBrocaHost, load_llama_broca_host
from .predictive_coding import lexical_surprise_gap
from .substrate_graph import EpisodeAssociationGraph, merge_epistemic_evidence_dict
from .tokenizer import SPEECH_BRIDGE_PREFIX, utterance_words


DEFAULT_BROCA_MODEL_ID = os.environ.get("ASI_BROCA_MODEL_ID", "meta-llama/Llama-3.2-1B-Instruct")

# Teacher-forcing / smoke demos only — pass explicit ``facts`` to ``seed_locations`` for anything real.
_DEMO_LOCATION_PAIRS: tuple[tuple[str, str], ...] = (
    ("ada", "rome"),
    ("byron", "paris"),
    ("curie", "tokyo"),
    ("darwin", "lima"),
    ("euclid", "oslo"),
    ("faraday", "cairo"),
    ("gauss", "vienna"),
    ("hopper", "lisbon"),
)


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


FEATURE_DIM = COGNITIVE_FRAME_DIM


@dataclass
class CognitiveFrame:
    """A non-linguistic content packet for the Broca interface to express.

    ``intent`` is an open vocabulary routing label (built-ins like ``memory_location``
    name bundled demos; substrates may emit ``spatial_navigation``, ``einstein_bio``, …).

    ``to_features()`` maps arbitrary intent/subject/answer strings through hashed sketches;
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

    def to_features(self) -> torch.Tensor:
        """Sketch-hash bottleneck over intent/subject/answer + numeric faculty scalars."""

        return pack_cognitive_frame(self.intent, self.subject, self.answer, float(self.confidence), self.evidence)


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

    def seed_locations(self, facts: Sequence[tuple[str, str]] | None = None) -> None:
        if facts is None:
            facts = _DEMO_LOCATION_PAIRS
        for name, city in facts:
            self.upsert(name, "location", city, confidence=1.0, evidence={"source": "seed_fact", "instruments": ["seed_schema"]})

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


class LexicalPlanGraft(BaseGraft):
    """Writes a planned next word into the frozen host's residual stream.

    This is the cleanest Broca analogy in the lab: the cognitive substrate
    decides what is to be said; this graft turns the intended lexical sequence
    into hidden-state directions that the frozen language host can emit.
    """

    def __init__(self, *, strength: float = 28.0):
        super().__init__()
        self.strength = float(strength)
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
        feats = feats.to(x.device, x.dtype)
        if feats.ndim == 1:
            feats = feats.view(1, -1).expand(x.shape[0], -1)
        if feats.shape[-1] != self.d_features:
            raise ValueError(f"expected broca_features dim {self.d_features}, got {feats.shape[-1]}")
        step = state.get("broca_step", torch.zeros(x.shape[0], device=x.device, dtype=torch.long))
        if not isinstance(step, torch.Tensor):
            step = torch.full((x.shape[0],), int(step), device=x.device, dtype=torch.long)
        step = step.to(x.device).long().view(-1).clamp(0, self.max_steps - 1)
        z = torch.cat([self.norm(feats), self.step_emb(step).to(x.dtype)], dim=-1)
        delta = (self.net(z) * self.strength).to(dtype=x.dtype)
        out = x.clone()
        last = state["last_indices"].to(x.device)
        rows = torch.arange(x.shape[0], device=x.device)
        out[rows, last] += delta
        return out


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


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
) -> str:
    plan_ids = list(tokenizer.encode_plan_words(plan_tokens))
    max_new_tokens = max_new_tokens or len(plan_ids)
    ids = tokenizer.encode(prefix if prefix is not None else SPEECH_BRIDGE_PREFIX)
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
            },
        )
        pred = int(logits[0, mask.long().sum().item() - 1].argmax().item())
        generated.append(pred)
    return decode_generation(tokenizer, generated)


def generate_from_features(model: nn.Module, tokenizer: Any, features: torch.Tensor, *, prefix: str | None = None, max_new_tokens: int = 32) -> str:
    ids = tokenizer.encode(prefix if prefix is not None else SPEECH_BRIDGE_PREFIX)
    generated: list[int] = []
    device = next(model.parameters()).device
    feats = features.to(device).float().view(1, -1)
    for step in range(max_new_tokens):
        row = ids + generated
        batch_ids, mask, _ = _batch_from_ids([row], tokenizer.pad_id, device=device)
        logits = model(batch_ids, mask, extra_state={"broca_features": feats, "broca_step": torch.tensor([step], device=device)})
        pred = int(logits[0, mask.long().sum().item() - 1].argmax().item())
        generated.append(pred)
        if decode_generation(tokenizer, generated).rstrip().endswith("."):
            break
    return decode_generation(tokenizer, generated)


def generate_without_broca(model: nn.Module, tokenizer: Any, *, prefix: str | None = None, max_new_tokens: int = 5) -> str:
    ids = tokenizer.encode(prefix if prefix is not None else SPEECH_BRIDGE_PREFIX)
    generated: list[int] = []
    device = next(model.parameters()).device
    for _ in range(max_new_tokens):
        row = ids + generated
        batch_ids, mask, _ = _batch_from_ids([row], tokenizer.pad_id, device=device)
        logits = model(batch_ids, mask)
        pred = int(logits[0, mask.long().sum().item() - 1].argmax().item())
        generated.append(pred)
    return decode_generation(tokenizer, generated)


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
        self.host, self.tokenizer = load_llama_broca_host(mid, device=resolved_device, token=hf_token)
        graft_strength = lexical_strength if lexical_strength is not None else default_lexical_strength(self.host)
        self.lexical_graft = LexicalPlanGraft(strength=graft_strength)
        self.host.add_graft("final_hidden", self.lexical_graft)
        self.memory = PersistentSemanticMemory(db_path, namespace=namespace)
        if self.memory.count() == 0:
            self.memory.seed_locations()
        self.journal = WorkspaceJournal(Path(db_path))
        self.episode_graph = EpisodeAssociationGraph(Path(db_path))
        self._last_journal_id: int | None = None
        self.workspace = GlobalWorkspace()
        self.pomdp = build_tiger_pomdp()
        self.active_agent = ActiveInferenceAgent(self.pomdp, horizon=1, learn=False)
        self.scm = build_simpson_scm()
        self.causal_pomdp = build_causal_epistemic_pomdp(self.scm)
        self.causal_agent = ActiveInferenceAgent(self.causal_pomdp, horizon=1, learn=False)
        self.unified_agent = CoupledEFEAgent(self.active_agent, self.causal_agent)

    def _intrinsic_scan(self, toks: list[str]) -> None:
        self.workspace.intrinsic_cues.clear()
        mu_pop = self.memory.mean_confidence()
        toks_set = set(toks)
        for ent in self.memory.subjects_for_predicate("location"):
            if ent not in toks_set:
                continue
            rec = self.memory.get(ent, "location")
            if rec is None:
                self.workspace.intrinsic_cues.append(IntrinsicCue(1.0, "memory_gap", {"subject": ent}))
            elif mu_pop is not None and rec[1] < mu_pop:
                self.workspace.intrinsic_cues.append(
                    IntrinsicCue(float(mu_pop - rec[1]), "memory_low_confidence", {"subject": ent, "confidence": rec[1]})
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
        frame: CognitiveFrame
        if "where" in toks and "is" in toks:
            try:
                subject = toks[toks.index("is") + 1]
            except Exception:
                subject = ""
            rec = self.memory.get(subject, "location") if subject else None
            if rec:
                obj, conf, ev = rec
                frame = CognitiveFrame("memory_location", subject=subject, answer=obj, confidence=conf, evidence=dict(ev))
                mu_pop = self.memory.mean_confidence()
                if mu_pop is not None:
                    frame.evidence["semantic_mean_confidence"] = mu_pop
                known_locations = self.memory.distinct_objects_for_predicate("location")
                mentioned_locations = [t for t in toks if t in known_locations]
                conflicting = bool(mentioned_locations and mentioned_locations[-1] != obj.lower())
                if conflicting:
                    plan_words = frame.speech_plan()
                    ce_g, ce_p, gap = lexical_surprise_gap(self.host, self.tokenizer, utterance=utterance, plan_words=plan_words)
                    frame.evidence["prediction_ce_graft"] = ce_g
                    frame.evidence["prediction_ce_plain"] = ce_p
                    frame.evidence["prediction_gap"] = gap
                    if gap > 0.0:
                        coupled = self.unified_agent.decide()
                        frame = CognitiveFrame(
                            "prediction_error",
                            subject=subject,
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
            else:
                frame = CognitiveFrame("unknown", subject=subject, answer="unknown", confidence=0.0, evidence={"missing": "semantic_memory"})
        elif "action" in toks or "take" in toks:
            coupled = self.unified_agent.decide()
            posterior_spatial = {
                self.pomdp.action_names[i]: float(p)
                for i, p in enumerate(coupled.spatial_decision.posterior_over_policies[: len(self.pomdp.action_names)])
            }
            causal_names = self.causal_pomdp.action_names
            posterior_causal = {
                causal_names[i]: float(p)
                for i, p in enumerate(coupled.causal_decision.posterior_over_policies[: len(causal_names)])
            }
            conf = (
                max(coupled.spatial_decision.posterior_over_policies)
                if coupled.faculty == "spatial"
                else max(coupled.causal_decision.posterior_over_policies)
            )
            frame = CognitiveFrame(
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
        elif "treatment" in toks or "help" in toks or "helps" in toks:
            p1 = self.scm.probability({"Y": 1}, interventions={"T": 1})
            p0 = self.scm.probability({"Y": 1}, interventions={"T": 0})
            ate = p1 - p0
            frame = CognitiveFrame(
                "causal_effect",
                subject="treatment",
                answer="helps" if ate >= 0 else "hurts",
                confidence=float(min(1.0, abs(ate))),
                evidence={"p_do_positive": p1, "p_do_negative": p0, "ate": ate},
            )
        else:
            frame = CognitiveFrame("unknown", answer="unknown", confidence=0.0, evidence={"route": "none"})
        jid = self.journal.append(utterance, frame)
        frame.evidence = {**frame.evidence, "journal_id": jid}
        if self._last_journal_id is not None:
            self.episode_graph.bump(self._last_journal_id, jid)
        self._last_journal_id = jid
        if frame.intent == "prediction_error":
            known_locations = self.memory.distinct_objects_for_predicate("location")
            cities_in = [t for t in toks if t in known_locations]
            if cities_in and frame.subject:
                self.memory.upsert(
                    frame.subject,
                    "location",
                    cities_in[-1],
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
        return generate_from_plan(self.host, self.tokenizer, frame.speech_plan())

    def answer(self, utterance: str) -> tuple[CognitiveFrame, str]:
        frame = self.comprehend(utterance)
        return frame, self.speak(frame)


def build_training_frames() -> list[CognitiveFrame]:
    frames: list[CognitiveFrame] = [
        CognitiveFrame("memory_location", subject=name, answer=city, confidence=1.0) for name, city in _DEMO_LOCATION_PAIRS
    ]
    frames.extend(
        [
            CognitiveFrame("active_action", answer="listen", confidence=0.72, evidence={"policy_posterior": {"listen": 0.72, "open_left": 0.14, "open_right": 0.14}}),
            CognitiveFrame("active_action", answer="open_left", confidence=0.92, evidence={"policy_posterior": {"listen": 0.04, "open_left": 0.92, "open_right": 0.04}}),
            CognitiveFrame("active_action", answer="open_right", confidence=0.91, evidence={"policy_posterior": {"listen": 0.05, "open_left": 0.04, "open_right": 0.91}}),
            CognitiveFrame("causal_effect", subject="treatment", answer="helps", confidence=0.9, evidence={"p_do_positive": 0.55, "p_do_negative": 0.45, "ate": 0.10}),
            CognitiveFrame("causal_effect", subject="treatment", answer="hurts", confidence=0.9, evidence={"p_do_positive": 0.35, "p_do_negative": 0.55, "ate": -0.20}),
            CognitiveFrame(
                "synthesis_bundle",
                subject="ada",
                answer="rome",
                confidence=0.85,
                evidence={
                    "episode_ids": [1, 2],
                    "instruments": ["working_memory_synthesis", "semantic_location", "scm_do_readout"],
                    "causal_ate": 0.1,
                },
            ),
            CognitiveFrame("prediction_error", subject="ada", answer="rome", confidence=0.55, evidence={"delta_ce": 0.4}),
            CognitiveFrame("unknown", answer="unknown", confidence=0.0),
        ]
    )
    return frames


def broca_canonical_eval_queries() -> list[str]:
    """Smoke queries for bundled comprehension demos (_DEMO_LOCATION_PAIRS + routing tuples)."""

    return [
        f"where is {_DEMO_LOCATION_PAIRS[0][0]} ?",
        " ".join(("what", "action", "should", "i", "take", "?")),
        " ".join(("does", "treatment", "help", "?")),
    ]


def _broca_teacher_forcing_dataset(tokenizer: Any, frames: Sequence[CognitiveFrame]) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[tuple[str, str]],
    int,
]:
    rows: list[list[int]] = []
    feats: list[torch.Tensor] = []
    steps_list: list[int] = []
    targets: list[int] = []
    examples: list[tuple[str, str]] = []
    prefix = tokenizer.encode(SPEECH_BRIDGE_PREFIX)
    max_step_seen = 0
    for frame in frames:
        plan = [str(tok).lower() for tok in frame.speech_plan()]
        plan_ids = list(tokenizer.encode_plan_words(plan))
        for j, target_id in enumerate(plan_ids):
            rows.append(prefix + plan_ids[:j])
            feats.append(frame.to_features())
            steps_list.append(j)
            targets.append(target_id)
            max_step_seen = max(max_step_seen, j)
        examples.append((" ".join(plan), frame.intent))
    bridge_max_steps = max(8, max_step_seen + 1)
    ids, mask, lengths = _batch_from_ids(rows, tokenizer.pad_id)
    return (
        ids,
        mask,
        lengths,
        torch.stack(feats),
        torch.tensor(steps_list, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long),
        examples,
        bridge_max_steps,
    )


def train_broca_bridge(
    seed: int = 0,
    steps: int = 80,
    *,
    llama_model_id: str | None = None,
    device: torch.device | str | None = None,
    hf_token: str | bool | None = None,
) -> dict:
    seed_everything(seed)
    resolved_device = device if isinstance(device, torch.device) else pick_torch_device(device)
    mid = llama_model_id or DEFAULT_BROCA_MODEL_ID
    model, tokenizer = load_llama_broca_host(mid, device=resolved_device, token=hf_token)
    frames = build_training_frames()
    ids, mask, lengths, feats, step_ids, targets, _examples, bridge_max_steps = _broca_teacher_forcing_dataset(tokenizer, frames)

    bridge = TrainableBrocaGraft(FEATURE_DIM, model.cfg.d_model, max_steps=bridge_max_steps, strength=1.0)
    model.add_graft("final_hidden", bridge)
    for par in bridge.parameters():
        par.requires_grad = True

    dev = next(model.parameters()).device
    ids = ids.to(dev)
    mask = mask.to(dev)
    lengths = lengths.to(dev)
    feats = feats.to(dev, dtype=torch.float32)
    step_ids = step_ids.to(dev)
    targets = targets.to(dev)

    bridge.to(device=dev)
    last = lengths - 1

    def eval_teacher() -> tuple[float, list[str]]:
        with torch.no_grad():
            logits = model(ids, mask, extra_state={"broca_features": feats, "broca_step": step_ids})
            pred_ids = logits[torch.arange(ids.shape[0], device=dev), last].argmax(dim=-1)
            acc = float((pred_ids == targets).float().mean().item())
            return acc, [tokenizer.decode_id(int(x)) for x in pred_ids[:12]]

    before, before_sample = eval_teacher()
    opt = torch.optim.AdamW(bridge.parameters(), lr=0.045, weight_decay=0.0)
    final_loss = 0.0
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        logits = model(ids, mask, extra_state={"broca_features": feats, "broca_step": step_ids})
        loss = F.cross_entropy(logits[torch.arange(ids.shape[0], device=dev), last], targets)
        loss.backward()
        opt.step()
        final_loss = float(loss.detach().item())
    after, after_sample = eval_teacher()

    targets_text = [" ".join(f.speech_plan()) for f in frames[:5]]
    max_lens = [
        len(tokenizer.encode_plan_words([str(t).lower() for t in f.speech_plan()])) + 4 for f in frames[:5]
    ]
    generated = [
        generate_from_features(model, tokenizer, f.to_features(), max_new_tokens=min(32, int(max_lens[i])))
        for i, f in enumerate(frames[:5])
    ]

    total, trainable = count_parameters(model)
    return {
        "model": model,
        "tokenizer": tokenizer,
        "before_accuracy": before,
        "after_accuracy": after,
        "before_sample": before_sample,
        "after_sample": after_sample,
        "generated": generated,
        "targets": targets_text,
        "final_loss": final_loss,
        "total_params": total,
        "trainable_params": trainable,
        "bridge_params": sum(p.numel() for p in bridge.parameters()),
    }


def run_broca_experiment(
    seed: int = 0,
    db_path: str | Path = "runs/broca_semantic_memory.sqlite",
    verbose: bool = True,
    *,
    llama_model_id: str | None = None,
    device: torch.device | str | None = None,
    hf_token: str | bool | None = None,
    train_bridge: bool = True,
    train_bridge_steps: int = 80,
) -> dict:
    path = Path(db_path)
    if path.exists():
        path.unlink()
    for suffix in ("-wal", "-shm"):
        sp = Path(str(path) + suffix)
        if sp.exists():
            sp.unlink()

    mind = BrocaMind(
        seed=seed,
        db_path=path,
        namespace=f"broca_{seed}",
        llama_model_id=llama_model_id,
        device=device,
        hf_token=hf_token,
    )
    total, trainable = count_parameters(mind.host)
    language_only = generate_without_broca(mind.host, mind.tokenizer, prefix=None, max_new_tokens=5)
    queries = broca_canonical_eval_queries()
    rows: list[dict] = []
    for q in queries:
        frame, utterance = mind.answer(q)
        rows.append({"query": q, "intent": frame.intent, "latent_answer": frame.answer, "speech": utterance, "evidence": frame.evidence})

    train_result: dict = {}
    if train_bridge:
        train_result = train_broca_bridge(
            seed=seed,
            steps=train_bridge_steps,
            llama_model_id=llama_model_id,
            device=device,
            hf_token=hf_token,
        )

    trainable_meta = {k: v for k, v in train_result.items() if k not in {"model", "tokenizer"}} if train_result else {}

    result = {
        "host_params": total,
        "host_trainable_params": trainable,
        "semantic_records": mind.memory.count(),
        "language_only": language_only,
        "rows": rows,
        "trainable_broca": trainable_meta,
        "model_id": llama_model_id or DEFAULT_BROCA_MODEL_ID,
    }

    if verbose:
        print("\n=== 6) LLM-as-Broca architecture ===")
        print("The language host is treated as a speech/planning interface. Memory, action selection, and causality live outside it and publish latent frames.")
        print(f"frozen Broca host params={total:,}; trainable host params before trainable bridge={trainable:,}")
        print(f"persistent semantic memory records={mind.memory.count()}; db={path}")
        print("\nLanguage host with no Broca/faculty state:")
        print(f"  {SPEECH_BRIDGE_PREFIX} -> {language_only}")
        print("\nCognitive substrate -> Broca verbalization:")
        print(f"{'query':<32} {'intent':<17} {'latent':<10} speech")
        print(f"{'-'*32} {'-'*17} {'-'*10} {'-'*34}")
        for r in rows:
            print(f"{r['query']:<32} {r['intent']:<17} {r['latent_answer']:<10} {r['speech']}")
        if train_bridge and train_result:
            print("\nTrainable Broca bridge:")
            print("  frozen host + trainable frame-to-residual graft, trained by teacher forcing on semantic frames")
            print(f"  bridge params={train_result['bridge_params']:,}; trainable total={train_result['trainable_params']:,}; final_loss={train_result['final_loss']:.4f}")
            print(f"  teacher-forced token accuracy before={train_result['before_accuracy']:.3f}; after={train_result['after_accuracy']:.3f}")
            for target, gen in zip(train_result["targets"], train_result["generated"]):
                ok = "✓" if target == gen else "·"
                print(f"  {ok} target='{target}' generated='{gen}'")
        elif not train_bridge:
            print("\nTrainable Broca bridge: skipped (--no-train-bridge)")

    return result


