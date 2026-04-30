
from __future__ import annotations

import json
import random
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .active_inference import ActiveInferenceAgent, build_tiger_pomdp
from .causal import build_simpson_scm
from .device_utils import pick_torch_device
from .grafts import BaseGraft
from .host import TinyCausalTransformer, TinyConfig, count_parameters, freeze_module
from .tokenizer import RegexTokenizer


BROCA_FACTS: list[tuple[str, str]] = [
    ("ada", "rome"),
    ("byron", "paris"),
    ("curie", "tokyo"),
    ("darwin", "lima"),
    ("euclid", "oslo"),
    ("faraday", "cairo"),
    ("gauss", "vienna"),
    ("hopper", "lisbon"),
]

INTENTS = ["memory_location", "active_action", "causal_effect", "unknown"]
ENTITIES = [name for name, _ in BROCA_FACTS]
CITIES = [city for _, city in BROCA_FACTS]
ACTIONS = ["listen", "open_left", "open_right"]
CAUSAL_WORDS = ["helps", "hurts"]
VALUES = CITIES + ACTIONS + CAUSAL_WORDS + ["unknown"]
NUMERIC_FEATURES = [
    "confidence",
    "p_do_positive",
    "p_do_negative",
    "ate",
    "policy_listen",
    "policy_open_left",
    "policy_open_right",
    "bias",
]
FEATURE_DIM = len(INTENTS) + len(ENTITIES) + len(VALUES) + len(NUMERIC_FEATURES)


@dataclass
class CognitiveFrame:
    """A non-linguistic content packet for the Broca interface to express."""

    intent: str
    subject: str = ""
    answer: str = "unknown"
    confidence: float = 1.0
    evidence: dict = field(default_factory=dict)

    def speech_plan(self) -> list[str]:
        if self.intent == "memory_location" and self.subject and self.answer != "unknown":
            return [self.subject, "is", "in", self.answer, "."]
        if self.intent == "active_action" and self.answer in ACTIONS:
            if self.answer == "listen":
                return ["i", "should", "listen", "first", "."]
            return ["i", "should", self.answer, "."]
        if self.intent == "causal_effect" and self.answer in CAUSAL_WORDS:
            return ["intervention", "says", "treatment", self.answer, "."]
        return ["i", "do", "not", "know", "."]

    def to_features(self) -> torch.Tensor:
        vec: list[float] = []
        vec.extend(1.0 if self.intent == x else 0.0 for x in INTENTS)
        vec.extend(1.0 if self.subject == x else 0.0 for x in ENTITIES)
        vec.extend(1.0 if self.answer == x else 0.0 for x in VALUES)
        ev = self.evidence or {}
        policy = ev.get("policy_posterior", {}) or {}
        nums = [
            float(self.confidence),
            float(ev.get("p_do_positive", 0.0)),
            float(ev.get("p_do_negative", 0.0)),
            float(ev.get("ate", 0.0)),
            float(policy.get("listen", 0.0)),
            float(policy.get("open_left", 0.0)),
            float(policy.get("open_right", 0.0)),
            1.0,
        ]
        vec.extend(nums)
        return torch.tensor(vec, dtype=torch.float32)


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

    def count(self) -> int:
        with self._connect() as con:
            row = con.execute("SELECT COUNT(*) FROM semantic_memory WHERE namespace=?", (self.namespace,)).fetchone()
        return int(row[0])

    def seed_locations(self, facts: Sequence[tuple[str, str]] = BROCA_FACTS) -> None:
        for name, city in facts:
            self.upsert(name, "location", city, confidence=1.0, evidence={"source": "seed_fact"})


class GlobalWorkspace:
    """A tiny blackboard where non-language faculties publish latent frames."""

    def __init__(self):
        self.frames: list[CognitiveFrame] = []

    def publish(self, frame: CognitiveFrame) -> CognitiveFrame:
        self.frames.append(frame)
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


def build_broca_tokenizer() -> RegexTokenizer:
    texts: list[str] = [
        "speak :",
        "where is ada ?",
        "what action should i take ?",
        "does treatment help ?",
        "i do not know .",
        "i should listen first .",
        "i should open_left .",
        "i should open_right .",
        "intervention says treatment helps .",
        "intervention says treatment hurts .",
    ]
    for name, city in BROCA_FACTS:
        texts.extend([f"where is {name} ?", f"{name} is in {city} .", name, city])
    extra = [
        "answer", "user", "broca", "workspace", "memory", "location", "causal", "effect",
        "belief", "policy", "latent", "frame", "language", "module", "unknown",
        "listen", "open_left", "open_right", "helps", "hurts", "first", "intervention",
        "says", "treatment", "do", "not", "know",
    ]
    return RegexTokenizer.fit(texts, extra_tokens=extra)


def make_broca_host(tokenizer: RegexTokenizer, seed: int = 0, *, d_model: int = 96) -> TinyCausalTransformer:
    seed_everything(seed)
    cfg = TinyConfig(vocab_size=len(tokenizer), d_model=d_model, n_layers=2, n_heads=4, d_ff=2 * d_model, max_seq_len=64, dropout=0.0)
    model = TinyCausalTransformer(cfg)
    model.eval()
    freeze_module(model)
    return model


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
    prefix: str = "speak :",
    max_new_tokens: int | None = None,
) -> str:
    plan_ids = list(tokenizer.encode_plan_words(plan_tokens))
    max_new_tokens = max_new_tokens or len(plan_ids)
    ids = tokenizer.encode(prefix)
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


def generate_from_features(model: nn.Module, tokenizer: Any, features: torch.Tensor, *, prefix: str = "speak :", max_new_tokens: int = 32) -> str:
    ids = tokenizer.encode(prefix)
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


def generate_without_broca(model: nn.Module, tokenizer: Any, *, prefix: str = "speak :", max_new_tokens: int = 5) -> str:
    ids = tokenizer.encode(prefix)
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

    host: TinyCausalTransformer | Any

    def __init__(
        self,
        *,
        seed: int = 0,
        db_path: str | Path = "runs/broca_semantic_memory.sqlite",
        namespace: str = "main",
        backend: str = "tiny",
        llama_model_id: str = "meta-llama/Llama-3.2-1B-Instruct",
        device: torch.device | str | None = None,
        hf_token: str | bool | None = None,
        lexical_strength: float | None = None,
    ):
        self.seed = seed
        self.backend = backend
        if backend == "llama":
            from .llama_broca_host import load_llama_broca_host

            resolved_device = device if isinstance(device, torch.device) else pick_torch_device(device)
            self.host, self.tokenizer = load_llama_broca_host(llama_model_id, device=resolved_device, token=hf_token)
        elif backend == "tiny":
            self.tokenizer = build_broca_tokenizer()
            self.host = make_broca_host(self.tokenizer, seed=seed)
        else:
            raise ValueError(f"unknown Broca backend {backend!r}; expected 'tiny' or 'llama'")
        graft_strength = lexical_strength if lexical_strength is not None else default_lexical_strength(self.host)
        self.lexical_graft = LexicalPlanGraft(strength=graft_strength)
        self.host.add_graft("final_hidden", self.lexical_graft)
        self.memory = PersistentSemanticMemory(db_path, namespace=namespace)
        if self.memory.count() == 0:
            self.memory.seed_locations()
        self.workspace = GlobalWorkspace()
        self.pomdp = build_tiger_pomdp(reliability=0.85)
        self.active_agent = ActiveInferenceAgent(self.pomdp, horizon=1, precision=8.0, learn=False)
        self.scm = build_simpson_scm()

    def comprehend(self, utterance: str) -> CognitiveFrame:
        toks = RegexTokenizer.tokenize(utterance)
        frame: CognitiveFrame
        if "where" in toks and "is" in toks:
            try:
                subject = toks[toks.index("is") + 1]
            except Exception:
                subject = ""
            rec = self.memory.get(subject, "location") if subject else None
            if rec:
                obj, conf, ev = rec
                frame = CognitiveFrame("memory_location", subject=subject, answer=obj, confidence=conf, evidence=ev)
            else:
                frame = CognitiveFrame("unknown", subject=subject, answer="unknown", confidence=0.0, evidence={"missing": "semantic_memory"})
        elif "action" in toks or "take" in toks:
            decision = self.active_agent.decide()
            posterior = {self.pomdp.action_names[i]: float(p) for i, p in enumerate(decision.posterior_over_policies[: len(self.pomdp.action_names)])}
            frame = CognitiveFrame(
                "active_action",
                answer=decision.action_name,
                confidence=float(max(decision.posterior_over_policies)),
                evidence={"expected_free_energy": min(ev.expected_free_energy for ev in decision.policies), "policy_posterior": posterior},
            )
        elif "treatment" in toks or "help" in toks or "helps" in toks:
            p1 = self.scm.probability({"Y": 1}, interventions={"T": 1})
            p0 = self.scm.probability({"Y": 1}, interventions={"T": 0})
            ate = p1 - p0
            frame = CognitiveFrame(
                "causal_effect",
                subject="treatment",
                answer="helps" if ate >= 0 else "hurts",
                confidence=min(1.0, abs(ate) * 4 + 0.5),
                evidence={"p_do_positive": p1, "p_do_negative": p0, "ate": ate},
            )
        else:
            frame = CognitiveFrame("unknown", answer="unknown", confidence=0.0, evidence={"route": "none"})
        return self.workspace.publish(frame)

    def speak(self, frame: CognitiveFrame) -> str:
        return generate_from_plan(self.host, self.tokenizer, frame.speech_plan())

    def answer(self, utterance: str) -> tuple[CognitiveFrame, str]:
        frame = self.comprehend(utterance)
        return frame, self.speak(frame)


def build_training_frames() -> list[CognitiveFrame]:
    frames: list[CognitiveFrame] = [CognitiveFrame("memory_location", subject=name, answer=city, confidence=1.0) for name, city in BROCA_FACTS]
    frames.extend(
        [
            CognitiveFrame("active_action", answer="listen", confidence=0.72, evidence={"policy_posterior": {"listen": 0.72, "open_left": 0.14, "open_right": 0.14}}),
            CognitiveFrame("active_action", answer="open_left", confidence=0.92, evidence={"policy_posterior": {"listen": 0.04, "open_left": 0.92, "open_right": 0.04}}),
            CognitiveFrame("active_action", answer="open_right", confidence=0.91, evidence={"policy_posterior": {"listen": 0.05, "open_left": 0.04, "open_right": 0.91}}),
            CognitiveFrame("causal_effect", subject="treatment", answer="helps", confidence=0.9, evidence={"p_do_positive": 0.55, "p_do_negative": 0.45, "ate": 0.10}),
            CognitiveFrame("causal_effect", subject="treatment", answer="hurts", confidence=0.9, evidence={"p_do_positive": 0.35, "p_do_negative": 0.55, "ate": -0.20}),
            CognitiveFrame("unknown", answer="unknown", confidence=0.0),
        ]
    )
    return frames


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
    prefix = tokenizer.encode("speak :")
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
    backend: str = "tiny",
    llama_model_id: str = "meta-llama/Llama-3.2-1B-Instruct",
    device: torch.device | str | None = None,
    hf_token: str | bool | None = None,
) -> dict:
    seed_everything(seed)
    resolved_device = device if isinstance(device, torch.device) else pick_torch_device(device)
    if backend == "llama":
        from .llama_broca_host import load_llama_broca_host

        model, tokenizer = load_llama_broca_host(llama_model_id, device=resolved_device, token=hf_token)
    elif backend == "tiny":
        tokenizer = build_broca_tokenizer()
        model = make_broca_host(tokenizer, seed=seed)
        model.to(resolved_device)
    else:
        raise ValueError(f"unknown backend {backend!r}; expected tiny or llama")
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
    backend: str = "tiny",
    llama_model_id: str = "meta-llama/Llama-3.2-1B-Instruct",
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
        backend=backend,
        llama_model_id=llama_model_id,
        device=device,
        hf_token=hf_token,
    )
    total, trainable = count_parameters(mind.host)
    language_only = generate_without_broca(mind.host, mind.tokenizer, prefix="speak :", max_new_tokens=5)
    queries = ["where is ada ?", "what action should i take ?", "does treatment help ?"]
    rows: list[dict] = []
    for q in queries:
        frame, utterance = mind.answer(q)
        rows.append({"query": q, "intent": frame.intent, "latent_answer": frame.answer, "speech": utterance, "evidence": frame.evidence})

    # Persistence: a fresh substrate process, same DB, same namespace, no reseeding needed.
    restarted = BrocaMind(
        seed=seed,
        db_path=path,
        namespace=f"broca_{seed}",
        backend=backend,
        llama_model_id=llama_model_id,
        device=device,
        hf_token=hf_token,
    )
    frame2, speech2 = restarted.answer("where is hopper ?")

    # Broca lesion: the substrate still computes the frame, but the host no longer verbalizes it.
    lesion_frame = mind.comprehend("where is ada ?")
    with mind.host.grafts_enabled(False):
        lesioned_speech = generate_from_plan(mind.host, mind.tokenizer, lesion_frame.speech_plan())

    train_result: dict = {}
    if train_bridge:
        train_result = train_broca_bridge(
            seed=seed,
            steps=train_bridge_steps,
            backend=backend,
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
        "restart": {"latent_answer": frame2.answer, "speech": speech2, "records": restarted.memory.count()},
        "broca_lesion": {"latent_answer": lesion_frame.answer, "speech_without_graft": lesioned_speech},
        "trainable_broca": trainable_meta,
        "backend": backend,
    }

    if verbose:
        print("\n=== 6) LLM-as-Broca architecture ===")
        print("The language host is treated as a speech/planning interface. Memory, action selection, and causality live outside it and publish latent frames.")
        print(f"frozen Broca host params={total:,}; trainable host params before trainable bridge={trainable:,}")
        print(f"persistent semantic memory records={mind.memory.count()}; db={path}")
        print("\nLanguage host with no Broca/faculty state:")
        print(f"  speak : -> {language_only}")
        print("\nCognitive substrate -> Broca verbalization:")
        print(f"{'query':<32} {'intent':<17} {'latent':<10} speech")
        print(f"{'-'*32} {'-'*17} {'-'*10} {'-'*34}")
        for r in rows:
            print(f"{r['query']:<32} {r['intent']:<17} {r['latent_answer']:<10} {r['speech']}")
        print("\nPersistence check after fresh substrate reload:")
        print(f"  query='where is hopper ?' latent={frame2.answer} speech='{speech2}' records={restarted.memory.count()}")
        print("\nBroca lesion check:")
        print(f"  substrate still has latent answer={lesion_frame.answer}; speech without graft='{lesioned_speech}'")
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


