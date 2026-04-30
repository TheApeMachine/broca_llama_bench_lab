from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F

from .active_inference import ActiveInferenceAgent, TigerDoorEnv, build_tiger_pomdp, random_episode, run_episode
from .causal import build_frontdoor_scm, build_simpson_scm
from .grafts import (
    ActiveInferenceTokenGraft,
    CausalEffectTokenGraft,
    FeatureVectorGraft,
    KVMemoryGraft,
    memorize_persistent_next_token,
)
from .host import TinyCausalTransformer, TinyConfig, count_parameters, freeze_module
from .memory import SQLiteActivationMemory
from .tokenizer import RegexTokenizer


FACTS: list[tuple[str, str]] = [
    ("where is ada ? answer :", "rome"),
    ("where is byron ? answer :", "paris"),
    ("where is curie ? answer :", "tokyo"),
    ("where is darwin ? answer :", "lima"),
    ("where is euclid ? answer :", "oslo"),
    ("where is faraday ? answer :", "cairo"),
    ("where is gauss ? answer :", "vienna"),
    ("where is hopper ? answer :", "lisbon"),
]

UNIFIED_TASKS: list[tuple[str, str, str]] = [
    ("memory", "where is ada ? answer :", "rome"),
    ("active_inference", "what action should i take ? answer :", "listen"),
    ("causal", "does treatment help ? answer :", "helps"),
]

EXTRA_TOKENS = [
    "berlin", "madrid", "lisbon", "vienna", "mars", "venus", "earth", "jupiter",
    "true", "false", "safe", "unsafe", "unknown",
    "listen", "open_left", "open_right", "action", "should", "take",
    "treatment", "help", "helps", "hurts", "causal", "memory", "intervene", "faculty",
]


@dataclass
class EvalRow:
    prompt: str
    target: str
    prediction: str
    confidence: float
    correct: bool
    kind: str = ""


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def build_tokenizer() -> RegexTokenizer:
    texts: list[str] = []
    for p, t in FACTS:
        texts.extend([p, t])
    for _, p, t in UNIFIED_TASKS:
        texts.extend([p, t])
    return RegexTokenizer.fit(texts, extra_tokens=EXTRA_TOKENS)


def make_model(tokenizer: RegexTokenizer, seed: int = 0, *, d_model: int = 96) -> TinyCausalTransformer:
    seed_everything(seed)
    cfg = TinyConfig(
        vocab_size=len(tokenizer),
        d_model=d_model,
        n_layers=2,
        n_heads=4,
        d_ff=2 * d_model,
        max_seq_len=48,
        dropout=0.0,
    )
    model = TinyCausalTransformer(cfg)
    model.eval()
    return model


@torch.no_grad()
def predict_next(model: TinyCausalTransformer, tokenizer: RegexTokenizer, prompt: str, *, extra_state: dict | None = None) -> tuple[str, float]:
    batch = tokenizer.batch_encode([prompt], device=next(model.parameters()).device)
    logits = model(batch.ids, batch.attention_mask, extra_state=extra_state)
    last = batch.lengths[0].item() - 1
    probs = F.softmax(logits[0, last], dim=-1)
    pred_id = int(probs.argmax().item())
    return tokenizer.decode_id(pred_id), float(probs[pred_id].item())


@torch.no_grad()
def evaluate_next_token(model: TinyCausalTransformer, tokenizer: RegexTokenizer, items: Sequence[tuple[str, str] | tuple[str, str, str]]) -> tuple[float, list[EvalRow]]:
    rows: list[EvalRow] = []
    for item in items:
        if len(item) == 2:
            kind = ""
            prompt, target = item  # type: ignore[misc]
        else:
            kind, prompt, target = item  # type: ignore[misc]
        pred, conf = predict_next(model, tokenizer, prompt)
        rows.append(EvalRow(prompt=prompt, target=target, prediction=pred, confidence=conf, correct=(pred == target), kind=kind))
    acc = sum(r.correct for r in rows) / max(1, len(rows))
    return acc, rows


def format_rows(rows: Sequence[EvalRow]) -> str:
    widths = {
        "kind": max([4] + [len(r.kind) for r in rows]),
        "prompt": max([6] + [len(r.prompt) for r in rows]),
        "target": max([6] + [len(r.target) for r in rows]),
        "pred": max([4] + [len(r.prediction) for r in rows]),
    }
    show_kind = any(r.kind for r in rows)
    if show_kind:
        head = f"{'kind':<{widths['kind']}}  {'prompt':<{widths['prompt']}}  {'target':<{widths['target']}}  {'pred':<{widths['pred']}}  conf    ok"
        rule = f"{'-' * widths['kind']}  {'-' * widths['prompt']}  {'-' * widths['target']}  {'-' * widths['pred']}  ------  --"
    else:
        head = f"{'prompt':<{widths['prompt']}}  {'target':<{widths['target']}}  {'pred':<{widths['pred']}}  conf    ok"
        rule = f"{'-' * widths['prompt']}  {'-' * widths['target']}  {'-' * widths['pred']}  ------  --"
    lines = [head, rule]
    for r in rows:
        ok = "✓" if r.correct else "·"
        if show_kind:
            lines.append(f"{r.kind:<{widths['kind']}}  {r.prompt:<{widths['prompt']}}  {r.target:<{widths['target']}}  {r.prediction:<{widths['pred']}}  {r.confidence:>6.3f}  {ok}")
        else:
            lines.append(f"{r.prompt:<{widths['prompt']}}  {r.target:<{widths['target']}}  {r.prediction:<{widths['pred']}}  {r.confidence:>6.3f}  {ok}")
    return "\n".join(lines)


def install_persistent_memory(
    model: TinyCausalTransformer,
    tokenizer: RegexTokenizer,
    store: SQLiteActivationMemory,
    *,
    namespace: str,
    facts: Sequence[tuple[str, str]] = FACTS,
    write: bool = True,
) -> KVMemoryGraft:
    graft = KVMemoryGraft(model.cfg.d_model, strength=18.0, threshold=0.86, temperature=0.025, query_mode="sequence_mean")
    model.add_graft("final_hidden", graft)
    if write:
        for prompt, target in facts:
            memorize_persistent_next_token(
                store,
                model,
                tokenizer,
                prompt,
                target,
                namespace=namespace,
                kind="fact",
                query_mode=graft.query_mode,
                value_scale=1.0,
            )
    store.load_into_graft(graft, namespace=namespace, kind="fact")
    return graft


def run_memory_experiment(seed: int = 0, db_path: str | Path = "runs/faculty_memory.sqlite", verbose: bool = True) -> dict:
    tokenizer = build_tokenizer()
    path = Path(db_path)
    if path.exists():
        path.unlink()
    for suffix in ("-wal", "-shm"):
        sp = Path(str(path) + suffix)
        if sp.exists():
            sp.unlink()
    namespace = f"seed_{seed}"
    store = SQLiteActivationMemory(path, default_namespace=namespace)
    model = make_model(tokenizer, seed)
    freeze_module(model)

    before, before_rows = evaluate_next_token(model, tokenizer, FACTS)
    graft = install_persistent_memory(model, tokenizer, store, namespace=namespace, write=True)
    after_write, after_rows = evaluate_next_token(model, tokenizer, FACTS)

    restarted = make_model(tokenizer, seed)
    freeze_module(restarted)
    fresh_graft = install_persistent_memory(restarted, tokenizer, store, namespace=namespace, write=False)
    loaded = len(fresh_graft.metadata)
    after_restart, restart_rows = evaluate_next_token(restarted, tokenizer, FACTS)

    if verbose:
        print("\n=== 1) Persistent activation-memory graft ===")
        total, trainable = count_parameters(restarted)
        print(f"frozen host params={total:,}; trainable after memory load={trainable:,}")
        print(f"Frozen host with no graft: accuracy={before:.3f}")
        print(format_rows(before_rows))
        print("\nAfter writing facts into SQLite-backed activation memory:")
        print(f"accuracy={after_write:.3f}; persisted_records={store.count(namespace=namespace)}")
        print(format_rows(after_rows))
        print("\nAfter fresh model process reloads the persistent memory DB:")
        print(f"accuracy={after_restart:.3f}; loaded_records={loaded}; db={path}")
        print(format_rows(restart_rows))
        print("\nGraft report:")
        print(restarted.graft_report())
        weights = fresh_graft.last_debug.get("weights")
        if weights is not None and fresh_graft.metadata:
            top = int(weights[0].argmax().item())
            print(f"top memory for last evaluated prompt: {fresh_graft.metadata[top]}")

    return {
        "before": before,
        "after_write": after_write,
        "after_restart": after_restart,
        "persisted_records": store.count(namespace=namespace),
        "db_path": str(path),
    }


def run_active_inference_experiment(seed: int = 0, episodes: int = 80, verbose: bool = True) -> dict:
    pomdp = build_tiger_pomdp(reliability=0.85)
    agent = ActiveInferenceAgent(pomdp, horizon=1, precision=8.0, learn=True)
    d0 = agent.decide()

    policy_rows = []
    for ev, prob in zip(d0.policies, d0.posterior_over_policies):
        if len(ev.policy) == 1:
            policy_rows.append(
                {
                    "policy": pomdp.action_names[ev.policy[0]],
                    "G": ev.expected_free_energy,
                    "risk": ev.risk,
                    "ambiguity": ev.ambiguity,
                    "epistemic": ev.epistemic_value,
                    "posterior": prob,
                }
            )

    inspect_env = TigerDoorEnv(seed=seed + 11)
    success, reward, trace = run_episode(agent, inspect_env, max_steps=3)

    active_success = 0
    active_reward = 0.0
    random_success = 0
    random_reward = 0.0
    active_env = TigerDoorEnv(seed=seed + 123)
    random_env = TigerDoorEnv(seed=seed + 123)
    for _ in range(episodes):
        ok, rew, _ = run_episode(agent, active_env, max_steps=3)
        active_success += int(ok)
        active_reward += rew
        rok, rrew = random_episode(random_env, max_steps=3)
        random_success += int(rok)
        random_reward += rrew

    result = {
        "first_action": d0.action_name,
        "policy_rows": policy_rows,
        "inspect_success": success,
        "inspect_reward": reward,
        "trace": trace,
        "active_success": active_success / episodes,
        "active_avg_reward": active_reward / episodes,
        "random_success": random_success / episodes,
        "random_avg_reward": random_reward / episodes,
    }

    if verbose:
        print("\n=== 2) Friston-style active inference faculty ===")
        print("Belief state:", dict(zip(pomdp.state_names, [round(float(x), 3) for x in d0.qs])))
        print("First action selected by minimizing expected free energy:", d0.action_name)
        print("policy        G       risk    ambiguity  epistemic  posterior")
        for row in policy_rows:
            print(f"{row['policy']:<10} {row['G']:>7.3f} {row['risk']:>7.3f} {row['ambiguity']:>9.3f} {row['epistemic']:>9.3f} {row['posterior']:>9.3f}")
        print("\nInspected episode:")
        for i, step in enumerate(trace, 1):
            print(f"{i}. action={step['action']:<10} observation={step['observation']:<10} reward={step['reward']:+.2f}")
            print(f"   posterior_state={step['posterior']}")
        print(f"\nMonte Carlo over {episodes} episodes:")
        print(f"active inference success={result['active_success']:.3f}, avg_reward={result['active_avg_reward']:.3f}")
        print(f"random baseline   success={result['random_success']:.3f}, avg_reward={result['random_avg_reward']:.3f}")
        # Show that the observation model is not static decoration.
        listen = pomdp.action_names.index("listen")
        print("learned listen likelihood columns after episodes:")
        for s, sname in enumerate(pomdp.state_names):
            col = {pomdp.observation_names[o]: round(pomdp.A[listen][o][s], 3) for o in range(pomdp.n_observations)}
            print(f"  state={sname}: {col}")

    return result


def run_causal_experiment(verbose: bool = True) -> dict:
    simpson = build_simpson_scm()
    naive_t1 = simpson.probability({"Y": 1}, given={"T": 1})
    naive_t0 = simpson.probability({"Y": 1}, given={"T": 0})
    do_t1 = simpson.probability({"Y": 1}, interventions={"T": 1})
    do_t0 = simpson.probability({"Y": 1}, interventions={"T": 0})
    backdoor = simpson.backdoor_sets("T", "Y", max_size=2)
    bd = backdoor[0]
    adj_t1 = simpson.backdoor_adjustment(treatment="T", treatment_value=1, outcome="Y", outcome_value=1, adjustment_set=bd)
    adj_t0 = simpson.backdoor_adjustment(treatment="T", treatment_value=0, outcome="Y", outcome_value=1, adjustment_set=bd)
    cf = simpson.counterfactual_probability({"Y": 1}, evidence={"S": 1, "T": 1, "Y": 1}, interventions={"T": 0})

    front = build_frontdoor_scm()
    fd_sets = front.frontdoor_sets("X", "Y", max_size=1)
    fd = fd_sets[0]
    fd_formula = front.frontdoor_adjustment(treatment="X", treatment_value=1, outcome="Y", outcome_value=1, mediator_set=fd)
    fd_do = front.probability({"Y": 1}, interventions={"X": 1})
    naive_x1 = front.probability({"Y": 1}, given={"X": 1})

    result = {
        "graph_parents": simpson.graph_parents(include_exogenous=False),
        "observational_t1": naive_t1,
        "observational_t0": naive_t0,
        "do_t1": do_t1,
        "do_t0": do_t0,
        "ate": do_t1 - do_t0,
        "backdoor_sets": [list(x) for x in backdoor],
        "adjusted_t1": adj_t1,
        "adjusted_t0": adj_t0,
        "counterfactual_success_if_untreated": cf,
        "frontdoor_sets": [list(x) for x in fd_sets],
        "frontdoor_formula_x1": fd_formula,
        "frontdoor_do_x1": fd_do,
        "frontdoor_naive_x1": naive_x1,
    }

    if verbose:
        print("\n=== 3) Pearl-style structural causal faculty ===")
        print("Graph parents:", result["graph_parents"])
        print(f"Naive observation: P(Y=1 | T=1)={naive_t1:.3f}; P(Y=1 | T=0)={naive_t0:.3f}")
        print(f"Intervention:      P(Y=1 | do(T=1))={do_t1:.3f}; P(Y=1 | do(T=0))={do_t0:.3f}; ATE={do_t1 - do_t0:+.3f}")
        print("Backdoor sets found by graph search:", backdoor)
        print(f"Backdoor-adjusted: P(Y=1 | do(T=1))={adj_t1:.3f}; P(Y=1 | do(T=0))={adj_t0:.3f}")
        print(f"Counterfactual:    P(Y_do(T=0)=1 | S=1,T=1,Y=1)={cf:.3f}")
        print("\nFront-door model with hidden confounder U between X and Y:")
        print("Frontdoor sets found by graph search:", fd_sets)
        print(f"Naive P(Y=1 | X=1)={naive_x1:.3f}; exact P(Y=1 | do(X=1))={fd_do:.3f}; frontdoor formula={fd_formula:.3f}")

    return result


def build_unified_stack(seed: int = 0, db_path: str | Path = "runs/faculty_stack.sqlite"):
    tokenizer = build_tokenizer()
    namespace = f"unified_{seed}"
    path = Path(db_path)
    if path.exists():
        path.unlink()
    for suffix in ("-wal", "-shm"):
        sp = Path(str(path) + suffix)
        if sp.exists():
            sp.unlink()
    store = SQLiteActivationMemory(path, default_namespace=namespace)
    model = make_model(tokenizer, seed)
    freeze_module(model)

    memory_graft = install_persistent_memory(model, tokenizer, store, namespace=namespace, facts=FACTS, write=True)

    pomdp = build_tiger_pomdp(reliability=0.85)
    active_agent = ActiveInferenceAgent(pomdp, horizon=1, precision=8.0, learn=False)
    active_graft = ActiveInferenceTokenGraft(
        active_agent,
        token_by_action={name: tokenizer.token_to_id[name] for name in pomdp.action_names if name in tokenizer.token_to_id},
        trigger_ids=[tokenizer.token_to_id["action"]],
        strength=20.0,
    )
    model.add_graft("final_hidden", active_graft)

    scm = build_simpson_scm()
    causal_graft = CausalEffectTokenGraft(
        scm,
        treatment="T",
        outcome="Y",
        outcome_value=1,
        positive_token=tokenizer.token_to_id["helps"],
        negative_token=tokenizer.token_to_id["hurts"],
        treatment_values=(1, 0),
        trigger_ids=[tokenizer.token_to_id["treatment"]],
        strength=20.0,
    )
    model.add_graft("final_hidden", causal_graft)
    return model, tokenizer, store, memory_graft, active_graft, causal_graft


def run_unified_stack_experiment(seed: int = 0, db_path: str | Path = "runs/faculty_stack.sqlite", verbose: bool = True) -> dict:
    tokenizer = build_tokenizer()
    plain = make_model(tokenizer, seed)
    freeze_module(plain)
    before, before_rows = evaluate_next_token(plain, tokenizer, UNIFIED_TASKS)

    model, tokenizer, store, memory_graft, active_graft, causal_graft = build_unified_stack(seed=seed, db_path=db_path)
    after, after_rows = evaluate_next_token(model, tokenizer, UNIFIED_TASKS)

    result = {
        "before": before,
        "after": after,
        "records": store.count(namespace=f"unified_{seed}"),
        "active_choice": active_graft.last_name,
        "causal_effects": causal_graft.last_effects,
    }
    if verbose:
        print("\n=== 4) Unified neural faculty stack ===")
        print("Same frozen host; three different faculties write into the residual stream.")
        print(f"Before graft stack: accuracy={before:.3f}")
        print(format_rows(before_rows))
        print(f"\nAfter graft stack:  accuracy={after:.3f}; persistent_records={result['records']}")
        print(format_rows(after_rows))
        print("\nGraft report:")
        print(model.graft_report())
        print("active inference token selected:", result["active_choice"])
        print("causal intervention estimates:", {k: round(v, 3) for k, v in result["causal_effects"].items()})
    return result



def _bridge_dataset(tokenizer: RegexTokenizer):
    """Feature vectors from causal estimates and policy posteriors.

    Layout:
      [is_causal, is_active, p_do_pos_or_listen, p_do_neg_or_open_left, ate_or_open_right, bias]
    """
    examples = [
        ([1.0, 0.0, 0.55, 0.45, +0.10, 1.0], "helps"),
        ([1.0, 0.0, 0.30, 0.60, -0.30, 1.0], "hurts"),
        ([1.0, 0.0, 0.72, 0.50, +0.22, 1.0], "helps"),
        ([1.0, 0.0, 0.42, 0.58, -0.16, 1.0], "hurts"),
        ([0.0, 1.0, 0.53, 0.24, 0.23, 1.0], "listen"),
        ([0.0, 1.0, 0.05, 0.90, 0.05, 1.0], "open_left"),
        ([0.0, 1.0, 0.05, 0.05, 0.90, 1.0], "open_right"),
        ([0.0, 1.0, 0.70, 0.15, 0.15, 1.0], "listen"),
    ]
    features = torch.tensor([x for x, _ in examples], dtype=torch.float32)
    targets = torch.tensor([tokenizer.token_to_id[y] for _, y in examples], dtype=torch.long)
    prompts = ["faculty answer :" for _ in examples]
    return prompts, features, targets, [y for _, y in examples]


def run_trainable_bridge_experiment(seed: int = 0, steps: int = 220, verbose: bool = True) -> dict:
    """Train only a graft that maps faculty-state vectors into hidden space."""

    tokenizer = build_tokenizer()
    model = make_model(tokenizer, seed)
    freeze_module(model)
    trigger_id = tokenizer.token_to_id["faculty"]
    bridge = FeatureVectorGraft(d_features=6, d_model=model.cfg.d_model, trigger_ids=[trigger_id], strength=1.0)
    model.add_graft("final_hidden", bridge)
    for p in bridge.parameters():
        p.requires_grad = True

    prompts, features, targets, target_names = _bridge_dataset(tokenizer)
    batch = tokenizer.batch_encode(prompts)
    last = batch.lengths - 1

    def eval_bridge():
        with torch.no_grad():
            logits = model(batch.ids, batch.attention_mask, extra_state={"faculty_features": features})
            pred_ids = logits[torch.arange(len(prompts)), last].argmax(dim=-1)
            preds = [tokenizer.decode_id(int(i)) for i in pred_ids]
            acc = sum(p == t for p, t in zip(preds, target_names)) / len(target_names)
            return acc, preds

    before, before_preds = eval_bridge()
    opt = torch.optim.AdamW(bridge.parameters(), lr=0.08, weight_decay=0.0)
    losses: list[float] = []
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        logits = model(batch.ids, batch.attention_mask, extra_state={"faculty_features": features})
        loss = torch.nn.functional.cross_entropy(logits[torch.arange(len(prompts)), last], targets)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))

    after, after_preds = eval_bridge()
    total, trainable = count_parameters(model)
    result = {
        "before": before,
        "after": after,
        "initial_predictions": before_preds,
        "final_predictions": after_preds,
        "targets": target_names,
        "bridge_params": sum(p.numel() for p in bridge.parameters()),
        "host_total_params": total,
        "trainable_params": trainable,
        "final_loss": losses[-1],
    }

    if verbose:
        print("\n=== 5) Trainable faculty-to-layer bridge ===")
        print("The host is frozen. Only a small feature-vector graft learns how to write faculty states into hidden activations.")
        print(f"host params={total:,}; trainable bridge params={result['bridge_params']:,}; trainable total={trainable:,}")
        print(f"before training bridge accuracy={before:.3f}; after={after:.3f}; final_loss={result['final_loss']:.4f}")
        print("targets:          ", target_names)
        print("before predictions:", before_preds)
        print("after predictions: ", after_preds)

    return result


def run_all(seed: int = 0, out_dir: str | Path = "runs", verbose: bool = True) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "memory": run_memory_experiment(seed=seed, db_path=out_dir / "faculty_memory.sqlite", verbose=verbose),
        "friston": run_active_inference_experiment(seed=seed, episodes=80, verbose=verbose),
        "pearl": run_causal_experiment(verbose=verbose),
        "unified": run_unified_stack_experiment(seed=seed, db_path=out_dir / "faculty_stack.sqlite", verbose=verbose),
        "trainable_bridge": run_trainable_bridge_experiment(seed=seed, verbose=verbose),
    }
    path = out_dir / f"results_seed{seed}.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if verbose:
        print(f"\nSaved run summary: {path}")
    return result


