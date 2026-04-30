from __future__ import annotations

import json
from pathlib import Path

from .active_inference import ActiveInferenceAgent, TigerDoorEnv, build_tiger_pomdp, random_episode, run_episode
from .causal import build_frontdoor_scm, build_simpson_scm


def run_active_inference_experiment(seed: int = 0, episodes: int = 80, verbose: bool = True) -> dict:
    pomdp = build_tiger_pomdp()
    agent = ActiveInferenceAgent(pomdp, horizon=1, learn=True)
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


def run_all(seed: int = 0, out_dir: str | Path = "runs", verbose: bool = True) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "friston": run_active_inference_experiment(seed=seed, episodes=80, verbose=verbose),
        "pearl": run_causal_experiment(verbose=verbose),
    }
    path = out_dir / f"results_seed{seed}.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if verbose:
        print(f"\nSaved run summary: {path}")
    return result
