
from __future__ import annotations

import argparse
from pathlib import Path

from .broca import run_broca_experiment
from .experiments import (
    run_active_inference_experiment,
    run_all as run_faculty_stack,
    run_causal_experiment,
    run_memory_experiment,
    run_trainable_bridge_experiment,
    run_unified_stack_experiment,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="ASI Broca Lab: make the language model a Broca-style interface over real faculties.")
    parser.add_argument("--mode", choices=["all", "broca", "memory", "friston", "pearl", "unified", "bridge"], default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument(
        "--broca-device",
        default=None,
        help="Torch device override (cpu, mps, cuda:0). Default auto-prefers Apple MPS, then CUDA, else CPU.",
    )
    parser.add_argument(
        "--broca-model-id",
        default=None,
        help="HF model id (default meta-llama/Llama-3.2-1B-Instruct or ASI_BROCA_MODEL_ID). Requires HF access for gated checkpoints.",
    )
    parser.add_argument(
        "--no-train-bridge",
        action="store_true",
        help="Skip the trainable residual bridge demo (loads faster for llama)",
    )
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.mode == "all":
        run_faculty_stack(seed=args.seed, out_dir=out, verbose=True)
        run_broca_experiment(
            seed=args.seed,
            db_path=out / "broca_semantic_memory.sqlite",
            verbose=True,
            llama_model_id=args.broca_model_id,
            device=args.broca_device,
            train_bridge=not args.no_train_bridge,
        )
    elif args.mode == "broca":
        run_broca_experiment(
            seed=args.seed,
            db_path=out / "broca_semantic_memory.sqlite",
            verbose=True,
            llama_model_id=args.broca_model_id,
            device=args.broca_device,
            train_bridge=not args.no_train_bridge,
        )
    elif args.mode == "memory":
        run_memory_experiment(seed=args.seed, db_path=out / "faculty_memory.sqlite", verbose=True)
    elif args.mode == "friston":
        run_active_inference_experiment(seed=args.seed, verbose=True)
    elif args.mode == "pearl":
        run_causal_experiment(verbose=True)
    elif args.mode == "unified":
        run_unified_stack_experiment(seed=args.seed, db_path=out / "faculty_stack.sqlite", verbose=True)
    elif args.mode == "bridge":
        run_trainable_bridge_experiment(seed=args.seed, verbose=True)


if __name__ == "__main__":
    main()


