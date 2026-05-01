from __future__ import annotations

import argparse
import numbers
from pathlib import Path

from .benchmarks.architecture_eval import run_broca_architecture_eval
from .experiments import (
    run_active_inference_experiment,
    run_all as run_faculty_stack,
    run_causal_experiment,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mosaic: make the language model a Broca-style interface over real faculties.")
    parser.add_argument("--mode", choices=["all", "broca", "friston", "pearl"], default="all")
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
        help="HF model id (default meta-llama/Llama-3.2-1B-Instruct or MODEL_ID). Requires HF access for gated checkpoints.",
    )
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def run_broca_benchmark() -> None:
        result = run_broca_architecture_eval(
            seed=args.seed,
            db_path=out / "broca_architecture_eval.sqlite",
            llama_model_id=args.broca_model_id,
            device=args.broca_device,
            output_path=out / "broca_architecture_eval.json",
        )
        metrics = result.get("metrics")
        if not isinstance(metrics, dict):
            raise KeyError("run_broca_architecture_eval result missing 'metrics' mapping")
        enhanced = metrics.get("enhanced_broca_architecture")
        if not isinstance(enhanced, dict):
            raise KeyError("metrics missing 'enhanced_broca_architecture'")
        print("\n=== Broca architecture benchmark ===")
        print(f"results={out / 'broca_architecture_eval.json'}")
        for key in ("speech_exact_accuracy", "answer_present_accuracy"):
            val = enhanced.get(key)
            if val is None:
                raise KeyError(
                    f"enhanced_broca_architecture missing required metric {key!r}; keys={sorted(enhanced.keys())}"
                )
            if not isinstance(val, numbers.Real):
                raise TypeError(
                    f"enhanced_broca_architecture[{key!r}] must be numeric, got {type(val).__name__}: {val!r}"
                )
        print(f"enhanced speech_exact_accuracy={float(enhanced.get('speech_exact_accuracy')):.3f}")
        print(f"enhanced answer_present_accuracy={float(enhanced.get('answer_present_accuracy')):.3f}")

    if args.mode == "all":
        run_faculty_stack(seed=args.seed, out_dir=out, verbose=True)
        run_broca_benchmark()
    elif args.mode == "broca":
        run_broca_benchmark()
    elif args.mode == "friston":
        run_active_inference_experiment(seed=args.seed, verbose=True)
    elif args.mode == "pearl":
        run_causal_experiment(verbose=True)


if __name__ == "__main__":
    main()
