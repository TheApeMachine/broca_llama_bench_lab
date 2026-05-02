"""Faculty demo CLI: ``mosaic demo`` / ``python -m core.main demo``.

Runs the architecture benchmark for ``--mode broca`` (default).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(prog="mosaic demo")
    parser.add_argument("--mode", default="broca", help="Only 'broca' is supported today.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.mode != "broca":
        print(f"Unsupported --mode {args.mode!r}; use broca.", file=sys.stderr)
        raise SystemExit(2)

    from core.benchmarks.architecture_eval import run_broca_architecture_eval
    from core.host.llama_broca_host import resolve_hf_hub_token
    from core.system.device import pick_torch_device
    from core.substrate.runtime import default_model_id, default_substrate_sqlite_path, ensure_parent_dir

    out = Path("runs") / "broca_architecture_eval_demo.json"
    ensure_parent_dir(out)
    db = default_substrate_sqlite_path()
    ensure_parent_dir(db)
    run_broca_architecture_eval(
        seed=args.seed,
        db_path=db,
        llama_model_id=default_model_id(),
        device=str(pick_torch_device(None)),
        hf_token=resolve_hf_hub_token(),
        output_path=out,
    )
    print(f"Wrote {out}", flush=True)
