"""CLI: ``python -m core.paper`` — refresh ``paper/include/experiment/``."""

from __future__ import annotations

import argparse
import json
import sys

from core.infra.logging_setup import configure_lab_logging

from .harness import refresh_paper_experiments


def main(argv: list[str] | None = None) -> None:
    configure_lab_logging()
    p = argparse.ArgumentParser(description="Regenerate paper experiment TeX from benchmark harness.")
    p.add_argument(
        "--json-out",
        type=str,
        default="",
        help="If set, write a short JSON status report to this path.",
    )
    args = p.parse_args(argv)

    report = refresh_paper_experiments()
    if args.json_out:
        path = args.json_out.strip()
        if path:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            print(path, file=sys.stderr)


if __name__ == "__main__":
    main()
