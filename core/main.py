"""Single top-level CLI: ``python -m core``, ``python -m core.main``, or ``mosaic``.

Every subcommand uses the same documented runtime (environment variables,
substrate DB path, model id resolution) via :mod:`core.cli` where a
:class:`core.substrate.controller.SubstrateController` is involved.

``demo`` and ``paper`` forward trailing arguments unchanged (so e.g.
``python -m core.main demo --mode broca`` works). Nested subparsers do not;
see Python :mod:`argparse` + ``REMAINDER`` behavior.
"""

from __future__ import annotations

import argparse
import sys
from typing import Callable


Handler = Callable[[list[str]], None]


def _strip_optional_ddash(args: list[str]) -> list[str]:
    if args and args[0] == "--":
        return args[1:]

    return args


def _cmd_chat(argv: list[str]) -> None:
    from .chat.repl import run_chat_repl

    run_chat_repl(argv)


def _cmd_chat_tui(argv: list[str]) -> None:
    from .tui.chat import run_chat_tui

    run_chat_tui(argv)


def _research_lab_or_exit(name: str):
    try:
        import research_lab  # noqa: F401
    except ImportError as exc:
        print(
            f"`mosaic {name}` requires the research_lab package. "
            f"Install benchmark extras: pip install -e \".[benchmark]\".",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc


def _cmd_bench(argv: list[str]) -> None:
    _research_lab_or_exit("bench")
    from research_lab.benchmarks.__main__ import main as bench_main

    bench_main(argv)


def _cmd_bench_tui(argv: list[str]) -> None:
    _research_lab_or_exit("bench-tui")
    from research_lab.tui.bench import run_bench_tui

    run_bench_tui(argv)


def _cmd_demo(argv: list[str]) -> None:
    _research_lab_or_exit("demo")
    from research_lab.demo import main as demo_main

    demo_main(_strip_optional_ddash(argv))


def _cmd_paper(argv: list[str]) -> None:
    _research_lab_or_exit("paper")
    from research_lab.paper.__main__ import main as paper_main

    paper_main(_strip_optional_ddash(argv))


_COMMANDS: dict[str, tuple[str, Handler]] = {
    "chat": ("Streaming terminal chat (full stack; same substrate as chat-tui).", _cmd_chat),
    "chat-tui": ("Textual chat dashboard.", _cmd_chat_tui),
    "tui": ("Alias for chat-tui.", _cmd_chat_tui),
    "bench": ("Unified benchmark harness (fixed configuration).", _cmd_bench),
    "bench-tui": ("Textual benchmark dashboard (wraps research_lab.benchmarks).", _cmd_bench_tui),
    "demo": ("Faculty experiments and Broca architecture benchmark.", _cmd_demo),
    "paper": ("Regenerate paper experiment TeX from benchmark harness.", _cmd_paper),
}


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    choices = sorted(_COMMANDS)
    parser = argparse.ArgumentParser(
        prog="mosaic",
        description=(
            "Mosaic lab CLI: chat, benchmarks, demo, and paper hooks share the same "
            "substrate runtime (see core.cli)."
        ),
        epilog="When installed as a package you can run the same commands via the `mosaic` console script.",
    )

    parser.add_argument(
        "command",
        choices=choices,
        help="Subcommand (see list in -h).",
    )

    parser.add_argument(
        "remainder",
        nargs=argparse.REMAINDER,
        help="Extra arguments passed to that command (e.g. demo --mode broca --seed 0).",
    )

    args = parser.parse_args(argv)

    _handler = _COMMANDS[args.command][1]
    _handler(args.remainder)


if __name__ == "__main__":
    main()
