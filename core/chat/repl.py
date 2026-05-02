"""Streaming terminal chat (full substrate stack)."""

from __future__ import annotations

import argparse
import sys

from core.cli import (
    build_substrate_controller,
    configure_lab_session,
    self_improve_enabled_from_env,
    start_background_stack,
    stop_background_stack,
)
from core.substrate.runtime import (
    BROCA_BACKGROUND_INTERVAL_S,
    CHAT_DO_SAMPLE,
    CHAT_MAX_NEW_TOKENS,
    CHAT_NAMESPACE,
    CHAT_TEMPERATURE,
    CHAT_TOP_P,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Mosaic chat (full substrate; no tuning flags).")
    p.add_argument("-h", "--help", action="help", help="Show this message and exit.")

    return p


def run_chat_repl(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = []

    _build_parser().parse_args(argv)
    configure_lab_session(silent_stderr_default=False)

    mind = build_substrate_controller()
    print(f"Mosaic substrate  db={mind.db_path.resolve()}  namespace={CHAT_NAMESPACE}", flush=True)

    dev = next(mind.host.parameters()).device
    print(f"Model: {mind.llama_model_id}  device: {dev}", flush=True)
    print(f"Persistent memory: records={mind.memory.count()}  journal_rows={mind.journal.count()}", flush=True)

    start_background_stack(mind)
    print(f"Background consolidation: every {BROCA_BACKGROUND_INTERVAL_S:.1f}s", flush=True)

    if self_improve_enabled_from_env():
        print(
            "Self-improve worker: Docker/GitHub PR loop enabled "
            "(BROCA_SELF_IMPROVE_INTERVAL_S or default interval).",
            flush=True,
        )

    print("Substrate biases the LLM via grafts; the LLM still chooses the surface form.", flush=True)
    print("Commands: /quit /exit — leave.", flush=True)
    print(flush=True)

    messages: list[dict[str, str]] = []

    def _on_token(piece: str) -> None:
        sys.stdout.write(piece)
        sys.stdout.flush()

    try:
        while True:
            try:
                line = input("You> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.", flush=True)

                break

            if not line:
                continue

            low = line.lower()

            if low in {"/quit", "/exit", ":q"}:
                print("Bye.", flush=True)

                break

            messages.append({"role": "user", "content": line})
            sys.stdout.write("Assistant> ")
            sys.stdout.flush()

            try:
                _frame, reply = mind.chat_reply(
                    messages,
                    max_new_tokens=CHAT_MAX_NEW_TOKENS,
                    do_sample=CHAT_DO_SAMPLE,
                    temperature=CHAT_TEMPERATURE,
                    top_p=CHAT_TOP_P,
                    on_token=_on_token,
                )
            except KeyboardInterrupt:
                sys.stdout.write("\n[generation interrupted]\n")
                sys.stdout.flush()
                messages.pop()

                continue

            sys.stdout.write("\n")
            sys.stdout.flush()
            messages.append({"role": "assistant", "content": reply.strip() or "[empty reply]"})
    finally:
        stop_background_stack(mind)


def main() -> None:
    run_chat_repl()


if __name__ == "__main__":
    main()
