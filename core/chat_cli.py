"""Terminal chat with the full Broca substrate (always on).

  python -m core.chat_cli

Model and device follow ``MODEL_ID`` / ``M_DEVICE`` (or auto device). Substrate
SQLite is ``runs/broca_substrate.sqlite`` unless tests set ``MOSAIC_TEST_DB``.

Logs: ``core`` sets DEBUG stderr logging when the package is imported unless
``LOG_SILENT=1`` or ``LOG_LEVEL=INFO`` (see ``logging_setup.py``).
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

import torch

from .broca import BrocaMind
from .device_utils import pick_torch_device
from .llama_broca_host import quiet_transformers_benchmark_log_warnings, resolve_hf_hub_token
from .logging_setup import configure_lab_logging
from .substrate_runtime import (
    BROCA_BACKGROUND_INTERVAL_S,
    CHAT_DO_SAMPLE,
    CHAT_MAX_NEW_TOKENS,
    CHAT_NAMESPACE,
    CHAT_TEMPERATURE,
    CHAT_TOP_P,
    default_model_id,
    default_substrate_sqlite_path,
    ensure_parent_dir,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Mosaic chat CLI (full substrate; no tuning flags).")
    p.add_argument("-h", "--help", action="help", help="Show this message and exit.")
    return p


def _default_cli_device_env() -> str | None:
    raw_m = os.environ.get("M_DEVICE")
    if raw_m is not None and str(raw_m).strip() != "":
        return str(raw_m).strip()
    legacy = os.environ.get("ASI_DEVICE")
    if legacy is not None and str(legacy).strip() != "":
        warnings.warn(
            "ASI_DEVICE is deprecated; set M_DEVICE for the default torch device override.",
            DeprecationWarning,
            stacklevel=2,
        )
        return str(legacy).strip()
    return None


def main() -> None:
    configure_lab_logging()
    _build_parser().parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    quiet_transformers_benchmark_log_warnings()

    model_id = default_model_id()
    resolved_device = pick_torch_device(_default_cli_device_env())
    token_kw: str | bool | None = resolve_hf_hub_token(None)

    db_path = default_substrate_sqlite_path()
    ensure_parent_dir(db_path)
    print(f"Broca substrate  db={db_path.resolve()}  namespace={CHAT_NAMESPACE}", flush=True)
    mind = BrocaMind(
        seed=0,
        db_path=db_path,
        namespace=CHAT_NAMESPACE,
        llama_model_id=model_id,
        device=resolved_device,
        hf_token=token_kw,
    )
    dev = resolved_device if isinstance(resolved_device, torch.device) else torch.device(str(resolved_device))
    print(f"Model: {model_id}  device: {dev}", flush=True)
    print(f"Persistent memory: records={mind.memory.count()}  journal_rows={mind.journal.count()}", flush=True)
    mind.start_background(interval_s=BROCA_BACKGROUND_INTERVAL_S)
    print(f"Background consolidation: every {BROCA_BACKGROUND_INTERVAL_S:.1f}s", flush=True)

    si_env = os.environ.get("BROCA_SELF_IMPROVE", "").strip().lower() in {"1", "true", "yes", "on"}
    if si_env:
        mind.start_self_improve_worker(interval_s=None, enabled=True)
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
        mind.stop_background()
        mind.stop_self_improve_worker()


if __name__ == "__main__":
    main()
