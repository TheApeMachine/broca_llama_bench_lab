"""Terminal chat: vanilla HF streaming or full Broca substrate.

Vanilla mode streams decoded chunks from ``model.generate``. With ``--broca``,
loads ``BrocaMind`` (SQLite-backed semantic memory + journal), routes each
line through ``comprehend``, then streams a free-form LLM reply whose
residual stream and logits are softly biased by the substrate via the graft
mechanism — the LLM still chooses surface form, fluency, and ordering.

  python -m core.chat_cli
  python -m core.chat_cli --broca --broca-db runs/broca_chat.sqlite

Logs: ``core`` sets DEBUG stderr logging when the package is imported unless
``LOG_SILENT=1`` or ``LOG_LEVEL=INFO`` (see ``logging_setup.py``).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any

import torch

from .broca import BrocaMind
from .device_utils import pick_torch_device
from .llama_broca_host import load_llama_broca_host, quiet_transformers_benchmark_log_warnings, resolve_hf_hub_token
from .logging_setup import configure_lab_logging

logger = logging.getLogger(__name__)


def _stream_reply(
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> str:
    try:
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
    except ImportError as e:  # pragma: no cover
        raise ImportError("Streaming chat requires transformers; pip install -r requirements-benchmark.txt") from e

    pad_id = getattr(tokenizer, "pad_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device),
        "streamer": streamer,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": pad_id if pad_id is not None else eos_id,
        "eos_token_id": eos_id,
    }
    if do_sample:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = max(float(temperature), 1e-5)
        gen_kwargs["top_p"] = float(top_p)
    else:
        gen_kwargs["do_sample"] = False

    chunks: list[str] = []
    stop_event = threading.Event()

    class _StopOnEvent(StoppingCriteria):
        def __init__(self, ev: threading.Event) -> None:
            self._ev = ev

        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:  # noqa: ARG002
            return self._ev.is_set()

    gen_kwargs["stopping_criteria"] = StoppingCriteriaList([_StopOnEvent(stop_event)])

    def _worker() -> None:
        with torch.inference_mode():
            model.generate(**gen_kwargs)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    try:
        for text in streamer:
            chunks.append(text)
            sys.stdout.write(text)
            sys.stdout.flush()
            if stop_event.is_set():
                break
    except KeyboardInterrupt:
        stop_event.set()
        sys.stdout.write("\n[generation interrupted]\n")
        sys.stdout.flush()
    thread.join(timeout=600.0)
    if thread.is_alive():
        logger.warning(
            "Vanilla HF stream: background generation thread still alive after join timeout (600s); "
            "model.generate may continue until completion."
        )
    return "".join(chunks)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stream a local Hugging Face chat model in the terminal.")
    p.add_argument(
        "--model",
        default=os.environ.get("MODEL_ID", "meta-llama/Llama-3.2-1B-Instruct"),
        help="HF model id (default: MODEL_ID or Llama-3.2-1B-Instruct).",
    )
    p.add_argument("--device", default=os.environ.get("ASI_DEVICE"), help="Torch device override (cpu, mps, cuda:0). Default: auto.")
    p.add_argument(
        "--token",
        default=None,
        help="HF hub token string, or omit to use HF_TOKEN / huggingface-cli login.",
    )
    p.add_argument("--max-new-tokens", type=int, default=512, help="Maximum new tokens per assistant reply.")
    p.add_argument("--system", default=None, help="Optional system message for the chat template.")
    p.add_argument("--sample", action="store_true", help="Use sampling (temperature / top-p) instead of greedy decoding.")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument(
        "--broca",
        action="store_true",
        help="Route through BrocaMind (semantic memory, faculties, SQLite journal). Speech streams via lexical graft.",
    )
    p.add_argument(
        "--broca-db",
        type=Path,
        default=None,
        help="SQLite path for BrocaMind (--broca). Default: runs/broca_chat.sqlite",
    )
    p.add_argument("--broca-namespace", default="chat", help="Semantic memory namespace for --broca.")
    p.add_argument("--no-background", action="store_true", help="Disable background memory consolidation in --broca mode.")
    p.add_argument("--background-interval", type=float, default=5.0, help="Seconds between background consolidation passes in --broca mode.")
    p.add_argument(
        "--debug-substrate",
        action="store_true",
        help="In --broca mode, print the cognitive frame's intent / answer / confidence after each reply.",
    )
    return p


def main() -> None:
    configure_lab_logging()
    args = _build_parser().parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    quiet_transformers_benchmark_log_warnings()

    resolved_device = pick_torch_device(args.device)
    token_kw: str | bool | None
    if args.token is None:
        token_kw = resolve_hf_hub_token(None)
    elif args.token.strip() == "":
        token_kw = True
    else:
        token_kw = args.token.strip()

    if args.broca:
        db_path = args.broca_db or Path("runs/broca_chat.sqlite")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Broca substrate  db={db_path.resolve()}  namespace={args.broca_namespace}", flush=True)
        mind = BrocaMind(
            seed=0,
            db_path=db_path,
            namespace=args.broca_namespace,
            llama_model_id=args.model,
            device=resolved_device,
            hf_token=token_kw,
        )
        dev = resolved_device if isinstance(resolved_device, torch.device) else torch.device(str(resolved_device))
        print(f"Model: {args.model}  device: {dev}", flush=True)
        print(f"Persistent memory: records={mind.memory.count()}  journal_rows={mind.journal.count()}", flush=True)
        if args.no_background:
            print("Background consolidation: off", flush=True)
        else:
            validated_interval = max(0.1, float(args.background_interval))
            mind.start_background(interval_s=validated_interval)
            print(f"Background consolidation: every {validated_interval:.1f}s", flush=True)
        print("Substrate biases the LLM via grafts; the LLM still chooses the surface form.", flush=True)
        print("Commands: /quit /exit — leave.", flush=True)
        print(flush=True)

        messages: list[dict[str, str]] = []
        supports_system_messages = getattr(mind, "supports_system_messages", None)
        if supports_system_messages is None:
            supports_system_messages = not isinstance(mind, BrocaMind)
        if args.system:
            stripped = args.system.strip()
            if stripped and supports_system_messages:
                messages.append({"role": "system", "content": stripped})
            elif stripped:
                print(
                    "(Note: BrocaMind does not use HF-style system prompts in routing; omit --system or use vanilla mode.)",
                    flush=True,
                )

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
                    frame, reply = mind.chat_reply(
                        messages,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=args.sample,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        on_token=_on_token,
                    )
                except KeyboardInterrupt:
                    sys.stdout.write("\n[generation interrupted]\n")
                    sys.stdout.flush()
                    messages.pop()
                    continue
                sys.stdout.write("\n")
                sys.stdout.flush()
                if args.debug_substrate:
                    print(
                        f"[substrate intent={frame.intent} subject={frame.subject or '-'} "
                        f"answer={frame.answer} conf={frame.confidence:.2f}]",
                        flush=True,
                    )
                messages.append({"role": "assistant", "content": reply.strip() or "[empty reply]"})
        finally:
            mind.stop_background()

        return

    host, hf_wrap = load_llama_broca_host(
        args.model,
        device=resolved_device,
        token=token_kw,
        trust_remote_code=args.trust_remote_code,
    )
    model = host.llm
    tokenizer = hf_wrap.inner
    tmpl = getattr(tokenizer, "apply_chat_template", None)
    if not callable(tmpl):
        print("Tokenizer has no apply_chat_template; use an Instruct/chat checkpoint.", file=sys.stderr)
        sys.exit(2)

    messages: list[dict[str, str]] = []
    if args.system:
        messages.append({"role": "system", "content": args.system.strip()})

    print(f"Model: {args.model}  device: {resolved_device}", flush=True)
    print("Vanilla HF chat mode: no SQLite/Broca persistent memory. Start with --broca to use the substrate.", flush=True)
    print("Commands: /quit /exit — leave; blank lines ignored. Ctrl+C interrupts the current reply.", flush=True)
    print(flush=True)

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
        try:
            prompt_t = tmpl(messages, add_generation_prompt=True, return_tensors="pt")
        except Exception as exc:  # pragma: no cover
            print(f"apply_chat_template failed: {exc}", file=sys.stderr)
            messages.pop()
            continue
        if isinstance(prompt_t, torch.Tensor):
            input_ids = prompt_t.to(resolved_device)
        else:
            input_ids = prompt_t["input_ids"].to(resolved_device)

        sys.stdout.write("Assistant> ")
        sys.stdout.flush()
        reply = _stream_reply(
            model,
            tokenizer,
            input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.sample,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print(flush=True)
        messages.append({"role": "assistant", "content": reply.strip() or "[empty reply]"})


if __name__ == "__main__":
    main()
