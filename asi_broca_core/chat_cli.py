"""Terminal chat: vanilla HF streaming or full Broca substrate.

Vanilla mode streams decoded chunks from ``model.generate``. With ``--broca``,
loads ``BrocaMind`` (SQLite-backed semantic memory + journal), routes each line
through ``comprehend``, then streams greedy tokens from ``LexicalPlanGraft``.

  python -m asi_broca_core.chat_cli
  python -m asi_broca_core.chat_cli --broca --broca-db runs/broca_chat.sqlite
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
from pathlib import Path
from typing import Any, Sequence

import torch

from .broca import BrocaMind, decode_generation
from .device_utils import pick_torch_device
from .llama_broca_host import load_llama_broca_host, quiet_transformers_benchmark_log_warnings, resolve_hf_hub_token
from .tokenizer import SPEECH_BRIDGE_PREFIX


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
        from transformers import TextIteratorStreamer
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
    except KeyboardInterrupt:
        sys.stdout.write("\n[generation interrupted]\n")
        sys.stdout.flush()
    thread.join(timeout=600.0)
    return "".join(chunks)


def _batch_rows(rows: Sequence[Sequence[int]], pad_id: int, *, device: torch.device):
    max_len = max(1, max(len(r) for r in rows))
    ids = torch.full((len(rows), max_len), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((len(rows), max_len), dtype=torch.bool, device=device)
    for i, row in enumerate(rows):
        if not row:
            continue
        ids[i, : len(row)] = torch.tensor(row, dtype=torch.long, device=device)
        mask[i, : len(row)] = True
    return ids, mask


def stream_broca_plan_tokens(
    host: torch.nn.Module,
    tokenizer: Any,
    plan_words: Sequence[str],
    *,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    """Greedy token emission under LexicalPlanGraft (streams decoded pieces)."""

    plan_ids = list(tokenizer.encode_plan_words(list(plan_words)))
    ids = list(tokenizer.encode(SPEECH_BRIDGE_PREFIX))
    generated: list[int] = []
    pad_id = int(tokenizer.pad_id)
    steps = range(min(max_new_tokens, len(plan_ids)))
    for step in steps:
        row = ids + generated
        batch_ids, mask = _batch_rows([row], pad_id, device=device)
        logits = host(
            batch_ids,
            mask,
            extra_state={
                "broca_plan_token_ids": torch.tensor([plan_ids], device=device),
                "broca_step": torch.tensor([step], device=device),
                "tokenizer": tokenizer,
            },
        )
        last = int(mask.long().sum().item()) - 1
        pred = int(logits[0, last].argmax().item())
        piece = tokenizer.decode_id(pred)
        sys.stdout.write(piece)
        sys.stdout.flush()
        generated.append(pred)
    sys.stdout.write("\n")
    sys.stdout.flush()
    return decode_generation(tokenizer, generated)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stream a local Hugging Face chat model in the terminal.")
    p.add_argument(
        "--model",
        default=os.environ.get("ASI_BROCA_MODEL_ID", "meta-llama/Llama-3.2-1B-Instruct"),
        help="HF model id (default: ASI_BROCA_MODEL_ID or Llama-3.2-1B-Instruct).",
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
    return p


def main() -> None:
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
        if args.system:
            print("(Note: --system applies only to vanilla HF chat mode, not --broca routing.)", flush=True)
        print("Substrate comprehension uses your raw text (e.g. 'where is ada ?'). Speech streams via lexical graft.", flush=True)
        print("Commands: /quit /exit — leave.", flush=True)
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

            frame, _speech_ref = mind.answer(line)
            plan = frame.speech_plan()
            print(f"[substrate intent={frame.intent} latent={frame.answer}]", flush=True)
            sys.stdout.write("Assistant> ")
            sys.stdout.flush()
            stream_broca_plan_tokens(
                mind.host,
                mind.tokenizer,
                plan,
                device=dev,
                max_new_tokens=args.max_new_tokens,
            )

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
