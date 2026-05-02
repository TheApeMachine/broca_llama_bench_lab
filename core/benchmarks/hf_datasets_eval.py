
"""Native Hugging Face-datasets benchmark harness for the Broca/Llama stack.

This module intentionally does not depend on EleutherAI's lm-eval harness.  It
loads public benchmark datasets through ``datasets.load_dataset`` and scores
vanilla ``AutoModelForCausalLM``, a ``LlamaBrocaHost`` replay on the same
weights, and (for Llama checkpoints) full :class:`~core.cognition.substrate.SubstrateController` on that
host --- by length-normalized continuation log-likelihood (multiple choice) or
deterministic generation with normalized exact matching where applicable.

Real runs use the unified entry point::

  HF_TOKEN=... python -m core.benchmarks
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as _dt
import itertools
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Protocol, Sequence

import torch
import torch.nn.functional as F

from core.cognition.substrate import SubstrateController
from core.system.device import inference_dtype, pick_torch_device
from core.host.hf_tokenizer_compat import HuggingFaceBrocaTokenizer
from core.host.llama_broca_host import (
    LlamaBrocaHost,
    load_llama_broca_host,
    quiet_transformers_benchmark_log_warnings,
    resolve_hf_hub_token,
)
from core.substrate.runtime import CHAT_NAMESPACE, default_substrate_sqlite_path


DEFAULT_LLAMA_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
DEFAULT_NATIVE_PRESETS: dict[str, list[str]] = {
    "smoke": ["boolq", "piqa"],
    "quick": ["boolq", "piqa", "arc_easy", "winogrande"],
    "standard": ["boolq", "piqa", "arc_easy", "arc_challenge", "winogrande", "hellaswag"],
    "reasoning": ["arc_challenge", "hellaswag", "winogrande", "commonsenseqa", "openbookqa"],
    "full": ["boolq", "piqa", "arc_easy", "arc_challenge", "winogrande", "hellaswag", "commonsenseqa", "openbookqa", "mmlu_abstract_algebra", "gsm8k"],
}

_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
_WORD_RE = re.compile(r"[a-z0-9]+")
_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


@dataclass(frozen=True)
class BenchmarkExample:
    """A normalized benchmark row.

    For ``mode='mc'``, ``choices`` and ``gold_index`` are used.  The harness
    scores each continuation under the model and picks the maximum.
    ``gold_index`` may be ``None`` when the split row has no usable gold label
    (e.g. malformed Winogrande answers).
    For ``mode='generate'``, ``expected_text`` is compared with parsed model
    generations.
    """

    task: str
    id: str
    prompt: str
    choices: tuple[str, ...] = ()
    gold_index: int | None = None
    mode: str = "mc"
    expected_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class TaskSpec:
    name: str
    dataset_name: str
    config_name: str | None
    split: str
    builder: Callable[[Mapping[str, Any], int], BenchmarkExample | None]
    description: str
    mode: str = "mc"


def _clean(s: Any) -> str:
    return " ".join(str(s).replace("\n", " ").split())


def _choice_label_at(i: int) -> str:
    """Map 0-based choice index to Excel-style labels (A..Z, AA, AB, ...)."""

    n = i + 1
    label = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        label = chr(65 + r) + label
    return label


def _lettered_prompt(question: str, choices: Sequence[str], *, prefix: str = "Question") -> str:
    lines = [f"{prefix}: {_clean(question)}", "Choices:"]
    for i, choice in enumerate(choices):
        label = _choice_label_at(i)
        lines.append(f"{label}. {_clean(choice)}")
    lines.append("Answer:")
    return "\n".join(lines)


def _choice_labels(n: int) -> tuple[str, ...]:
    return tuple(f" {_choice_label_at(i)}" for i in range(n))


def _find_answer_key_index(labels: Sequence[Any], answer_key: Any) -> int | None:
    ak = str(answer_key).strip()
    normalized = [str(x).strip() for x in labels]
    if ak in normalized:
        return normalized.index(ak)
    ak_upper = ak.upper()
    normalized_upper = [x.upper() for x in normalized]
    if ak_upper in normalized_upper:
        return normalized_upper.index(ak_upper)
    # Some datasets use 1/2/3 while choices are A/B/C or vice versa.
    if ak.isdigit():
        idx = int(ak) - 1
        if 0 <= idx < len(labels):
            return idx
    if ak_upper in _LETTERS:
        idx = _LETTERS.index(ak_upper)
        if 0 <= idx < len(labels):
            return idx
    return None


def build_boolq(row: Mapping[str, Any], idx: int) -> BenchmarkExample:
    passage = _clean(row["passage"])
    question = _clean(row["question"])
    prompt = f"Passage:\n{passage}\n\nQuestion: {question}\nIs the answer yes or no?\nAnswer:"
    gold = 1 if bool(row["answer"]) else 0
    return BenchmarkExample("boolq", str(idx), prompt, (" no", " yes"), gold, metadata={"dataset": "boolq"})


def build_piqa(row: Mapping[str, Any], idx: int) -> BenchmarkExample:
    sol1 = _clean(row["sol1"])
    sol2 = _clean(row["sol2"])
    prompt = _lettered_prompt(row["goal"], [sol1, sol2], prefix="Goal")
    return BenchmarkExample("piqa", str(idx), prompt, _choice_labels(2), int(row["label"]), metadata={"dataset": "lighteval/piqa"})


def build_arc(task_name: str) -> Callable[[Mapping[str, Any], int], BenchmarkExample | None]:
    def _builder(row: Mapping[str, Any], idx: int) -> BenchmarkExample | None:
        choices_obj = row["choices"]
        labels = list(choices_obj["label"])
        texts = [_clean(x) for x in choices_obj["text"]]
        gold = _find_answer_key_index(labels, row["answerKey"])
        if gold is None:
            return None
        prompt = _lettered_prompt(row["question"], texts)
        labels_for_model = tuple(f" {str(label).strip()}" for label in labels)
        return BenchmarkExample(task_name, str(idx), prompt, labels_for_model, gold, metadata={"labels": labels, "dataset": "allenai/ai2_arc"})
    return _builder


def build_winogrande(row: Mapping[str, Any], idx: int) -> BenchmarkExample:
    sentence = str(row["sentence"])
    if "_" in sentence:
        prefix, suffix = sentence.split("_", 1)
    else:
        prefix, suffix = sentence, ""
    choices = (str(row["option1"]) + suffix, str(row["option2"]) + suffix)
    raw = row.get("answer")
    gold: int | None
    if raw is None or (isinstance(raw, str) and not str(raw).strip()):
        gold = None
    else:
        try:
            parsed = int(str(raw).strip())
        except ValueError:
            gold = None
        else:
            gold = parsed - 1 if parsed in (1, 2) else None
    return BenchmarkExample(
        "winogrande",
        str(idx),
        prefix,
        choices,
        gold,
        metadata={"suffix": suffix, "dataset": "winogrande/winogrande_xl"},
    )


def build_hellaswag(row: Mapping[str, Any], idx: int) -> BenchmarkExample:
    ctx = _clean(row.get("ctx", ""))
    endings = tuple(" " + _clean(x) for x in row["endings"])
    label = int(row["label"])
    return BenchmarkExample("hellaswag", str(idx), ctx, endings, label, metadata={"dataset": "hellaswag"})


def build_commonsenseqa(row: Mapping[str, Any], idx: int) -> BenchmarkExample | None:
    choice_obj = row["choices"]
    labels = list(choice_obj["label"])
    texts = [_clean(x) for x in choice_obj["text"]]
    gold = _find_answer_key_index(labels, row["answerKey"])
    if gold is None:
        return None
    prompt = _lettered_prompt(row["question"], texts)
    return BenchmarkExample("commonsenseqa", str(idx), prompt, _choice_labels(len(texts)), gold, metadata={"dataset": "commonsense_qa", "labels": labels})


def build_openbookqa(row: Mapping[str, Any], idx: int) -> BenchmarkExample | None:
    choice_obj = row["choices"]
    labels = list(choice_obj["label"])
    texts = [_clean(x) for x in choice_obj["text"]]
    gold = _find_answer_key_index(labels, row["answerKey"])
    if gold is None:
        return None
    prompt = _lettered_prompt(row["question_stem"], texts)
    return BenchmarkExample("openbookqa", str(idx), prompt, tuple(f" {l}" for l in labels), gold, metadata={"dataset": "openbookqa", "labels": labels})


def build_mmlu(row: Mapping[str, Any], idx: int) -> BenchmarkExample:
    choices = [_clean(x) for x in row["choices"]]
    prompt = _lettered_prompt(row["question"], choices)
    return BenchmarkExample("mmlu_abstract_algebra", str(idx), prompt, _choice_labels(len(choices)), int(row["answer"]), metadata={"dataset": "cais/mmlu"})


def _extract_gsm8k_gold(answer: str) -> str:
    text = str(answer)
    marker = "####"
    if marker in text:
        return text.split(marker, 1)[1].strip()
    nums = _NUMBER_RE.findall(text)
    return nums[-1].replace(",", "") if nums else text.strip()


def build_gsm8k(row: Mapping[str, Any], idx: int) -> BenchmarkExample:
    q = _clean(row["question"])
    prompt = f"Solve the math problem. Show brief reasoning, then give the final answer after 'Answer:'.\n\nProblem: {q}\nAnswer:"
    expected = _extract_gsm8k_gold(str(row["answer"]))
    return BenchmarkExample(
        "gsm8k",
        str(idx),
        prompt,
        mode="generate",
        expected_text=expected,
        metadata={"dataset": "openai/gsm8k"},
    )


TASK_REGISTRY: dict[str, TaskSpec] = {
    "boolq": TaskSpec("boolq", "boolq", None, "validation", build_boolq, "BoolQ yes/no reading comprehension."),
    # ``piqa`` / ``YBisk/piqa`` ship dataset scripts; recent ``datasets`` rejects those.  ``lighteval/piqa``
    # is the same splits/columns loaded from parquet.
    "piqa": TaskSpec("piqa", "lighteval/piqa", None, "validation", build_piqa, "PIQA physical commonsense two-choice benchmark."),
    "arc_easy": TaskSpec("arc_easy", "allenai/ai2_arc", "ARC-Easy", "validation", build_arc("arc_easy"), "AI2 ARC Easy science QA."),
    "arc_challenge": TaskSpec("arc_challenge", "allenai/ai2_arc", "ARC-Challenge", "validation", build_arc("arc_challenge"), "AI2 ARC Challenge science QA."),
    "winogrande": TaskSpec("winogrande", "winogrande", "winogrande_xl", "validation", build_winogrande, "WinoGrande commonsense cloze task."),
    "hellaswag": TaskSpec("hellaswag", "hellaswag", None, "validation", build_hellaswag, "HellaSwag commonsense continuation benchmark."),
    "commonsenseqa": TaskSpec("commonsenseqa", "commonsense_qa", None, "validation", build_commonsenseqa, "CommonsenseQA multiple-choice commonsense benchmark."),
    "openbookqa": TaskSpec("openbookqa", "openbookqa", "main", "validation", build_openbookqa, "OpenBookQA elementary science benchmark."),
    "mmlu_abstract_algebra": TaskSpec("mmlu_abstract_algebra", "cais/mmlu", "abstract_algebra", "test", build_mmlu, "MMLU abstract algebra subset."),
    "gsm8k": TaskSpec("gsm8k", "openai/gsm8k", "main", "test", build_gsm8k, "GSM8K grade-school math generation benchmark.", mode="generate"),
}


def resolve_task_names(tasks: str | Sequence[str] | None = None, *, preset: str | None = None) -> list[str]:
    if tasks is None or (isinstance(tasks, str) and not tasks.strip()):
        preset = preset or "quick"
        if preset not in DEFAULT_NATIVE_PRESETS:
            raise ValueError(f"unknown native preset {preset!r}; choices={sorted(DEFAULT_NATIVE_PRESETS)}")
        return list(DEFAULT_NATIVE_PRESETS[preset])
    if isinstance(tasks, str):
        out = [t.strip() for t in tasks.split(",") if t.strip()]
    else:
        out = [str(t).strip() for t in tasks if str(t).strip()]
    unknown = [t for t in out if t not in TASK_REGISTRY]
    if unknown:
        raise ValueError(f"unknown task(s): {unknown}; choices={sorted(TASK_REGISTRY)}")
    return out


def _take_rows(ds: Iterable[Mapping[str, Any]], *, limit: int | None, seed: int, shuffle: bool) -> list[tuple[int, Mapping[str, Any]]]:
    indexed: list[tuple[int, Mapping[str, Any]]] = []
    for i, row in enumerate(ds):
        indexed.append((i, row))
        if limit is not None and not shuffle and len(indexed) >= limit:
            break
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(indexed)
        if limit is not None:
            indexed = indexed[:limit]
    return indexed if limit is None else indexed[:limit]


def load_task_examples(
    task_name: str,
    *,
    limit: int | None = None,
    split: str | None = None,
    streaming: bool = False,
    seed: int = 0,
    shuffle: bool = False,
) -> list[BenchmarkExample]:
    """Load and normalize one benchmark task through Hugging Face Datasets."""

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - depends on benchmark extra
        raise ImportError(
            "Native benchmarks require `datasets`; run `uv sync --extra benchmark` or `pip install -e \".[benchmark]\"`."
        ) from exc

    spec = TASK_REGISTRY[task_name]
    split_name = split or spec.split
    if spec.config_name is None:
        ds = load_dataset(spec.dataset_name, split=split_name, streaming=streaming)
    else:
        ds = load_dataset(spec.dataset_name, spec.config_name, split=split_name, streaming=streaming)

    if streaming:
        raw_rows = list(itertools.islice(enumerate(ds), limit or 1000000000))
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(raw_rows)
            if limit is not None:
                raw_rows = raw_rows[:limit]
    else:
        if shuffle and hasattr(ds, "shuffle"):
            ds = ds.shuffle(seed=seed)
        raw_rows = [(i, ds[i]) for i in range(len(ds) if limit is None else min(limit, len(ds)))]

    examples: list[BenchmarkExample] = []
    for i, row in raw_rows:
        ex = spec.builder(row, i)
        if ex is not None:
            examples.append(ex)
    return examples


@dataclass
class ChoicePrediction:
    pred_index: int
    gold_index: int
    correct: bool
    scores: list[float]
    raw_logprobs: list[float]
    token_counts: list[int]


def normalize_answer_text(text: str) -> str:
    return " ".join(_WORD_RE.findall(str(text).lower()))


def normalize_number_text(text: str) -> str:
    nums = _NUMBER_RE.findall(str(text).replace(",", ""))
    if not nums:
        return normalize_answer_text(text)
    x = nums[-1].lstrip("+")
    if x.endswith(".0"):
        x = x[:-2]
    return x


def _shared_hf_format_prompt(tokenizer: Any, prompt: str, *, chat_template: bool) -> str:
    if not chat_template:
        return prompt
    tmpl = getattr(tokenizer, "apply_chat_template", None)
    if not callable(tmpl):
        return prompt
    return str(tmpl([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True))


def _shared_hf_encode_context_choice(
    tokenizer: Any,
    max_seq_len: int,
    context: str,
    choice: str,
) -> tuple[list[int], int, int]:
    tok = tokenizer
    context_ids = list(tok.encode(context, add_special_tokens=False))
    choice_ids = list(tok.encode(choice, add_special_tokens=False))
    if not choice_ids:
        choice_ids = list(tok.encode(" " + choice, add_special_tokens=False))
    bos = getattr(tok, "bos_token_id", None)
    prefix_extra = 1 if bos is not None else 0
    budget_context = max(0, max_seq_len - len(choice_ids) - prefix_extra)
    if len(context_ids) > budget_context:
        context_ids = context_ids[-budget_context:]
    ids = ([int(bos)] if bos is not None else []) + context_ids + choice_ids
    cont_start = prefix_extra + len(context_ids)
    return ids, cont_start, len(choice_ids)


class HFLocalCausalLM:
    """A small local evaluator around ``transformers.AutoModelForCausalLM``."""

    def __init__(
        self,
        model_id: str = DEFAULT_LLAMA_MODEL,
        *,
        device: str | torch.device | None = None,
        token: str | bool | None = None,
        dtype: torch.dtype | None = None,
        max_seq_len: int | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        try:
            quiet_transformers_benchmark_log_warnings()
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on benchmark extra
            raise ImportError(
                "Native benchmarks require `transformers`; run `uv sync --extra benchmark` or `pip install -e \".[benchmark]\"`."
            ) from exc

        self.device = device if isinstance(device, torch.device) else pick_torch_device(device)
        self.model_id = model_id
        self.token_arg = resolve_hf_hub_token(token)
        self.dtype = dtype or inference_dtype(self.device)
        attn_impl = "eager" if self.device.type == "mps" else "sdpa"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
            token=self.token_arg,
            trust_remote_code=trust_remote_code,
            # Llama/BPE models: True is WordPiece-focused and emits noisy transformers warnings while being ignored anyway.
            clean_up_tokenization_spaces=False,
        )
        if getattr(self.tokenizer, "pad_token_id", None) is None and getattr(self.tokenizer, "eos_token_id", None) is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = int(self.tokenizer.eos_token_id)
        model_kwargs = dict(
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            token=self.token_arg,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_impl,
        )
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs).to(self.device)
        except TypeError:
            # Older/non-Llama architectures may not accept attn_implementation.
            model_kwargs.pop("attn_implementation", None)
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs).to(self.device)
        self.model.eval()
        cfg_len = int(getattr(self.model.config, "max_position_embeddings", 4096))
        self.max_seq_len = int(max_seq_len or min(cfg_len, 4096))

    def format_prompt(self, prompt: str, *, chat_template: bool = False) -> str:
        return _shared_hf_format_prompt(self.tokenizer, prompt, chat_template=chat_template)

    def _encode_context_choice(self, context: str, choice: str) -> tuple[list[int], int, int]:
        return _shared_hf_encode_context_choice(self.tokenizer, self.max_seq_len, context, choice)

    @torch.no_grad()
    def score_choices(self, prompt: str, choices: Sequence[str], *, normalize: bool = True, chat_template: bool = False) -> tuple[list[float], list[float], list[int]]:
        context = self.format_prompt(prompt, chat_template=chat_template)
        encoded = [self._encode_context_choice(context, c) for c in choices]
        max_len = max(len(ids) for ids, _, _ in encoded)
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.tokenizer, "eos_token_id", 0) or 0
        input_ids = torch.full((len(encoded), max_len), int(pad_id), dtype=torch.long, device=self.device)
        attn = torch.zeros((len(encoded), max_len), dtype=torch.long, device=self.device)
        for i, (ids, _, _) in enumerate(encoded):
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=self.device)
            attn[i, : len(ids)] = 1
        out = self.model(input_ids=input_ids, attention_mask=attn)
        logits = out.logits.float()
        logprobs = F.log_softmax(logits, dim=-1)

        raw_scores: list[float] = []
        norm_scores: list[float] = []
        token_counts: list[int] = []
        for row, (ids, cont_start, cont_len) in enumerate(encoded):
            total = 0.0
            usable = 0
            # p(token[j]) is predicted by logits[j-1].
            for j in range(cont_start, cont_start + cont_len):
                if j <= 0 or j >= len(ids):
                    continue
                total += float(logprobs[row, j - 1, int(ids[j])].item())
                usable += 1
            raw_scores.append(total)
            token_counts.append(max(usable, 1))
            norm_scores.append(total / max(usable, 1) if normalize else total)
        return norm_scores, raw_scores, token_counts

    @torch.no_grad()
    def generate(self, prompt: str, *, max_new_tokens: int = 128, chat_template: bool = True) -> str:
        text = self.format_prompt(prompt, chat_template=chat_template)
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=getattr(self.tokenizer, "pad_token_id", None),
            eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
        )
        gen_ids = out[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(
            gen_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()


class HFLocalLlamaBrocaShell:
    """Scores MC/generation through ``LlamaBrocaHost`` so you can diff against ``HFLocalCausalLM``. Uses the same weights as the vanilla run."""

    def __init__(
        self,
        model_id: str = DEFAULT_LLAMA_MODEL,
        *,
        device: str | torch.device | None = None,
        token: str | bool | None = None,
        max_seq_len: int | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        quiet_transformers_benchmark_log_warnings()
        self.device = device if isinstance(device, torch.device) else pick_torch_device(device)
        self.model_id = model_id
        self.host, wrapped = load_llama_broca_host(
            model_id,
            device=self.device,
            token=resolve_hf_hub_token(token),
            trust_remote_code=trust_remote_code,
        )
        self.tokenizer = wrapped.inner
        self.host.eval()
        cfg_len = int(getattr(self.host.config, "max_position_embeddings", 4096))
        self.max_seq_len = int(max_seq_len or min(cfg_len, 4096))

    @classmethod
    def wrapping_same_lm(cls, hf_local: HFLocalCausalLM) -> HFLocalLlamaBrocaShell:
        """Route logits through ``LlamaBrocaHost`` using the already-loaded HF causal LM (no second hub fetch)."""

        quiet_transformers_benchmark_log_warnings()
        self = cls.__new__(cls)
        self.device = hf_local.device
        self.model_id = hf_local.model_id
        self.host = LlamaBrocaHost(hf_local.model)
        self.tokenizer = hf_local.tokenizer
        self.host.eval()
        self.max_seq_len = hf_local.max_seq_len
        return self

    def format_prompt(self, prompt: str, *, chat_template: bool = False) -> str:
        return _shared_hf_format_prompt(self.tokenizer, prompt, chat_template=chat_template)

    def _encode_context_choice(self, context: str, choice: str) -> tuple[list[int], int, int]:
        return _shared_hf_encode_context_choice(self.tokenizer, self.max_seq_len, context, choice)

    @torch.no_grad()
    def score_choices(self, prompt: str, choices: Sequence[str], *, normalize: bool = True, chat_template: bool = False) -> tuple[list[float], list[float], list[int]]:
        context = self.format_prompt(prompt, chat_template=chat_template)
        encoded = [self._encode_context_choice(context, c) for c in choices]
        max_len = max(len(ids) for ids, _, _ in encoded)
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.tokenizer, "eos_token_id", 0) or 0
        input_ids = torch.full((len(encoded), max_len), int(pad_id), dtype=torch.long, device=self.device)
        attn = torch.zeros((len(encoded), max_len), dtype=torch.bool, device=self.device)
        for i, (ids, _, _) in enumerate(encoded):
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=self.device)
            attn[i, : len(ids)] = True
        logits = self.host(input_ids, attention_mask=attn).float()
        logprobs = F.log_softmax(logits, dim=-1)

        raw_scores: list[float] = []
        norm_scores: list[float] = []
        token_counts: list[int] = []
        for row, (ids, cont_start, cont_len) in enumerate(encoded):
            total = 0.0
            usable = 0
            for j in range(cont_start, cont_start + cont_len):
                if j <= 0 or j >= len(ids):
                    continue
                total += float(logprobs[row, j - 1, int(ids[j])].item())
                usable += 1
            raw_scores.append(total)
            token_counts.append(max(usable, 1))
            norm_scores.append(total / max(usable, 1) if normalize else total)
        return norm_scores, raw_scores, token_counts

    @torch.no_grad()
    def generate(self, prompt: str, *, max_new_tokens: int = 128, chat_template: bool = True) -> str:
        text = self.format_prompt(prompt, chat_template=chat_template)
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.device)
        out = self.host.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=getattr(self.tokenizer, "pad_token_id", None),
            eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
        )
        gen_ids = out[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(
            gen_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()


class HFLocalSubstrateBench:
    """Full substrate stack: comprehend → graft-biased ``LlamaBrocaHost`` forward.

    Expects ``mind.host`` to be the same :class:`~core.host.llama_broca_host.LlamaBrocaHost`
    instance already used for the paired ``broca_shell`` pass (no second weight load).
    """

    def __init__(self, mind: SubstrateController, hf_local: HFLocalCausalLM) -> None:
        self.mind = mind
        self.device = hf_local.device
        self.max_seq_len = hf_local.max_seq_len
        self.tokenizer = mind.tokenizer.inner

    def format_prompt(self, prompt: str, *, chat_template: bool = False) -> str:
        return _shared_hf_format_prompt(self.tokenizer, prompt, chat_template=chat_template)

    def _encode_context_choice(self, context: str, choice: str) -> tuple[list[int], int, int]:
        return _shared_hf_encode_context_choice(self.tokenizer, self.max_seq_len, context, choice)

    @torch.no_grad()
    def score_choices(
        self, prompt: str, choices: Sequence[str], *, normalize: bool = True, chat_template: bool = False
    ) -> tuple[list[float], list[float], list[int]]:
        context = self.format_prompt(prompt, chat_template=chat_template)
        frame = self.mind.comprehend(context)
        feats = self.mind.broca_features_from_frame(frame).to(self.device)
        bias = self.mind.content_logit_bias_from_frame(frame)
        substrate_confidence = float(max(0.0, min(1.0, float(frame.confidence))))
        encoded = [self._encode_context_choice(context, c) for c in choices]
        max_len = max(len(ids) for ids, _, _ in encoded)
        substrate_inertia = math.log1p(float(max(len(ids) for ids, _, _ in encoded)))
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.tokenizer, "eos_token_id", 0) or 0
        input_ids = torch.full((len(encoded), max_len), int(pad_id), dtype=torch.long, device=self.device)
        attn = torch.zeros((len(encoded), max_len), dtype=torch.bool, device=self.device)
        for i, (ids, _, _) in enumerate(encoded):
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=self.device)
            attn[i, : len(ids)] = True
        extra_state: dict[str, Any] = {
            "tokenizer": self.mind.tokenizer,
            "broca_features": feats,
            "substrate_confidence": substrate_confidence,
            "substrate_inertia": float(substrate_inertia),
        }
        if bias:
            extra_state["broca_logit_bias"] = bias
            extra_state["broca_logit_bias_decay"] = 1.0
        logits = self.mind.host(input_ids, attention_mask=attn, extra_state=extra_state).float()
        logprobs = F.log_softmax(logits, dim=-1)
        raw_scores: list[float] = []
        norm_scores: list[float] = []
        token_counts: list[int] = []
        for row, (ids, cont_start, cont_len) in enumerate(encoded):
            total = 0.0
            usable = 0
            for j in range(cont_start, cont_start + cont_len):
                if j <= 0 or j >= len(ids):
                    continue
                total += float(logprobs[row, j - 1, int(ids[j])].item())
                usable += 1
            raw_scores.append(total)
            token_counts.append(max(usable, 1))
            norm_scores.append(total / max(usable, 1) if normalize else total)
        return norm_scores, raw_scores, token_counts

    @torch.no_grad()
    def generate(self, prompt: str, *, max_new_tokens: int = 128, chat_template: bool = True) -> str:
        text = self.format_prompt(prompt, chat_template=chat_template)
        _frame, reply = self.mind.chat_reply(
            [{"role": "user", "content": text}],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
        )
        return reply.strip()


class BenchmarkBackendProtocol(Protocol):
    """Structural protocol for MC scoring and greedy generation backends."""

    def score_choices(
        self,
        prompt: str,
        choices: Sequence[str],
        *,
        normalize: bool = True,
        chat_template: bool = False,
    ) -> tuple[list[float], list[float], list[int]]:
        ...

    def generate(self, prompt: str, *, max_new_tokens: int = 128, chat_template: bool = True) -> str:
        ...


def evaluate_example(
    backend: BenchmarkBackendProtocol | HFLocalCausalLM,
    ex: BenchmarkExample,
    *,
    normalize: bool = True,
    chat_template: bool = False,
    generation_max_new_tokens: int = 128,
) -> dict[str, Any]:
    if ex.mode == "mc":
        scores, raw, counts = backend.score_choices(ex.prompt, ex.choices, normalize=normalize, chat_template=chat_template)
        pred = int(max(range(len(scores)), key=lambda i: scores[i]))
        gold = int(ex.gold_index) if ex.gold_index is not None else -1
        return {
            "task": ex.task,
            "id": ex.id,
            "mode": ex.mode,
            "prompt": ex.prompt,
            "choices": list(ex.choices),
            "gold_index": gold,
            "pred_index": pred,
            "correct": pred == gold,
            "scores": scores,
            "raw_logprobs": raw,
            "token_counts": counts,
            "metadata": ex.metadata,
        }
    if ex.mode == "generate":
        pred_text = backend.generate(ex.prompt, max_new_tokens=generation_max_new_tokens, chat_template=chat_template)
        expected = str(ex.expected_text or "")
        # GSM8K is numeric; other future gen tasks can use normalized exact.
        if ex.task == "gsm8k":
            pred_norm = normalize_number_text(pred_text)
            gold_norm = normalize_number_text(expected)
        else:
            pred_norm = normalize_answer_text(pred_text)
            gold_norm = normalize_answer_text(expected)
        return {
            "task": ex.task,
            "id": ex.id,
            "mode": ex.mode,
            "prompt": ex.prompt,
            "expected_text": expected,
            "pred_text": pred_text,
            "pred_norm": pred_norm,
            "gold_norm": gold_norm,
            "correct": pred_norm == gold_norm,
            "metadata": ex.metadata,
        }
    raise ValueError(f"unknown example mode {ex.mode!r}")


def summarize_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "accuracy": 0.0}
    correct = sum(1 for r in rows if bool(r.get("correct")))
    return {
        "n": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
    }


def export_csv(results: Mapping[str, Any], out_dir: Path) -> None:
    """Write per-task metrics and aggregate summary as CSV files."""

    out_dir = Path(out_dir)
    per_task = results.get("per_task") or {}
    fieldnames = ("task", "n", "correct", "accuracy", "seconds")
    task_rows: list[dict[str, Any]] = []
    for task in sorted(per_task.keys()):
        m = per_task[task]
        if not isinstance(m, Mapping):
            continue
        task_rows.append({
            "task": task,
            "n": m.get("n", ""),
            "correct": m.get("correct", ""),
            "accuracy": m.get("accuracy", ""),
            "seconds": m.get("seconds", ""),
        })
    per_path = out_dir / "per_task.csv"
    if task_rows:
        with per_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(fieldnames))
            w.writeheader()
            w.writerows(task_rows)
    agg = results.get("aggregate") or {}
    agg_path = out_dir / "aggregate.csv"
    with agg_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(("metric", "value"))
        for k in sorted(agg.keys()):
            w.writerow((k, agg[k]))


def plot_results(results: Mapping[str, Any], out_dir: Path) -> None:
    """Save a per-task accuracy bar chart (PNG) when matplotlib is available."""

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:  # pragma: no cover - optional benchmark extra
        return
    out_dir = Path(out_dir)
    per_task = results.get("per_task") or {}
    tasks: list[str] = []
    accs: list[float] = []
    for task in sorted(per_task.keys()):
        m = per_task[task]
        if not isinstance(m, Mapping):
            continue
        try:
            accs.append(float(m["accuracy"]))
            tasks.append(task)
        except (KeyError, TypeError, ValueError):
            continue
    if not tasks:
        return
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(max(6.0, 0.35 * len(tasks)), 4.0))
    ax.bar(tasks, accs, color=sns.color_palette("deep", n_colors=1)[0])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("accuracy")
    ax.set_xlabel("task")
    ax.set_title(str(results.get("model_id", "model")))
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_by_task.png", dpi=150)
    try:
        fig.savefig(out_dir / "accuracy_by_task.pdf", format="pdf")
    except Exception:  # pragma: no cover - rare backend/font issues
        pass
    plt.close(fig)


def render_latex_table(results: Mapping[str, Any], out_dir: Path) -> None:
    """Emit a small LaTeX ``tabular`` snippet for the per-task accuracies."""

    out_dir = Path(out_dir)
    per_task = results.get("per_task") or {}
    lines = [
        r"\begin{tabular}{lrr}",
        r"\hline",
        r"Task & $n$ & Accuracy \\",
        r"\hline",
    ]
    for task in sorted(per_task.keys()):
        m = per_task[task]
        if not isinstance(m, Mapping):
            continue
        safe_task = str(task).replace("_", r"\_")
        lines.append(
            f"{safe_task} & {m.get('n', '')} & {float(m.get('accuracy', 0.0)):.4f} \\\\",
        )
    lines.extend([r"\hline", r"\end{tabular}", ""])
    (out_dir / "summary_table.tex").write_text("\n".join(lines), encoding="utf-8")


def generate_artifacts(results: Mapping[str, Any], out_dir: str | Path) -> None:
    """Publication-oriented exports matching the structure of ``summary.json``."""

    export_csv(results, Path(out_dir))
    plot_results(results, Path(out_dir))
    render_latex_table(results, Path(out_dir))


def evaluate_task(
    backend: BenchmarkBackendProtocol | HFLocalCausalLM,
    task_name: str,
    *,
    limit: int | None,
    split: str | None = None,
    streaming: bool = False,
    seed: int = 0,
    shuffle: bool = False,
    normalize: bool = True,
    chat_template: bool = False,
    generation_max_new_tokens: int = 128,
    progress_label: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    examples = load_task_examples(task_name, limit=limit, split=split, streaming=streaming, seed=seed, shuffle=shuffle)
    # Late import keeps the benchmark module usable in environments where the
    # optional event_bus shim is unavailable; publish() is no-op without subs.
    try:
        from core.system.event_bus import get_default_bus

        bus = get_default_bus()
    except ImportError:
        bus = None
    total = len(examples)
    label = progress_label or task_name
    if bus is not None:
        bus.publish("bench.task.start", {"task": task_name, "label": label, "total": total})
    rows: list[dict[str, Any]] = []
    correct = 0
    for i, ex in enumerate(examples, start=1):
        row = evaluate_example(
            backend,
            ex,
            normalize=normalize,
            chat_template=chat_template,
            generation_max_new_tokens=generation_max_new_tokens,
        )
        rows.append(row)
        if row.get("correct"):
            correct += 1
        if bus is not None:
            bus.publish(
                "bench.example",
                {
                    "task": task_name,
                    "label": label,
                    "i": i,
                    "total": total,
                    "correct": bool(row.get("correct")),
                    "running_correct": correct,
                    "running_acc": correct / max(1, i),
                },
            )
    summary = summarize_rows(rows)
    summary.update(
        {
            "task": task_name,
            "dataset": TASK_REGISTRY[task_name].dataset_name,
            "config": TASK_REGISTRY[task_name].config_name,
            "split": split or TASK_REGISTRY[task_name].split,
            "mode": TASK_REGISTRY[task_name].mode,
            "limit": limit,
        }
    )
    return summary, rows


def _run_hf_tasks_to_dir(
    backend: BenchmarkBackendProtocol | HFLocalCausalLM | HFLocalLlamaBrocaShell | HFLocalSubstrateBench,
    *,
    run_root: Path,
    task_out_dir: Path,
    tasks: Sequence[str],
    limit: int | None,
    split: str | None,
    streaming: bool,
    seed: int,
    shuffle: bool,
    normalize: bool,
    chat_template: bool,
    generation_max_new_tokens: int,
    silent: bool = False,
    arm_label: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    task_out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from core.system.event_bus import get_default_bus

        bus = get_default_bus()
    except ImportError:
        bus = None
    per_task: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    for task in tasks:
        start = time.time()
        progress_label = f"{arm_label}:{task}" if arm_label else task
        task_summary, rows = evaluate_task(
            backend,
            task,
            limit=limit,
            split=split,
            streaming=streaming,
            seed=seed,
            shuffle=shuffle,
            normalize=normalize,
            chat_template=chat_template,
            generation_max_new_tokens=generation_max_new_tokens,
            progress_label=progress_label,
        )
        task_summary["seconds"] = time.time() - start
        per_task[task] = task_summary
        task_path = task_out_dir / f"{task}.jsonl"
        with task_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        all_rows.extend(rows)
        rel_out = task_path.relative_to(run_root)
        if not silent:
            print(f"{task:20s} n={task_summary['n']:4d} acc={task_summary['accuracy']:.3f} wrote={rel_out}", flush=True)
        if bus is not None:
            bus.publish(
                "bench.task.complete",
                {
                    "task": task,
                    "arm": arm_label,
                    "label": progress_label,
                    "n": int(task_summary.get("n", 0)),
                    "correct": int(task_summary.get("correct", 0)),
                    "accuracy": float(task_summary.get("accuracy", 0.0)),
                    "seconds": float(task_summary["seconds"]),
                },
            )
    return per_task, all_rows


def _print_leaderboard_comparison_table(
    *,
    tasks: Sequence[str],
    root: Path,
    per_vanilla: dict[str, Any],
    per_shell: dict[str, Any],
    macro_v: float,
    macro_s: float,
    micro_v: float,
    micro_s: float,
    micro_nv: int,
    micro_cv: int,
    micro_cs: int,
    per_mind: dict[str, Any] | None = None,
    macro_m: float | None = None,
    micro_m: float | None = None,
    micro_cm: int | None = None,
) -> None:
    has_mind = (
        per_mind is not None
        and macro_m is not None
        and micro_m is not None
        and micro_cm is not None
    )
    width = 120 if has_mind else 88
    print("\n" + "=" * width, flush=True)
    if has_mind:
        print(
            "LEADERBOARD  vanilla_lm  |  broca_shell (LlamaBrocaHost)  |  broca_mind (BrocaMind + same host)",
            flush=True,
        )
    else:
        print(
            "LEADERBOARD COMPARISON  vanilla_lm (HFLocalCausalLM)  vs  broca_shell (LlamaBrocaHost, same weights)",
            flush=True,
        )
    print("=" * width, flush=True)
    if has_mind:
        hdr = (
            f"{'task':<16} {'n':>5} {'vanilla':>9} {'shell':>9} {'mind':>9} "
            f"{'dS':>8} {'dM':>8}  jsonl paths"
        )
    else:
        hdr = f"{'task':<18} {'n':>5} {'vanilla_acc':>12} {'broca_acc':>12} {'delta':>10}  jsonl_paths"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for t in tasks:
        pv = per_vanilla[t]
        ps = per_shell[t]
        acc_v = float(pv["accuracy"])
        acc_s = float(ps["accuracy"])
        n = int(pv["n"])
        rel_v = Path(t + ".jsonl").as_posix()
        rel_sh = Path("broca_shell") / (t + ".jsonl")
        if has_mind:
            assert per_mind is not None
            pm = per_mind[t]
            acc_mm = float(pm["accuracy"])
            d_s = acc_s - acc_v
            d_mm = acc_mm - acc_v
            rel_mm = Path("broca_mind") / (t + ".jsonl")
            print(
                f"{t:<16} {n:5d} {acc_v:9.3f} {acc_s:9.3f} {acc_mm:9.3f} {d_s:+8.3f} {d_mm:+8.3f}  "
                f"{rel_v} | {rel_sh.as_posix()} | {rel_mm.as_posix()}",
                flush=True,
            )
        else:
            delta = acc_s - acc_v
            print(
                f"{t:<18} {n:5d} {acc_v:12.3f} {acc_s:12.3f} {delta:+10.3f}  {rel_v}  |  {rel_sh.as_posix()}",
                flush=True,
            )
    print("-" * len(hdr), flush=True)
    if has_mind:
        assert macro_m is not None and micro_m is not None and micro_cm is not None
        print(
            f"{'macro_accuracy':<16}      {macro_v:9.3f} {macro_s:9.3f} {macro_m:9.3f} "
            f"{macro_s - macro_v:+8.3f} {macro_m - macro_v:+8.3f}",
            flush=True,
        )
        print(
            f"{'micro_accuracy':<16} {micro_nv:5d} {micro_v:9.3f} {micro_s:9.3f} {micro_m:9.3f} "
            f"{micro_s - micro_v:+8.3f} {micro_m - micro_v:+8.3f}  "
            f"(hits V={micro_cv} S={micro_cs} M={micro_cm} / n={micro_nv})",
            flush=True,
        )
    else:
        print(
            f"{'macro_accuracy':<18}      {macro_v:12.3f} {macro_s:12.3f} {macro_s - macro_v:+10.3f}",
            flush=True,
        )
        print(
            f"{'micro_accuracy':<18}  {micro_nv:5d} {micro_v:12.3f} {micro_s:12.3f} {micro_s - micro_v:+10.3f}  "
            f"(hits {micro_cv} vanilla vs {micro_cs} shell / n={micro_nv})",
            flush=True,
        )
    print("=" * width, flush=True)
    print(f"artifact root: {root}", flush=True)


def run_hf_datasets_benchmark(
    *,
    model_id: str = DEFAULT_LLAMA_MODEL,
    tasks: Sequence[str],
    output_dir: str | Path = "runs/hf_benchmarks",
    limit: int | None = 50,
    split: str | None = None,
    device: str | None = None,
    hf_token: str | bool | None = None,
    streaming: bool = False,
    seed: int = 0,
    shuffle: bool = False,
    normalize: bool = True,
    chat_template: bool = False,
    max_seq_len: int | None = None,
    generation_max_new_tokens: int = 128,
    trust_remote_code: bool = False,
    compare_llama_broca_host_shell: bool | None = None,
) -> dict[str, Any]:
    """Run HF-datasets MC/generation benchmarks.

    ``compare_llama_broca_host_shell``: ``None`` (default) turns on paired runs for
    Llama checkpoints: ``broca_shell`` (host forward, no substrate) and ``broca_mind``
    (full :class:`~core.cognition.substrate.SubstrateController` on the same loaded host). Pass ``False`` to run
    vanilla only; pass ``True`` to force when possible (skip with a message if not Llama).
    """

    run_stamp = _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(output_dir) / f"hf_native_{run_stamp}"
    out.mkdir(parents=True, exist_ok=True)

    backend = HFLocalCausalLM(
        model_id,
        device=device,
        token=hf_token,
        max_seq_len=max_seq_len,
        trust_remote_code=trust_remote_code,
    )
    mtype = str(getattr(getattr(backend.model, "config", None), "model_type", "") or "").lower()

    if compare_llama_broca_host_shell is False:
        do_compare = False
    elif compare_llama_broca_host_shell is True:
        do_compare = mtype == "llama"
        if not do_compare:
            print(
                "\n--- LlamaBrocaHost comparison requested but model_type is not Llama; skipping shell replay ---",
                flush=True,
            )
    else:
        do_compare = mtype == "llama"

    if do_compare:
        print(
            "\n--- Native HF leaderboard (vanilla_lm + broca_shell + broca_mind · same tasks, same checkpoint) ---",
            flush=True,
        )
    else:
        print("\n--- Native HF datasets benchmark · vanilla_lm (HFLocalCausalLM) ---", flush=True)

    try:
        from core.system.event_bus import get_default_bus

        bus = get_default_bus()
    except ImportError:
        bus = None
    if bus is not None:
        bus.publish(
            "bench.phase.start",
            {
                "phase": "native",
                "model_id": model_id,
                "device": str(backend.device),
                "tasks": list(tasks),
                "limit": limit,
                "compare_shell": bool(do_compare),
                "compare_broca_mind": bool(do_compare),
            },
        )
    silent_first = bool(do_compare)
    per_task, _all_rows = _run_hf_tasks_to_dir(
        backend,
        run_root=out,
        task_out_dir=out,
        tasks=tasks,
        limit=limit,
        split=split,
        streaming=streaming,
        seed=seed,
        shuffle=shuffle,
        normalize=normalize,
        chat_template=chat_template,
        generation_max_new_tokens=generation_max_new_tokens,
        silent=silent_first,
        arm_label="vanilla_lm" if do_compare else None,
    )

    macro = sum(float(v["accuracy"]) for v in per_task.values()) / max(1, len(per_task))
    micro_n = sum(int(v["n"]) for v in per_task.values())
    micro_correct = sum(int(v["correct"]) for v in per_task.values())
    micro_acc = micro_correct / max(1, micro_n)
    macro = round(float(macro), 2)
    micro_acc = round(float(micro_acc), 2)
    if not do_compare:
        print(f"\nvanilla_lm  macro_accuracy={macro:.3f} micro_accuracy={micro_acc:.3f}", flush=True)

    result: dict[str, Any] = {
        "kind": "hf_datasets_native_benchmark",
        "created_at_utc": run_stamp,
        "model_id": model_id,
        "device": str(backend.device),
        "tasks": list(tasks),
        "limit_per_task": limit,
        "split_override": split,
        "streaming": streaming,
        "shuffle": shuffle,
        "seed": seed,
        "backend_label": "vanilla_hf_automodel",
        "scoring": {
            "multiple_choice": "length-normalized continuation log-likelihood",
            "generation": "deterministic greedy decode with normalized exact matching",
            "chat_template": bool(chat_template),
            "max_seq_len": backend.max_seq_len,
        },
        "per_task": per_task,
        "aggregate": {
            "macro_accuracy": macro,
            "micro_accuracy": micro_acc,
            "micro_n": micro_n,
            "micro_correct": micro_correct,
        },
        "artifacts": {
            "summary": "summary.json",
            "task_jsonl": [f"{t}.jsonl" for t in tasks],
            "per_task_csv": "per_task.csv",
            "aggregate_csv": "aggregate.csv",
            "accuracy_plot_png": "accuracy_by_task.png",
            "accuracy_plot_pdf": "accuracy_by_task.pdf",
            "latex_table": "summary_table.tex",
        },
    }

    comparison: dict[str, Any] = {}
    if do_compare:
        shell_back = HFLocalLlamaBrocaShell.wrapping_same_lm(backend)
        shell_dir = out / "broca_shell"
        per_shell, _rows_shell = _run_hf_tasks_to_dir(
            shell_back,
            run_root=out,
            task_out_dir=shell_dir,
            tasks=tasks,
            limit=limit,
            split=split,
            streaming=streaming,
            seed=seed,
            shuffle=shuffle,
            normalize=normalize,
            chat_template=chat_template,
            generation_max_new_tokens=generation_max_new_tokens,
            silent=True,
            arm_label="broca_shell",
        )
        macro_s = sum(float(v["accuracy"]) for v in per_shell.values()) / max(1, len(per_shell))
        micro_n_s = sum(int(v["n"]) for v in per_shell.values())
        micro_c_s = sum(int(v["correct"]) for v in per_shell.values())
        micro_acc_s = micro_c_s / max(1, micro_n_s)
        macro_s = round(float(macro_s), 2)
        micro_acc_s = round(float(micro_acc_s), 2)
        comparison = {
            "llama_broca_shell": {
                "device": str(shell_back.device),
                "aggregate": {
                    "macro_accuracy": macro_s,
                    "micro_accuracy": micro_acc_s,
                    "micro_n": micro_n_s,
                    "micro_correct": micro_c_s,
                    "macro_delta_vs_vanilla_lm": round(macro_s - macro, 2),
                    "micro_delta_vs_vanilla_lm": round(micro_acc_s - micro_acc, 2),
                },
                "per_task": per_shell,
                "artifacts_subdir": "broca_shell",
            }
        }
        mind_preload = (shell_back.host, HuggingFaceBrocaTokenizer(shell_back.tokenizer))
        bm = SubstrateController(
            seed=int(seed),
            db_path=default_substrate_sqlite_path(),
            namespace=CHAT_NAMESPACE,
            preload_host_tokenizer=mind_preload,
            llama_model_id=model_id,
            device=str(shell_back.device),
            hf_token=hf_token,
        )
        mind_backend = HFLocalSubstrateBench(bm, backend)
        mind_dir = out / "broca_mind"
        per_mind, _rows_mind = _run_hf_tasks_to_dir(
            mind_backend,
            run_root=out,
            task_out_dir=mind_dir,
            tasks=tasks,
            limit=limit,
            split=split,
            streaming=streaming,
            seed=seed,
            shuffle=shuffle,
            normalize=normalize,
            chat_template=chat_template,
            generation_max_new_tokens=generation_max_new_tokens,
            silent=True,
            arm_label="broca_mind",
        )
        macro_m = sum(float(v["accuracy"]) for v in per_mind.values()) / max(1, len(per_mind))
        micro_n_m = sum(int(v["n"]) for v in per_mind.values())
        micro_c_m = sum(int(v["correct"]) for v in per_mind.values())
        micro_acc_m = micro_c_m / max(1, micro_n_m)
        macro_m = round(float(macro_m), 2)
        micro_acc_m = round(float(micro_acc_m), 2)
        comparison["broca_mind"] = {
            "device": str(shell_back.device),
            "aggregate": {
                "macro_accuracy": macro_m,
                "micro_accuracy": micro_acc_m,
                "micro_n": micro_n_m,
                "micro_correct": micro_c_m,
                "macro_delta_vs_vanilla_lm": round(macro_m - macro, 2),
                "micro_delta_vs_vanilla_lm": round(micro_acc_m - micro_acc, 2),
                "macro_delta_vs_llama_broca_shell": round(macro_m - macro_s, 2),
                "micro_delta_vs_llama_broca_shell": round(micro_acc_m - micro_acc_s, 2),
            },
            "per_task": per_mind,
            "artifacts_subdir": "broca_mind",
        }
        _print_leaderboard_comparison_table(
            tasks=tasks,
            root=out,
            per_vanilla=per_task,
            per_shell=per_shell,
            macro_v=macro,
            macro_s=macro_s,
            micro_v=micro_acc,
            micro_s=micro_acc_s,
            micro_nv=micro_n,
            micro_cv=micro_correct,
            micro_cs=micro_c_s,
            per_mind=per_mind,
            macro_m=macro_m,
            micro_m=micro_acc_m,
            micro_cm=micro_c_m,
        )

    if comparison:
        result["comparison"] = comparison

    (out / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote summary.json -> {out / 'summary.json'}", flush=True)
    generate_artifacts(result, out)
    if bus is not None:
        bus.publish(
            "bench.phase.complete",
            {
                "phase": "native",
                "macro_accuracy": float(macro),
                "micro_accuracy": float(micro_acc),
                "comparison": bool(comparison),
                "summary_path": str(out / "summary.json"),
            },
        )
    return result


def print_hf_datasets_benchmark_help() -> None:
    print("Standalone module for library imports. Run the unified harness via:\n  python -m core.benchmarks\n")


def main(argv: Sequence[str] | None = None) -> None:
    hp = argparse.ArgumentParser(add_help=False)
    hp.add_argument("-h", "--help", action="store_true")
    hpre, trailing = hp.parse_known_args(argv if argv is not None else sys.argv[1:])
    if hpre.help:
        print_hf_datasets_benchmark_help()
        return
    if trailing:
        print("hf_datasets_eval has no tuning flags; use `python -m core.benchmarks`.", file=sys.stderr)
        raise SystemExit(2)




if __name__ == "__main__":
    main()
