from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]")

SPEECH_BRIDGE_PREFIX = "speak :"


def speech_seed_ids(tokenizer, prefix: str | None = None) -> list[int]:
    """Token IDs used to seed Broca speech generation.

    The default is a neutral BOS/pad context rather than a semantic magic string.
    Passing ``prefix`` preserves the older explicit cue behavior.
    """

    if prefix is not None:
        return list(tokenizer.encode(prefix))
    enc = getattr(tokenizer, "encode")
    try:
        ids = list(enc("", add_bos=True))
    except TypeError:
        ids = list(enc(""))
    if ids:
        return ids
    pad = getattr(tokenizer, "pad_id", None)
    return [int(pad)] if pad is not None else []


def utterance_words(text: str) -> list[str]:
    """Lowercase word/punct tokens for routing (same rules as ``RegexTokenizer.tokenize``)."""

    return _TOKEN_RE.findall(text.lower())


@dataclass
class Batch:
    ids: torch.Tensor
    attention_mask: torch.Tensor
    lengths: torch.Tensor


class RegexTokenizer:
    """Tiny deterministic tokenizer for the local experiments.

    The point of this lab is not tokenization. This keeps the host model
    inspectable while still letting grafts operate on real token IDs and hidden
    activations.
    """

    PAD = "<pad>"
    UNK = "<unk>"
    BOS = "<bos>"
    EOS = "<eos>"

    def __init__(self, vocab: Sequence[str] | None = None):
        base = [self.PAD, self.UNK, self.BOS, self.EOS]
        if vocab is None:
            vocab = base
        self.vocab = list(dict.fromkeys(vocab))
        for tok in reversed(base):
            if tok not in self.vocab:
                self.vocab.insert(0, tok)
        self.token_to_id = {tok: i for i, tok in enumerate(self.vocab)}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return utterance_words(text)

    @classmethod
    def fit(cls, texts: Iterable[str], extra_tokens: Iterable[str] = ()) -> "RegexTokenizer":
        toks: set[str] = set()
        for text in texts:
            toks.update(cls.tokenize(text))
        toks.update(t.lower() for t in extra_tokens)
        base = [cls.PAD, cls.UNK, cls.BOS, cls.EOS]
        return cls(base + sorted(tok for tok in toks if tok not in base))

    def __len__(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.PAD]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.UNK]

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        toks = self.tokenize(text)
        if add_bos:
            toks = [self.BOS] + toks
        if add_eos:
            toks = toks + [self.EOS]
        return [self.token_to_id.get(tok, self.unk_id) for tok in toks]

    def encode_plan_words(self, words: Iterable[str]) -> list[int]:
        """Token IDs for a planned utterance (one ID per regex token, used by Broca grafts)."""

        return [self.token_to_id.get(str(w).lower(), self.unk_id) for w in words]

    def batch_encode(self, texts: Sequence[str], *, device: torch.device | str | None = None) -> Batch:
        encoded = [self.encode(t) for t in texts]
        max_len = max(1, max(len(row) for row in encoded))
        ids = torch.full((len(encoded), max_len), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(encoded), max_len), dtype=torch.bool)
        lengths = torch.tensor([max(1, len(row)) for row in encoded], dtype=torch.long)
        for i, row in enumerate(encoded):
            if not row:
                continue
            ids[i, : len(row)] = torch.tensor(row, dtype=torch.long)
            mask[i, : len(row)] = True
        if device is not None:
            ids = ids.to(device)
            mask = mask.to(device)
            lengths = lengths.to(device)
        return Batch(ids=ids, attention_mask=mask, lengths=lengths)

    def decode_id(self, idx: int) -> str:
        return self.id_to_token.get(int(idx), self.UNK)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.vocab, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "RegexTokenizer":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))
