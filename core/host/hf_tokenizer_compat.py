"""Hugging Face tokenizer façade so Broca pathways can share one surface with RegexTokenizer."""

from __future__ import annotations

import copy
from typing import Any, Iterable, Sequence

import torch

from .tokenizer import Batch


def _fallback_pad_id(tok: Any) -> int:
    eos = getattr(tok, "eos_token_id", None)
    pad = getattr(tok, "pad_token_id", None)
    if pad is None and eos is not None:
        return int(eos)
    if pad is None:
        return 0
    return int(pad)


class HuggingFaceBrocaTokenizer:
    """Wrapper around HF PretrainedTokenizer compatible with graft / generation helpers.

    ``ensure_pad_token``: when the inner tokenizer has no ``pad_token_id``, alignment
    borrows the EOS id. To avoid mutating a shared ``pretrained`` instance, a
    shallow copy is stored in ``self.inner`` and pad tokens are set only on that copy.
    """

    def __init__(self, pretrained: Any, *, ensure_pad_token: bool = True) -> None:
        inner = copy.copy(pretrained)
        self.inner = inner
        if ensure_pad_token and getattr(inner, "pad_token_id", None) is None:
            eos = getattr(inner, "eos_token_id", None)
            if eos is not None:
                inner.pad_token = inner.eos_token
                inner.pad_token_id = int(eos)

    @property
    def pad_id(self) -> int:
        return _fallback_pad_id(self.inner)

    @property
    def unk_id(self) -> int:
        u = getattr(self.inner, "unk_token_id", None)
        return int(u) if u is not None else self.pad_id

    def encode(self, text: str, *, add_special_tokens: bool = False, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = list(self.inner.encode(text, add_special_tokens=add_special_tokens))
        if add_bos and getattr(self.inner, "bos_token_id", None) is not None:
            bos = int(self.inner.bos_token_id)
            if not ids or ids[0] != bos:
                ids = [bos] + ids
        if add_eos and getattr(self.inner, "eos_token_id", None) is not None:
            eos = int(self.inner.eos_token_id)
            if not ids or ids[-1] != eos:
                ids = ids + [eos]
        return ids

    def decode_id(self, idx: int) -> str:
        return self.inner.decode([int(idx)], skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def decode_tokens(self, ids: Sequence[int]) -> str:
        return self.inner.decode(list(ids), skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def encode_plan_words(self, words: Iterable[str], *, lowercase: bool = False) -> list[int]:
        """Map a lexical plan (word-ish strings) into model token IDs (sub-word aware).

        By default casing is preserved for case-sensitive sub-word tokenizers. Pass
        ``lowercase=True`` to fold plan tokens to lowercase before encoding.
        """

        ws = [str(w) for w in words]
        text = " ".join(w.lower() for w in ws) if lowercase else " ".join(ws)
        return self.encode(text, add_special_tokens=False)

    def primary_token_id(self, word: str) -> int:
        """First sub-token id for ``word`` (used where we need a single CE label / trigger id)."""

        ids = self.encode(str(word).strip(), add_special_tokens=False)
        return int(ids[0]) if ids else self.pad_id

    def batch_encode(self, texts: Sequence[str], *, device: torch.device | str | None = None) -> Batch:
        if not texts:
            z = torch.zeros(0, 0, dtype=torch.long)
            zb = torch.zeros(0, 0, dtype=torch.bool)
            zl = torch.zeros(0, dtype=torch.long)
            if device is not None:
                z = z.to(device)
                zb = zb.to(device)
                zl = zl.to(device)
            return Batch(ids=z, attention_mask=zb, lengths=zl)
        encoded = [self.encode(t, add_special_tokens=False) for t in texts]
        max_len = max(len(row) for row in encoded)
        pad = self.pad_id
        ids = torch.full((len(encoded), max_len), pad, dtype=torch.long)
        mask = torch.zeros((len(encoded), max_len), dtype=torch.bool)
        lengths = torch.tensor([len(row) for row in encoded], dtype=torch.long)
        for i, row in enumerate(encoded):
            if row:
                ids[i, : len(row)] = torch.tensor(row, dtype=torch.long)
                mask[i, : len(row)] = True
        if device is not None:
            ids = ids.to(device)
            mask = mask.to(device)
            lengths = lengths.to(device)
        return Batch(ids=ids, attention_mask=mask, lengths=lengths)
