"""Hugging Face tokenizer façade so Broca pathways can share one surface with RegexTokenizer."""

from __future__ import annotations

from typing import Any, Iterable, Sequence


def _fallback_pad_id(tok: Any) -> int:
    eos = getattr(tok, "eos_token_id", None)
    pad = getattr(tok, "pad_token_id", None)
    if pad is None and eos is not None:
        return int(eos)
    if pad is None:
        return 0
    return int(pad)


class HuggingFaceBrocaTokenizer:
    """Wrapper around HF PretrainedTokenizer compatible with graft / generation helpers."""

    def __init__(self, pretrained: Any, *, ensure_pad_token: bool = True) -> None:
        self.inner = pretrained
        if ensure_pad_token and getattr(pretrained, "pad_token_id", None) is None:
            eos = getattr(pretrained, "eos_token_id", None)
            if eos is not None:
                pretrained.pad_token = pretrained.eos_token
                pretrained.pad_token_id = int(eos)

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

    def encode_plan_words(self, words: Iterable[str]) -> list[int]:
        """Map a lexical plan (word-ish strings) into model token IDs (sub-word aware)."""

        text = " ".join(str(w).lower() for w in words)
        return self.encode(text, add_special_tokens=False)


