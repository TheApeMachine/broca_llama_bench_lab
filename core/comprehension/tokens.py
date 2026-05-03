"""LexicalTokens — small helpers for breaking utterances into routing tokens.

Replaces the four free helpers (``_word_tokens``, ``_lexical_tokens``,
``_is_question``, plus the regex split that was duplicated across files).
Kept stateless and as classmethods so callers don't have to instantiate.
"""

from __future__ import annotations

from typing import Any, Sequence

from ..host.tokenizer import utterance_words


class LexicalTokens:
    """Stateless wrapper for utterance → routing-token operations."""

    @classmethod
    def words(cls, toks: Sequence[str]) -> list[str]:
        """Filter to alphanumeric tokens (drops pure-punctuation entries)."""

        return [t for t in toks if any(ch.isalnum() for ch in t)]

    @classmethod
    def lexical(cls, value: Any) -> list[str]:
        """Lowercase, replace underscores with spaces, split into word tokens."""

        text = str(value).replace("_", " ").strip().lower()
        return cls.words(utterance_words(text))

    @classmethod
    def is_question(cls, toks: Sequence[str]) -> bool:
        return any(t == "?" for t in toks)
