"""Fast lexical intent decisions before the extraction encoder wakes up."""

from __future__ import annotations

import re
from typing import Sequence


class LexicalIntentClassifier:
    """High-precision intent recognizer for utterances that do not need GLiNER.

    This classifier only handles forms where surface syntax is already enough:
    greetings, acknowledgements, direct requests, hard commands, and explicit
    questions. Declarative statements intentionally return ``None`` so the
    substrate still uses the extraction encoder when it needs a storable claim.
    """

    _GREETING_FORMS: frozenset[tuple[str, ...]] = frozenset(
        {
            ("hi",),
            ("hello",),
            ("hey",),
            ("good", "morning"),
            ("good", "afternoon"),
            ("good", "evening"),
        }
    )
    _FEEDBACK_FORMS: frozenset[tuple[str, ...]] = frozenset(
        {
            ("ok",),
            ("okay",),
            ("thanks",),
            ("thank", "you"),
            ("got", "it"),
            ("yes",),
            ("no",),
            ("sure",),
        }
    )
    _REQUEST_PREFIXES: frozenset[tuple[str, ...]] = frozenset(
        {
            ("tell", "me"),
            ("please",),
            ("can", "you"),
            ("could", "you"),
            ("would", "you"),
            ("will", "you"),
            ("help", "me"),
            ("write",),
            ("create",),
            ("make",),
            ("give",),
            ("show",),
            ("explain",),
            ("summarize",),
            ("find",),
            ("search",),
            ("look", "up"),
            ("run",),
            ("open",),
            ("close",),
        }
    )
    _COMMAND_PREFIXES: frozenset[tuple[str, ...]] = frozenset(
        {
            ("stop",),
            ("never",),
            ("dont",),
            ("do", "not"),
        }
    )
    _QUESTION_PREFIXES: frozenset[tuple[str, ...]] = frozenset(
        {
            ("who",),
            ("what",),
            ("where",),
            ("when",),
            ("why",),
            ("how",),
            ("does",),
            ("do",),
            ("did",),
            ("is",),
            ("are",),
            ("should",),
        }
    )

    def classify(
        self,
        text: str,
        *,
        labels: Sequence[str],
    ) -> tuple[str, float, dict[str, float]] | None:
        tokens = self._tokens(text)
        if not tokens:
            return None

        label, matched = self._label_and_match_count(text, tokens)
        if label is None or label not in labels:
            return None

        confidence = float(matched) / float(max(1, len(tokens)))
        scores = {str(lab): 0.0 for lab in labels}
        scores[label] = confidence
        return label, confidence, scores

    def _label_and_match_count(self, text: str, tokens: tuple[str, ...]) -> tuple[str | None, int]:
        if tokens in self._GREETING_FORMS:
            return "greeting", len(tokens)

        if tokens in self._FEEDBACK_FORMS:
            return "feedback", len(tokens)

        command_len = self._matched_prefix_len(tokens, self._COMMAND_PREFIXES)
        if command_len > 0:
            return "command", command_len

        request_len = self._matched_prefix_len(tokens, self._REQUEST_PREFIXES)
        if request_len > 0:
            return "request", request_len

        question_len = self._matched_prefix_len(tokens, self._QUESTION_PREFIXES)
        if question_len > 0:
            return "question", question_len

        if text.rstrip().endswith("?"):
            return "question", len(tokens)

        return None, 0

    @staticmethod
    def _matched_prefix_len(tokens: tuple[str, ...], prefixes: frozenset[tuple[str, ...]]) -> int:
        for prefix in prefixes:
            n = len(prefix)
            if len(tokens) >= n and tokens[:n] == prefix:
                return n
        return 0

    @staticmethod
    def _tokens(text: str) -> tuple[str, ...]:
        return tuple(re.findall(r"[A-Za-z0-9_]+", text.lower().replace("'", "")))
