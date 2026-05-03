"""MemoryQueryParser — turn a question utterance into a :class:`ParsedQuery`.

Picks a subject from the utterance against the substrate's known subjects
(falling back to the last token when none match), then ranks the predicates
recorded for that subject by lexical similarity to the utterance plus a
small confidence bonus.
"""

from __future__ import annotations

import logging
from typing import Callable, Sequence

from ..frame import ParsedQuery, TextEncoder
from .text_relevance import TextRelevance
from .tokens import LexicalTokens


logger = logging.getLogger(__name__)


class MemoryQueryParser:
    """Stateless wrapper that resolves a question into ``(subject, predicate)``."""

    @classmethod
    def choose_subject(
        cls, words: Sequence[str], known_subjects: Sequence[str]
    ) -> str | None:
        if not words:
            return None
        known = {s.lower(): s.lower() for s in known_subjects}
        for word in words:
            got = known.get(word.lower())
            if got is not None:
                return got
        if known:
            return None
        return words[-1].lower()

    @classmethod
    def choose_predicate(
        cls,
        utterance: str,
        records: Sequence[tuple[str, str, float, dict]],
        text_encoder: TextEncoder | None,
    ) -> str:
        if not records:
            return ""
        if len(records) == 1:
            return records[0][0]
        query_vec = TextRelevance.vector(utterance, text_encoder)
        scored: list[tuple[float, str]] = []
        for pred, obj, conf, ev in records:
            evidence_text = " ".join(
                str(x)
                for x in (pred, obj, ev.get("predicate_surface", ""), ev.get("parser", ""))
            )
            score = TextRelevance.cosine(
                query_vec, TextRelevance.vector(evidence_text, text_encoder)
            ) + 0.05 * float(conf)
            scored.append((score, pred))
        return max(scored, key=lambda item: item[0])[1]

    @classmethod
    def parse(
        cls,
        toks: Sequence[str],
        *,
        utterance: str,
        known_subjects: Sequence[str],
        records_for_subject: Callable[[str], Sequence[tuple[str, str, float, dict]]],
        text_encoder: TextEncoder | None,
    ) -> ParsedQuery | None:
        """Resolve a question into an existing subject/predicate memory lookup."""

        if not LexicalTokens.is_question(toks):
            return None
        words = LexicalTokens.words(toks)
        if not words:
            logger.debug("MemoryQueryParser.parse: empty words utterance=%r", utterance)
            return None
        subject = cls.choose_subject(words, known_subjects)
        if subject is None or not str(subject).strip():
            logger.debug(
                "MemoryQueryParser.parse: no subject utterance=%r words=%s",
                utterance,
                words,
            )
            return None
        records = list(records_for_subject(subject))
        predicate = cls.choose_predicate(utterance, records, text_encoder)
        if not predicate:
            logger.debug(
                "MemoryQueryParser.parse: no predicate utterance=%r subject=%r n_records=%d",
                utterance,
                subject,
                len(records),
            )
            return None
        return ParsedQuery(
            subject=subject,
            predicate=predicate,
            confidence=1.0,
            evidence={
                "parser": "open_memory_query",
                "source_words": words,
                "predicate_candidates": [r[0] for r in records],
            },
        )
