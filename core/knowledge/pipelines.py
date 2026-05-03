"""Scrapy item pipelines for the knowledge gathering system.

Four-stage pipeline:
1. TextCleaningPipeline: HTML -> clean text via Trafilatura
2. ChunkingPipeline: Split text into LLM-safe chunks (~1500 tokens)
3. TripleExtractionPipeline: Extract (subject, predicate, object) triples
4. SemanticMemoryStorePipeline: Store triples in SymbolicMemory

Each pipeline stage is also usable standalone (without Scrapy) for testing
and for the non-Scrapy path (direct URL fetch via trafilatura.fetch_url).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

logger = logging.getLogger(__name__)

try:
    import trafilatura
    HAS_TRAFILATURA = True
except ImportError:
    HAS_TRAFILATURA = False

_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-z0-9]+")


@dataclass
class ExtractedTriple:
    """A single extracted (subject, predicate, object) triple with provenance."""
    subject: str
    predicate: str
    object: str
    confidence: float = 0.9
    source_url: str = ""
    source_chunk: str = ""
    extraction_method: str = "llm_relation_extractor"


@dataclass
class PageItem:
    """Represents a crawled page moving through the pipeline."""
    url: str
    html: str = ""
    clean_text: str = ""
    title: str = ""
    date: str = ""
    hostname: str = ""
    chunks: list[str] = field(default_factory=list)
    triples: list[ExtractedTriple] = field(default_factory=list)
    depth: int = 0
    status: int = 200
    error: str | None = None


# ---------------------------------------------------------------------------
# Stage 1: HTML -> clean text
# ---------------------------------------------------------------------------

class TextCleaningPipeline:
    """Extract clean text from raw HTML using Trafilatura.

    Trafilatura handles boilerplate removal, navigation stripping, and
    article extraction. We use favor_precision=True for KG construction
    to minimize garbage triples from nav/footer text.
    """

    def __init__(
        self,
        *,
        favor_precision: bool = True,
        include_tables: bool = True,
        include_links: bool = False,
        target_language: str | None = "en",
        min_text_length: int = 100,
    ):
        if not HAS_TRAFILATURA:
            raise ImportError(
                "TextCleaningPipeline requires trafilatura. "
                "Install with: pip install trafilatura"
            )
        self.favor_precision = favor_precision
        self.include_tables = include_tables
        self.include_links = include_links
        self.target_language = target_language
        self.min_text_length = min_text_length

    def process(self, item: PageItem) -> PageItem:
        """Extract clean text from item.html, populate item.clean_text."""
        if not item.html:
            item.error = "empty_html"
            return item

        try:
            result = trafilatura.bare_extraction(
                item.html,
                url=item.url,
                include_tables=self.include_tables,
                include_links=self.include_links,
                include_comments=False,
                include_images=False,
                favor_precision=self.favor_precision,
                target_language=self.target_language,
                with_metadata=True,
            )
        except Exception as exc:
            logger.warning("Trafilatura extraction failed for %s: %s", item.url, exc)
            item.error = f"trafilatura_error: {exc}"
            return item

        if result and result.get("text"):
            item.clean_text = str(result["text"])
            item.title = str(result.get("title") or "")
            item.date = str(result.get("date") or "")
            item.hostname = str(result.get("hostname") or "")
        else:
            # Fallback: try extract() for simpler output
            try:
                text = trafilatura.extract(
                    item.html,
                    url=item.url,
                    include_tables=self.include_tables,
                    favor_precision=self.favor_precision,
                )
                item.clean_text = text or ""
            except Exception:
                item.clean_text = ""

        if len(item.clean_text) < self.min_text_length:
            item.error = f"text_too_short ({len(item.clean_text)} chars)"
            logger.debug("Page too short after extraction: %s (%d chars)", item.url, len(item.clean_text))

        return item

    # Scrapy pipeline interface
    def process_item(self, item: dict, spider: Any) -> dict:
        page = PageItem(url=item.get("url", ""), html=item.get("html", ""),
                        depth=item.get("depth", 0), status=item.get("status", 200))
        page = self.process(page)
        item["clean_text"] = page.clean_text
        item["title"] = page.title
        item["date"] = page.date
        item["hostname"] = page.hostname
        item["error"] = page.error
        if page.error and not page.clean_text:
            from scrapy.exceptions import DropItem
            raise DropItem(f"No extractable text: {page.error} ({page.url})")
        return item


# ---------------------------------------------------------------------------
# Stage 2: Text chunking
# ---------------------------------------------------------------------------

class ChunkingPipeline:
    """Split clean text into overlapping chunks suitable for LLM processing.

    Chunks at sentence boundaries when possible. Each chunk is sized to stay
    well within the LLM's context window while providing enough context for
    meaningful triple extraction.
    """

    def __init__(
        self,
        *,
        max_chars: int = 4000,
        overlap_chars: int = 400,
        min_chunk_chars: int = 50,
    ):
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars
        self.min_chunk_chars = min_chunk_chars

    def chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks at sentence boundaries."""
        if not text or len(text) < self.min_chunk_chars:
            return [text] if text else []

        chunks: list[str] = []
        start = 0

        while start < len(text):
            end = min(start + self.max_chars, len(text))

            # Try to break at a sentence boundary
            if end < len(text):
                # Look for the last sentence-ending punctuation before max_chars
                search_start = start + (self.max_chars * 2 // 3)  # Don't break too early
                last_break = -1
                for match in _SENTENCE_END_RE.finditer(text, search_start, end):
                    last_break = match.end()
                if last_break > start + self.min_chunk_chars:
                    end = last_break

            chunk = text[start:end].strip()
            if len(chunk) >= self.min_chunk_chars:
                chunks.append(chunk)

            # Advance with overlap
            start = end - self.overlap_chars
            if start <= (end - self.max_chars + self.overlap_chars):
                # Prevent infinite loop
                start = end

        return chunks

    def process(self, item: PageItem) -> PageItem:
        """Chunk item.clean_text into item.chunks."""
        item.chunks = self.chunk_text(item.clean_text)
        return item

    # Scrapy pipeline interface
    def process_item(self, item: dict, spider: Any) -> dict:
        text = item.get("clean_text", "")
        item["chunks"] = self.chunk_text(text)
        return item


# ---------------------------------------------------------------------------
# Stage 3: Triple extraction
# ---------------------------------------------------------------------------

class TripleExtractionPipeline:
    """Extract (subject, predicate, object) triples from text chunks.

    Supports two extraction backends:
    - :class:`core.cognition.encoder_relation_extractor.EncoderRelationExtractor`
      (the substrate's GLiNER-backed extractor, the production path)
    - Lightweight regex/heuristic fallback (for crawling without GPU)

    The encoder backend respects the same intent-gating rules the chat path
    does. The heuristic backend is lower quality but runs without a model.
    """

    def __init__(
        self,
        *,
        extractor: Any | None = None,
        use_heuristic: bool = False,
        confidence_threshold: float = 0.6,
        max_triples_per_chunk: int = 20,
    ):
        self.extractor = extractor  # EncoderRelationExtractor or compatible
        self.use_heuristic = use_heuristic or (extractor is None)
        self.confidence_threshold = confidence_threshold
        self.max_triples_per_chunk = max_triples_per_chunk

    def extract_from_chunk(self, chunk: str, *, source_url: str = "") -> list[ExtractedTriple]:
        """Extract triples from a single text chunk."""
        if self.use_heuristic:
            return self._heuristic_extract(chunk, source_url=source_url)
        return self._llm_extract(chunk, source_url=source_url)

    def _llm_extract(self, chunk: str, *, source_url: str = "") -> list[ExtractedTriple]:
        """Run the encoder-backed extractor on each sentence."""
        if self.extractor is None:
            return []

        triples: list[ExtractedTriple] = []
        # Split chunk into sentences for per-sentence extraction
        sentences = _SENTENCE_END_RE.split(chunk)

        for sentence in sentences[:50]:  # Cap per chunk
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue

            words = _WORD_RE.findall(sentence.lower())
            if len(words) < 3:
                continue

            try:
                claim = self.extractor.extract_claim(sentence, words)
                if claim is not None:
                    triples.append(ExtractedTriple(
                        subject=claim.subject,
                        predicate=claim.predicate,
                        object=claim.obj,
                        confidence=float(claim.confidence),
                        source_url=source_url,
                        source_chunk=sentence[:200],
                        extraction_method="llm_relation_extractor",
                    ))
            except Exception as exc:
                logger.debug("LLM extraction failed for sentence: %s", exc)
                continue

            if len(triples) >= self.max_triples_per_chunk:
                break

        return triples

    def _heuristic_extract(self, chunk: str, *, source_url: str = "") -> list[ExtractedTriple]:
        """Lightweight SVO extraction using sentence structure heuristics.

        This is lower quality than LLM extraction (~40-50% F1) but runs
        without a model. Useful for bulk crawling where you'll filter
        low-confidence triples later.
        """
        triples: list[ExtractedTriple] = []
        sentences = _SENTENCE_END_RE.split(chunk)

        # Simple patterns: "X is Y", "X are Y", "X was Y", "X has Y"
        patterns = [
            (r"^(.+?)\s+(?:is|are|was|were)\s+(?:a|an|the)?\s*(.+?)$", "is_a"),
            (r"^(.+?)\s+(?:is|are|was|were)\s+(?:located|based|found)\s+(?:in|at|on)\s+(.+?)$", "located_in"),
            (r"^(.+?)\s+(?:is|are|was|were)\s+(?:made|composed|built)\s+(?:of|from)\s+(.+?)$", "made_of"),
            (r"^(.+?)\s+(?:has|have|had)\s+(.+?)$", "has"),
            (r"^(.+?)\s+(?:contains?|includes?)\s+(.+?)$", "contains"),
            (r"^(.+?)\s+(?:created?|invented?|developed?|founded?)\s+(.+?)$", "created"),
            (r"^(.+?)\s+(?:is|are|was|were)\s+(.+?)$", "is"),
        ]

        for sentence in sentences[:30]:
            sentence = sentence.strip()
            if len(sentence) < 10 or len(sentence) > 300:
                continue

            for pattern, predicate in patterns:
                match = re.match(pattern, sentence, re.IGNORECASE)
                if match:
                    subject = match.group(1).strip().lower()[:100]
                    obj = match.group(2).strip().rstrip(".!?,;:").lower()[:100]

                    # Basic quality filters
                    if len(subject) < 2 or len(obj) < 2:
                        continue
                    if len(subject.split()) > 8 or len(obj.split()) > 12:
                        continue

                    triples.append(ExtractedTriple(
                        subject=subject,
                        predicate=predicate,
                        object=obj,
                        confidence=0.6,
                        source_url=source_url,
                        source_chunk=sentence[:200],
                        extraction_method="heuristic_svo",
                    ))
                    break  # One triple per sentence max for heuristic

            if len(triples) >= self.max_triples_per_chunk:
                break

        return triples

    def process(self, item: PageItem) -> PageItem:
        """Extract triples from all chunks in item."""
        all_triples: list[ExtractedTriple] = []
        for chunk in item.chunks:
            triples = self.extract_from_chunk(chunk, source_url=item.url)
            all_triples.extend(triples)
        item.triples = all_triples
        return item

    # Scrapy pipeline interface
    def process_item(self, item: dict, spider: Any) -> dict:
        chunks = item.get("chunks", [])
        url = item.get("url", "")
        all_triples: list[dict] = []
        for chunk in chunks:
            triples = self.extract_from_chunk(chunk, source_url=url)
            all_triples.extend(
                {"subject": t.subject, "predicate": t.predicate, "object": t.object,
                 "confidence": t.confidence, "source_url": t.source_url}
                for t in triples
            )
        item["triples"] = all_triples
        return item


# ---------------------------------------------------------------------------
# Stage 4: Memory storage
# ---------------------------------------------------------------------------

class SemanticMemoryStorePipeline:
    """Store extracted triples in SymbolicMemory with provenance.

    Deduplicates against existing memory: if the same (subject, predicate)
    pair already exists with the same object, corroborates it (boosting
    confidence). If it conflicts, records as a challenger claim for later
    belief revision.
    """

    def __init__(
        self,
        *,
        memory: Any = None,
        confidence_threshold: float = 0.6,
        deduplicate: bool = True,
    ):
        self.memory = memory  # SymbolicMemory instance
        self.confidence_threshold = confidence_threshold
        self.deduplicate = deduplicate
        self._stored_count = 0
        self._skipped_count = 0
        self._corroborated_count = 0

    @property
    def stats(self) -> dict[str, int]:
        return {
            "stored": self._stored_count,
            "skipped": self._skipped_count,
            "corroborated": self._corroborated_count,
        }

    def store_triple(self, triple: ExtractedTriple) -> str:
        """Store a single triple. Returns status: 'stored', 'corroborated', 'skipped'."""
        if self.memory is None:
            return "skipped"

        if triple.confidence < self.confidence_threshold:
            self._skipped_count += 1
            return "skipped"

        evidence = {
            "source_url": triple.source_url,
            "source_chunk": triple.source_chunk[:500],
            "extraction_method": triple.extraction_method,
            "extraction_timestamp": time.time(),
        }

        if self.deduplicate:
            # Use observe_claim for proper belief revision integration
            result = self.memory.observe_claim(
                triple.subject,
                triple.predicate,
                triple.object,
                confidence=triple.confidence,
                evidence=evidence,
            )
            status = result.get("status", "unknown")
            if status == "accepted":
                self._stored_count += 1
                return "stored"
            elif status == "corroborated":
                self._corroborated_count += 1
                return "corroborated"
            else:
                # Conflict — still recorded as a claim for later revision
                self._stored_count += 1
                return "stored"
        else:
            # Direct upsert without dedup
            self.memory.upsert(
                triple.subject,
                triple.predicate,
                triple.object,
                confidence=triple.confidence,
                evidence=evidence,
            )
            self._stored_count += 1
            return "stored"

    def process(self, item: PageItem) -> PageItem:
        """Store all triples from item into memory."""
        for triple in item.triples:
            self.store_triple(triple)
        return item

    # Scrapy pipeline interface
    def process_item(self, item: dict, spider: Any) -> dict:
        for triple_dict in item.get("triples", []):
            triple = ExtractedTriple(
                subject=triple_dict["subject"],
                predicate=triple_dict["predicate"],
                object=triple_dict["object"],
                confidence=triple_dict.get("confidence", 0.9),
                source_url=triple_dict.get("source_url", ""),
            )
            self.store_triple(triple)
        return item
