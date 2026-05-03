"""High-level knowledge gathering API for the Mosaic substrate.

KnowledgeSeeder is the main entry point. It orchestrates Scrapy crawling,
text extraction, triple extraction, and memory storage in a single call.

Two modes:
1. Full crawl (Scrapy): follows links, respects robots.txt, caches responses
2. Quick fetch (Trafilatura direct): single-page fetch without link following

The seeder can run with or without an LLM — without one, it falls back to
heuristic SVO extraction (lower quality but no GPU needed).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse

from .pipelines import (
    ChunkingPipeline,
    ExtractedTriple,
    PageItem,
    SemanticMemoryStorePipeline,
    TextCleaningPipeline,
    TripleExtractionPipeline,
)

logger = logging.getLogger(__name__)

try:
    import trafilatura
    HAS_TRAFILATURA = True
except ImportError:
    HAS_TRAFILATURA = False

try:
    from scrapy.crawler import CrawlerProcess
    HAS_SCRAPY = True
except ImportError:
    HAS_SCRAPY = False


@dataclass
class GatherResult:
    """Summary of a knowledge gathering run."""
    urls_requested: int = 0
    pages_fetched: int = 0
    pages_extracted: int = 0
    chunks_processed: int = 0
    triples_extracted: int = 0
    triples_stored: int = 0
    triples_corroborated: int = 0
    triples_skipped: int = 0
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    def summary_line(self) -> str:
        return (
            f"Gathered {self.triples_stored} triples from {self.pages_extracted} pages "
            f"({self.triples_corroborated} corroborated, {self.triples_skipped} below threshold) "
            f"in {self.duration_seconds:.1f}s"
        )


class KnowledgeSeeder:
    """Orchestrates web crawling and knowledge extraction into semantic memory.

    Usage:
        from core.broca import SymbolicMemory
        memory = SymbolicMemory("runs/knowledge.sqlite", namespace="web")
        seeder = KnowledgeSeeder(memory=memory)
        result = seeder.gather(urls=["https://en.wikipedia.org/wiki/Python_(programming_language)"])
        print(result.summary_line())

    With an LLM extractor (higher quality):
        from core.broca import BrocaMind
        mind = BrocaMind(...)
        seeder = KnowledgeSeeder(
            memory=mind.memory,
            extractor=mind.relation_extractor,
        )
    """

    def __init__(
        self,
        *,
        memory: Any,
        extractor: Any | None = None,
        confidence_threshold: float = 0.6,
        max_depth: int = 2,
        follow_links: bool = True,
        max_pages: int = 100,
        chunk_max_chars: int = 4000,
        chunk_overlap: int = 400,
    ):
        self.memory = memory
        self.extractor = extractor
        self.confidence_threshold = confidence_threshold
        self.max_depth = max_depth
        self.follow_links = follow_links
        self.max_pages = max_pages

        # Initialize pipeline stages
        self.text_cleaner = TextCleaningPipeline(
            favor_precision=True,
            include_tables=True,
            target_language="en",
        )
        self.chunker = ChunkingPipeline(
            max_chars=chunk_max_chars,
            overlap_chars=chunk_overlap,
        )
        self.triple_extractor = TripleExtractionPipeline(
            extractor=extractor,
            use_heuristic=(extractor is None),
            confidence_threshold=confidence_threshold,
        )
        self.memory_store = SemanticMemoryStorePipeline(
            memory=memory,
            confidence_threshold=confidence_threshold,
        )

    def gather(
        self,
        urls: Sequence[str],
        *,
        allowed_domains: Sequence[str] | None = None,
        use_scrapy: bool | None = None,
    ) -> GatherResult:
        """Gather knowledge from the given URLs.

        Args:
            urls: Seed URLs to crawl/fetch.
            allowed_domains: Restrict link following to these domains.
                Derived from URLs if not specified.
            use_scrapy: Force Scrapy crawl (True), direct fetch (False),
                or auto-detect (None: Scrapy if installed and follow_links).

        Returns:
            GatherResult with statistics about the gathering run.
        """
        start = time.time()
        result = GatherResult(urls_requested=len(urls))

        # Decide crawl mode
        if use_scrapy is None:
            use_scrapy = HAS_SCRAPY and self.follow_links and len(urls) > 0
        elif use_scrapy and not HAS_SCRAPY:
            logger.warning("Scrapy requested but not installed; falling back to direct fetch")
            use_scrapy = False

        if use_scrapy:
            result = self._gather_scrapy(urls, allowed_domains=allowed_domains, result=result)
        else:
            result = self._gather_direct(urls, result=result)

        result.duration_seconds = time.time() - start
        logger.info("Knowledge gathering complete: %s", result.summary_line())
        return result

    def gather_text(self, text: str, *, source_url: str = "direct_input") -> GatherResult:
        """Extract and store triples from raw text (no crawling).

        Useful for ingesting text from non-web sources (files, APIs, user input).
        """
        start = time.time()
        result = GatherResult(urls_requested=0, pages_fetched=1)

        item = PageItem(url=source_url, clean_text=text)
        item = self.chunker.process(item)
        result.chunks_processed = len(item.chunks)

        item = self.triple_extractor.process(item)
        result.triples_extracted = len(item.triples)
        result.pages_extracted = 1

        item = self.memory_store.process(item)
        stats = self.memory_store.stats
        result.triples_stored = stats["stored"]
        result.triples_corroborated = stats["corroborated"]
        result.triples_skipped = stats["skipped"]

        result.duration_seconds = time.time() - start
        return result

    def _gather_direct(self, urls: Sequence[str], result: GatherResult) -> GatherResult:
        """Fetch pages directly via Trafilatura (no link following)."""
        if not HAS_TRAFILATURA:
            result.errors.append("trafilatura not installed")
            return result

        for url in urls[:self.max_pages]:
            try:
                # Fetch the page
                downloaded = trafilatura.fetch_url(url)
                if not downloaded:
                    result.errors.append(f"fetch_failed: {url}")
                    continue
                result.pages_fetched += 1

                # Process through pipeline
                item = PageItem(url=url, html=downloaded)
                item = self.text_cleaner.process(item)
                if item.error and not item.clean_text:
                    result.errors.append(f"{item.error}: {url}")
                    continue
                result.pages_extracted += 1

                item = self.chunker.process(item)
                result.chunks_processed += len(item.chunks)

                item = self.triple_extractor.process(item)
                result.triples_extracted += len(item.triples)

                item = self.memory_store.process(item)

            except Exception as exc:
                logger.warning("Direct fetch failed for %s: %s", url, exc)
                result.errors.append(f"exception: {url}: {exc}")

        stats = self.memory_store.stats
        result.triples_stored = stats["stored"]
        result.triples_corroborated = stats["corroborated"]
        result.triples_skipped = stats["skipped"]
        return result

    def _gather_scrapy(
        self,
        urls: Sequence[str],
        *,
        allowed_domains: Sequence[str] | None = None,
        result: GatherResult,
    ) -> GatherResult:
        """Full Scrapy crawl with link following."""
        from .spider import KnowledgeSpider, POLITE_SETTINGS

        # Collect items via a custom pipeline that accumulates them
        collected_items: list[dict] = []

        class CollectorPipeline:
            def process_item(self_, item: dict, spider: Any) -> dict:
                collected_items.append(dict(item))
                return item

        settings = {
            **POLITE_SETTINGS,
            "DEPTH_LIMIT": self.max_depth,
            "CLOSESPIDER_PAGECOUNT": self.max_pages,
            "ITEM_PIPELINES": {
                f"{CollectorPipeline.__module__}.{CollectorPipeline.__qualname__}": 100,
            },
            # Override to use our collector
            "LOG_LEVEL": "WARNING",
        }

        # Scrapy needs the pipeline class to be importable, so we register it differently
        # Use a simpler approach: custom signal handler
        process = CrawlerProcess(settings=settings)

        # Connect item collection via signals
        from scrapy import signals

        items_collected: list[dict] = []

        def item_scraped(item, response, spider):
            items_collected.append(dict(item))

        # Derive allowed domains
        if allowed_domains is None:
            allowed_domains = list({
                urlparse(u).netloc for u in urls if urlparse(u).netloc
            })

        crawler = process.create_crawler(KnowledgeSpider)
        crawler.signals.connect(item_scraped, signal=signals.item_scraped)

        process.crawl(
            crawler,
            start_urls=list(urls),
            allowed_domains=list(allowed_domains),
            depth_limit=self.max_depth,
            follow_links=self.follow_links,
        )

        # This blocks until crawl is complete
        process.start()

        result.pages_fetched = len(items_collected)

        # Process collected items through our pipelines
        for raw_item in items_collected:
            try:
                item = PageItem(
                    url=raw_item.get("url", ""),
                    html=raw_item.get("html", ""),
                    depth=raw_item.get("depth", 0),
                    status=raw_item.get("status", 200),
                )

                item = self.text_cleaner.process(item)
                if item.error and not item.clean_text:
                    continue
                result.pages_extracted += 1

                item = self.chunker.process(item)
                result.chunks_processed += len(item.chunks)

                item = self.triple_extractor.process(item)
                result.triples_extracted += len(item.triples)

                item = self.memory_store.process(item)

            except Exception as exc:
                logger.warning("Pipeline processing failed: %s", exc)
                result.errors.append(f"pipeline_error: {exc}")

        stats = self.memory_store.stats
        result.triples_stored = stats["stored"]
        result.triples_corroborated = stats["corroborated"]
        result.triples_skipped = stats["skipped"]
        return result
