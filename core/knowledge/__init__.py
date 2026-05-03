"""Knowledge gathering pipeline for the Mosaic cognitive substrate.

Integrates Scrapy for polite web crawling with Trafilatura for content extraction
and :class:`core.cognition.encoder_relation_extractor.EncoderRelationExtractor`
for triple extraction. Extracted
(subject, predicate, object) triples are stored in SymbolicMemory
with full provenance (source URL, extraction timestamp, confidence).

Architecture:
    Scrapy Spider -> Trafilatura (HTML->text) -> Chunking -> Triple Extraction -> Memory

Usage:
    # Programmatic
    from core.knowledge import KnowledgeSeeder
    seeder = KnowledgeSeeder(memory=mind.memory)
    seeder.gather(urls=["https://en.wikipedia.org/wiki/Python_(programming_language)"])

    # CLI
    python -m core.knowledge --urls https://example.com --depth 2
"""

from __future__ import annotations

from .seeder import KnowledgeSeeder, GatherResult
from .spider import KnowledgeSpider
from .pipelines import (
    TextCleaningPipeline,
    ChunkingPipeline,
    TripleExtractionPipeline,
    SemanticMemoryStorePipeline,
)

__all__ = [
    "KnowledgeSeeder",
    "GatherResult",
    "KnowledgeSpider",
    "TextCleaningPipeline",
    "ChunkingPipeline",
    "TripleExtractionPipeline",
    "SemanticMemoryStorePipeline",
]
