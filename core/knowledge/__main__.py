"""CLI for the knowledge gathering pipeline.

Usage:
    # Gather from specific URLs (direct fetch, no link following)
    python -m core.knowledge --urls https://en.wikipedia.org/wiki/BoolQ https://en.wikipedia.org/wiki/PIQA

    # Crawl with link following (requires scrapy)
    python -m core.knowledge --urls https://example.com --follow --depth 3

    # Seed from a file of URLs (one per line)
    python -m core.knowledge --url-file seeds.txt --follow --depth 2

    # Use specific database/namespace
    python -m core.knowledge --urls https://example.com --db runs/knowledge.sqlite --namespace web

    # Verbose output
    python -m core.knowledge --urls https://example.com -v
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Mosaic knowledge gathering: crawl web pages and extract triples into semantic memory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--urls", nargs="+", default=[],
        help="Seed URLs to crawl/fetch.",
    )
    parser.add_argument(
        "--url-file", type=Path, default=None,
        help="File containing seed URLs (one per line).",
    )
    parser.add_argument(
        "--follow", action="store_true",
        help="Follow links from seed pages (requires scrapy).",
    )
    parser.add_argument(
        "--depth", type=int, default=2,
        help="Max crawl depth when following links (default: 2).",
    )
    parser.add_argument(
        "--max-pages", type=int, default=100,
        help="Maximum number of pages to process (default: 100).",
    )
    parser.add_argument(
        "--db", type=Path, default=None,
        help="SQLite database path (default: runs/broca_substrate.sqlite).",
    )
    parser.add_argument(
        "--namespace", type=str, default="web_knowledge",
        help="Memory namespace for stored triples (default: web_knowledge).",
    )
    parser.add_argument(
        "--confidence-threshold", type=float, default=0.6,
        help="Minimum confidence to store a triple (default: 0.6).",
    )
    parser.add_argument(
        "--allowed-domains", nargs="*", default=None,
        help="Restrict crawling to these domains (default: derived from URLs).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging.",
    )
    parser.add_argument(
        "--json-out", type=str, default="",
        help="Write result summary to this JSON file.",
    )

    args = parser.parse_args(argv)

    # Collect URLs
    urls = list(args.urls)
    if args.url_file and args.url_file.is_file():
        with open(args.url_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    urls.append(line)

    if not urls:
        print("Error: no URLs provided. Use --urls or --url-file.", file=sys.stderr)
        sys.exit(1)

    # Configure logging
    import logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    # Set up memory
    from core.broca import SymbolicMemory
    from core.substrate_runtime import default_substrate_sqlite_path

    db_path = args.db or default_substrate_sqlite_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    memory = SymbolicMemory(db_path, namespace=args.namespace)

    # Create seeder
    from .seeder import KnowledgeSeeder

    seeder = KnowledgeSeeder(
        memory=memory,
        extractor=None,  # Heuristic mode (no LLM needed for CLI)
        confidence_threshold=args.confidence_threshold,
        max_depth=args.depth,
        follow_links=args.follow,
        max_pages=args.max_pages,
    )

    print(f"Gathering knowledge from {len(urls)} seed URL(s)...", flush=True)
    print(f"  Database: {db_path}", flush=True)
    print(f"  Namespace: {args.namespace}", flush=True)
    print(f"  Follow links: {args.follow}", flush=True)
    print(f"  Max depth: {args.depth}", flush=True)
    print(f"  Max pages: {args.max_pages}", flush=True)
    print(f"  Confidence threshold: {args.confidence_threshold}", flush=True)
    print("", flush=True)

    # Run
    result = seeder.gather(
        urls=urls,
        allowed_domains=args.allowed_domains,
        use_scrapy=args.follow,  # Only use Scrapy when following links
    )

    # Print results
    print("", flush=True)
    print("=" * 60, flush=True)
    print("KNOWLEDGE GATHERING COMPLETE", flush=True)
    print("=" * 60, flush=True)
    print(f"  Pages fetched:       {result.pages_fetched}", flush=True)
    print(f"  Pages extracted:     {result.pages_extracted}", flush=True)
    print(f"  Chunks processed:    {result.chunks_processed}", flush=True)
    print(f"  Triples extracted:   {result.triples_extracted}", flush=True)
    print(f"  Triples stored:      {result.triples_stored}", flush=True)
    print(f"  Triples corroborated:{result.triples_corroborated}", flush=True)
    print(f"  Triples skipped:     {result.triples_skipped}", flush=True)
    print(f"  Duration:            {result.duration_seconds:.1f}s", flush=True)
    if result.errors:
        print(f"  Errors:              {len(result.errors)}", flush=True)
        for err in result.errors[:5]:
            print(f"    - {err}", flush=True)
        if len(result.errors) > 5:
            print(f"    ... and {len(result.errors) - 5} more", flush=True)
    print("=" * 60, flush=True)

    # Memory stats
    n_facts = memory.count()
    avg_conf = memory.mean_confidence()
    print(f"\n  Memory now holds {n_facts} facts (avg confidence: {avg_conf:.3f})" if avg_conf else
          f"\n  Memory now holds {n_facts} facts", flush=True)

    # JSON output
    if args.json_out:
        import json
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "urls_requested": result.urls_requested,
            "pages_fetched": result.pages_fetched,
            "pages_extracted": result.pages_extracted,
            "chunks_processed": result.chunks_processed,
            "triples_extracted": result.triples_extracted,
            "triples_stored": result.triples_stored,
            "triples_corroborated": result.triples_corroborated,
            "triples_skipped": result.triples_skipped,
            "duration_seconds": result.duration_seconds,
            "errors": result.errors[:20],
            "memory_facts": n_facts,
        }, indent=2), encoding="utf-8")
        print(f"\n  Wrote summary to {out_path}", flush=True)

    memory.close()


if __name__ == "__main__":
    main()
