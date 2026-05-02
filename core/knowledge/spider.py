"""Scrapy spider for polite, depth-limited web crawling.

Implements a CrawlSpider with:
- robots.txt obedience
- AutoThrottle for adaptive politeness
- Configurable depth limit and domain restriction
- BFS priority for breadth-first coverage
- HTTP caching for development re-runs
"""

from __future__ import annotations

import logging
from typing import Any, Sequence
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

try:
    import scrapy
    from scrapy.linkextractors import LinkExtractor
    from scrapy.spiders import CrawlSpider, Rule

    HAS_SCRAPY = True
except ImportError:
    HAS_SCRAPY = False

# Polite crawl settings — used as defaults by KnowledgeSeeder
POLITE_SETTINGS: dict[str, Any] = {
    # Respect robots.txt — non-negotiable
    "ROBOTSTXT_OBEY": True,

    # AutoThrottle: adaptive delay based on server response time
    "AUTOTHROTTLE_ENABLED": True,
    "AUTOTHROTTLE_START_DELAY": 2.0,
    "AUTOTHROTTLE_MAX_DELAY": 30.0,
    "AUTOTHROTTLE_TARGET_CONCURRENCY": 1.0,

    # Hard limits as safety net
    "DOWNLOAD_DELAY": 1.0,
    "CONCURRENT_REQUESTS": 4,
    "CONCURRENT_REQUESTS_PER_DOMAIN": 1,
    "CONCURRENT_REQUESTS_PER_IP": 1,

    # User-agent transparency
    "USER_AGENT": "MosaicKnowledgeCrawler/1.0 (cognitive substrate research; +https://huggingface.co/theapemachine/mosaic)",

    # Retry with backoff
    "RETRY_ENABLED": True,
    "RETRY_TIMES": 3,
    "RETRY_HTTP_CODES": [429, 500, 502, 503, 504],

    # HTTP cache for dev (avoids re-fetching during prompt tuning)
    "HTTPCACHE_ENABLED": True,
    "HTTPCACHE_DIR": "runs/scrapy_cache",
    "HTTPCACHE_EXPIRATION_SECS": 86400,  # 24h
    "HTTPCACHE_IGNORE_HTTP_CODES": [301, 302, 429],

    # Crawl behavior
    "DEPTH_LIMIT": 2,
    "DEPTH_PRIORITY": 1,  # BFS (breadth-first)

    # Logging
    "LOG_LEVEL": "WARNING",
    "LOG_FORMAT": "%(asctime)s [%(name)s] %(levelname)s: %(message)s",

    # Disable telemetry
    "TELNETCONSOLE_ENABLED": False,
}


if HAS_SCRAPY:

    class KnowledgeSpider(CrawlSpider):
        """Polite breadth-first crawler that yields raw HTML pages.

        Configured via constructor args from KnowledgeSeeder:
        - start_urls: seed URLs to begin crawling
        - allowed_domains: restrict link following to these domains
        - depth_limit: max link-following depth (overrides settings)
        - deny_extensions: file types to skip (default: media/binary)
        """

        name = "mosaic_knowledge"

        # Default: skip binary/media files
        DENY_EXTENSIONS = [
            "pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx",
            "jpg", "jpeg", "png", "gif", "svg", "webp", "ico",
            "mp3", "mp4", "avi", "mov", "wmv", "flv", "webm",
            "zip", "tar", "gz", "rar", "7z",
            "exe", "dmg", "apk", "deb", "rpm",
            "css", "js", "woff", "woff2", "ttf", "eot",
        ]

        def __init__(
            self,
            start_urls: Sequence[str] | None = None,
            allowed_domains: Sequence[str] | None = None,
            depth_limit: int | None = None,
            deny_extensions: Sequence[str] | None = None,
            follow_links: bool = True,
            *args: Any,
            **kwargs: Any,
        ):
            self.start_urls = list(start_urls or [])

            # Derive allowed_domains from start_urls if not explicit
            if allowed_domains is not None:
                self.allowed_domains = list(allowed_domains)
            else:
                self.allowed_domains = list({
                    urlparse(u).netloc for u in self.start_urls if urlparse(u).netloc
                })

            deny_ext = list(deny_extensions or self.DENY_EXTENSIONS)

            if depth_limit is not None:
                self.custom_settings = {
                    **(self.custom_settings or {}),
                    "DEPTH_LIMIT": int(depth_limit),
                }

            # Build rules dynamically based on follow_links
            if follow_links:
                self._rules = [
                    Rule(
                        LinkExtractor(
                            allow_domains=self.allowed_domains,
                            deny_extensions=deny_ext,
                        ),
                        callback="parse_page",
                        follow=True,
                    ),
                ]
            else:
                self._rules = []

            super().__init__(*args, **kwargs)

            # CrawlSpider requires self.rules to be set before _compile_rules
            self.rules = tuple(self._rules)
            self._compile_rules()

        def start_requests(self):
            """Yield requests for each seed URL."""
            for url in self.start_urls:
                yield scrapy.Request(
                    url,
                    callback=self.parse_page if not self._rules else self.parse,
                    meta={"source": "seed", "depth": 0},
                    dont_filter=True,
                )

        def parse_page(self, response):
            """Extract raw page data for downstream pipeline processing."""
            if not hasattr(response, "text") or not response.text:
                logger.debug("Skipping non-text response: %s", response.url)
                return

            content_type = response.headers.get("Content-Type", b"").decode("utf-8", errors="replace").lower()
            if "text/html" not in content_type and "text/plain" not in content_type:
                logger.debug("Skipping non-HTML content type %s: %s", content_type, response.url)
                return

            yield {
                "url": response.url,
                "html": response.text,
                "status": response.status,
                "depth": response.meta.get("depth", 0),
                "content_type": content_type,
                "source": response.meta.get("source", "follow"),
            }

else:
    # Stub when Scrapy is not installed
    class KnowledgeSpider:  # type: ignore[no-redef]
        """Stub: install scrapy to use the knowledge gathering spider."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "KnowledgeSpider requires scrapy. Install with: "
                "pip install scrapy trafilatura"
            )
