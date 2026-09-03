"""
VeritasAI V1 — Web Evidence Evaluator Service
===============================================
Discovers and extracts external web evidence for a given claim string.

Pipeline
--------
1. ``search_claim(claim)``   — Generate queries, call search backend, return
                               a list of raw search result dicts.
2. ``fetch_page(url)``        — Download a single URL, parse with BeautifulSoup,
                               return clean text (truncated to MAX_CONTENT_CHARS).
3. ``evaluate_claim(claim)``  — Orchestrate both steps; return a list of
                               :class:`WebEvidence` objects ready for the next
                               pipeline stage.

Search backends (tried in priority order)
------------------------------------------
* **Serper** — ``https://google.serper.dev/search`` (requires ``SERPER_API_KEY``
  in the environment).  Returns structured JSON; fast and reliable.
* **DuckDuckGo HTML fallback** — Scrapes the DDG HTML results page.  No API key
  needed; rate-limited by DDG so suitable for light / development use only.

Security notes
--------------
* ``robots.txt`` is checked before fetching any non-search page.
* A descriptive ``User-Agent`` header is set on every request.
* Hard connect + read timeouts prevent hanging.
* No CAPTCHA bypass, paywall bypass, or authentication circumvention.
* Content is truncated at ``MAX_CONTENT_CHARS`` to avoid memory exhaustion.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
import urllib.robotparser
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

USER_AGENT: str = (
    "VeritasAI/1.0 (fact-verification research bot; "
    "contact: admin@veritas-ai.example.com)"
)

#: Maximum characters of page body text to keep per page.
MAX_CONTENT_CHARS: int = 8_000

#: HTTP timeouts (seconds).
CONNECT_TIMEOUT: float = 5.0
READ_TIMEOUT: float = 10.0

#: Serper search endpoint.
SERPER_ENDPOINT: str = "https://google.serper.dev/search"

#: DuckDuckGo HTML search endpoint (fallback).
DDG_ENDPOINT: str = "https://html.duckduckgo.com/html/"

#: Maximum search results to request from the backend.
MAX_RESULTS: int = 10

#: Maximum number of pages to actually fetch per claim evaluation.
MAX_PAGES_TO_FETCH: int = 5

# HTML tags whose text content we always discard (nav, ads, boilerplate).
_NOISE_TAGS = {
    "script", "style", "noscript", "header", "footer",
    "nav", "aside", "form", "button", "svg", "iframe",
}

# ---------------------------------------------------------------------------
# V1 source-quality block-list
# ---------------------------------------------------------------------------

#: Domains whose content is not suitable as fact-checking evidence in V1.
#: These are social/UGC platforms where:
#:   - content is user-generated with no editorial review,
#:   - pages are often blocked by robots.txt or require login,
#:   - snippets are short, joke-like, or context-free.
#:
#: Reddit is included as a conservative V1 choice: Reddit pages are typically
#: blocked or return login walls, leaving only thin snippets that make poor
#: NLI evidence.  Remove "reddit.com" from this set if Reddit evidence proves
#: useful in a future version.
#:
#: Extend this set as needed; do NOT add filter logic elsewhere in the codebase.
BLOCKED_DOMAINS: frozenset[str] = frozenset({
    "facebook.com",
    "www.facebook.com",
    "instagram.com",
    "www.instagram.com",
    "tiktok.com",
    "www.tiktok.com",
    "twitter.com",
    "www.twitter.com",
    "x.com",
    "www.x.com",
    "reddit.com",
    "www.reddit.com",
    "old.reddit.com",
    "m.reddit.com",
})


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class WebEvidence:
    """
    A single piece of web evidence retrieved for a claim.

    Attributes
    ----------
    title : str
        Page or article title as reported by the search engine or ``<title>`` tag.
    url : str
        Canonical URL of the source page.
    source_name : str
        Human-readable domain / publication name (e.g. ``"bbc.co.uk"``).
    published_at : Optional[str]
        Publication date string if discoverable from meta tags; ``None`` otherwise.
    content : str
        Extracted readable text, truncated to ``MAX_CONTENT_CHARS``.
    search_query : str
        The query string that surfaced this result.
    fetched_successfully : bool
        ``True`` if the page was fetched and parsed without error.
    error : Optional[str]
        Human-readable error description when ``fetched_successfully`` is ``False``.
    """

    title: str
    url: str
    source_name: str
    content: str
    search_query: str
    fetched_successfully: bool
    published_at: Optional[str] = None
    error: Optional[str] = None

    # Internal — not part of the public contract.
    _content_hash: str = field(default="", repr=False, compare=False)

    def __post_init__(self) -> None:
        self._content_hash = hashlib.md5(self.content.encode()).hexdigest()

    @property
    def snippet(self) -> str:
        """First 300 characters of content, suitable for logging / display."""
        return self.content[:300]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _domain(url: str) -> str:
    """Extract the netloc (e.g. ``'www.bbc.co.uk'``) from a URL."""
    try:
        return urlparse(url).netloc
    except Exception:
        return ""


def _is_blocked_domain(url: str) -> bool:
    """
    Return ``True`` if *url* belongs to a domain in :data:`BLOCKED_DOMAINS`.

    Comparison is against the full netloc (including ``www.`` prefix when
    present) so that both ``facebook.com`` and ``www.facebook.com`` are caught.
    Sub-domains (e.g. ``m.facebook.com``) are also blocked by checking whether
    the netloc ends with any blocked domain suffix.

    Parameters
    ----------
    url : str
        Full URL string.

    Returns
    -------
    bool
        ``True`` → block this URL; ``False`` → allow it through.
    """
    netloc = _domain(url).lower()
    if not netloc:
        return False
    if netloc in BLOCKED_DOMAINS:
        return True
    # Catch sub-domains: e.g. "m.facebook.com" ends with ".facebook.com".
    for blocked in BLOCKED_DOMAINS:
        if netloc.endswith("." + blocked.lstrip("www.")):
            return True
    return False


def _clean_text(soup: BeautifulSoup) -> str:
    """
    Remove noise tags then extract all text from *soup*, collapsing whitespace.

    Returns
    -------
    str
        Cleaned text, at most ``MAX_CONTENT_CHARS`` characters.
    """
    for tag in soup.find_all(_NOISE_TAGS):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)
    # Collapse runs of whitespace.
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text[:MAX_CONTENT_CHARS]


def _extract_published_at(soup: BeautifulSoup) -> Optional[str]:
    """
    Attempt to extract a publication date from common meta tags.

    Tries (in order):
    * ``<meta property="article:published_time">``
    * ``<meta name="date">``
    * ``<time datetime="...">``
    """
    for attr, value in [
        ("property", "article:published_time"),
        ("name", "date"),
        ("name", "pubdate"),
        ("name", "DC.date"),
    ]:
        tag = soup.find("meta", attrs={attr: value})
        if tag and tag.get("content"):
            return str(tag["content"])

    time_tag = soup.find("time", attrs={"datetime": True})
    if time_tag:
        return str(time_tag["datetime"])

    return None


def _robots_allowed(base_url: str, path: str, *, timeout: float = 3.0) -> bool:
    """
    Return ``True`` if ``robots.txt`` permits fetching *path* on *base_url*.

    Fails open (returns ``True``) on any network or parse error so that a
    temporary robots.txt outage does not silently block all evidence gathering.
    A warning is logged in that case.
    """
    robots_url = urljoin(base_url, "/robots.txt")
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(robots_url)
    try:
        rp.read()
        allowed: bool = rp.can_fetch(USER_AGENT, urljoin(base_url, path))
        if not allowed:
            logger.info("robots.txt disallows '%s%s'", base_url, path)
        return allowed
    except Exception as exc:
        logger.warning(
            "Could not read robots.txt for %s (%s) — proceeding cautiously.",
            base_url,
            exc,
        )
        return True


def _build_queries(claim: str) -> list[str]:
    """
    Produce up to three search query variants from a raw claim string.

    Strategy
    --------
    1. **Verbatim** — the claim as-is (most precise).
    2. **Fact-check intent** — prepend ``"fact check"`` to catch debunking articles.
    3. **News intent** — strip filler words, keep content words, append ``"news"``.

    Parameters
    ----------
    claim : str
        Raw user claim.

    Returns
    -------
    list[str]
        1–3 distinct query strings.
    """
    claim = claim.strip()
    queries: list[str] = [claim]

    fc_query = f"fact check {claim}"
    if fc_query not in queries:
        queries.append(fc_query)

    # Lightweight keyword extraction: drop common stopwords.
    _STOPWORDS = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "must", "can",
        "to", "of", "in", "on", "at", "by", "for", "with", "about",
        "against", "between", "into", "through", "during", "near",
        "and", "but", "or", "yet", "so", "that", "this", "these",
        "those", "it", "its", "i", "my", "we", "our", "you", "your",
        "he", "she", "they", "their", "two", "one", "three",
    }
    words = [w for w in re.split(r"\W+", claim.lower()) if w and w not in _STOPWORDS]
    if len(words) >= 3:
        news_query = " ".join(words[:8]) + " news"
        if news_query not in queries:
            queries.append(news_query)

    return queries


# ---------------------------------------------------------------------------
# Search backends
# ---------------------------------------------------------------------------

def _parse_serper_response(data: dict, query: str) -> list[dict]:
    """
    Parse a Serper JSON response into a list of uniform result dicts.

    Each dict has keys: ``title``, ``url``, ``snippet``.
    """
    results: list[dict] = []
    for item in data.get("organic", []):
        url = item.get("link", "")
        if not url:
            continue
        results.append(
            {
                "title": item.get("title", ""),
                "url": url,
                "snippet": item.get("snippet", ""),
                "query": query,
            }
        )
    return results


def _parse_ddg_html(html: str, query: str) -> list[dict]:
    """
    Parse DuckDuckGo HTML results into a list of uniform result dicts.
    """
    soup = BeautifulSoup(html, "lxml")
    results: list[dict] = []
    for result in soup.select("div.result"):
        a_tag = result.select_one("a.result__a")
        snippet_tag = result.select_one("a.result__snippet")
        if not a_tag:
            continue
        href = a_tag.get("href", "")
        # DDG sometimes wraps links in a redirect; extract the real URL.
        m = re.search(r"uddg=([^&]+)", href)
        if m:
            import urllib.parse
            href = urllib.parse.unquote(m.group(1))
        if not href.startswith("http"):
            continue
        results.append(
            {
                "title": a_tag.get_text(strip=True),
                "url": href,
                "snippet": snippet_tag.get_text(strip=True) if snippet_tag else "",
                "query": query,
            }
        )
    return results


# ---------------------------------------------------------------------------
# Main service
# ---------------------------------------------------------------------------

class WebEvaluatorService:
    """
    Discovers and extracts web evidence for a free-text claim.

    Parameters
    ----------
    serper_api_key : str, optional
        API key for Serper (https://serper.dev).  When blank the service
        falls back to DuckDuckGo HTML search (no key required, lower limits).
    max_results : int
        Maximum search results to request per query.
    max_pages : int
        Maximum pages to actually fetch and parse per ``evaluate_claim`` call.
    """

    def __init__(
        self,
        serper_api_key: str = "",
        max_results: int = MAX_RESULTS,
        max_pages: int = MAX_PAGES_TO_FETCH,
    ) -> None:
        self._serper_key = serper_api_key.strip()
        self._max_results = max_results
        self._max_pages = max_pages

        self._http_client: Optional[httpx.AsyncClient] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _client(self) -> httpx.AsyncClient:
        """Return (creating lazily) a shared async HTTP client."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                headers={"User-Agent": USER_AGENT},
                timeout=httpx.Timeout(
                    connect=CONNECT_TIMEOUT,
                    read=READ_TIMEOUT,
                    write=5.0,
                    pool=5.0,
                ),
                follow_redirects=True,
                max_redirects=5,
            )
        return self._http_client

    async def close(self) -> None:
        """Close the underlying HTTP client. Call when the service is torn down."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def search_claim(self, claim: str) -> list[dict]:
        """
        Generate search queries from *claim* and return raw search results.

        Each result dict contains:
        ``title``, ``url``, ``snippet``, ``query``.

        Parameters
        ----------
        claim : str
            Raw user claim string.

        Returns
        -------
        list[dict]
            Deduplicated search results across all generated queries.
            Empty list on failure or empty claim.
        """
        if not claim or not claim.strip():
            logger.warning("search_claim called with empty claim.")
            return []

        queries = _build_queries(claim)
        seen_urls: set[str] = set()
        all_results: list[dict] = []

        for query in queries:
            try:
                raw = await self._execute_search(query)
            except Exception as exc:
                logger.error("Search failed for query %r: %s", query, exc)
                continue

            for item in raw:
                url = item.get("url", "")
                if not url:
                    continue
                if _is_blocked_domain(url):
                    logger.debug(
                        "search_claim: blocked domain filtered — %s", url
                    )
                    continue
                if url not in seen_urls:
                    seen_urls.add(url)
                    all_results.append(item)

        logger.info(
            "search_claim: %d unique results across %d queries.",
            len(all_results),
            len(queries),
        )
        return all_results

    async def fetch_page(self, url: str) -> tuple[str, Optional[str], Optional[str]]:
        """
        Fetch a single URL and return ``(content, published_at, error)``.

        * *content* is the cleaned page text (empty string on failure).
        * *published_at* is an ISO date string or ``None``.
        * *error* is a human-readable error message or ``None`` on success.

        Robots.txt is checked before fetching.  Pages larger than
        ``MAX_CONTENT_CHARS`` are truncated.

        Parameters
        ----------
        url : str
            Full URL to fetch.

        Returns
        -------
        tuple[str, Optional[str], Optional[str]]
            ``(content, published_at, error)``
        """
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return "", None, f"Unsupported URL scheme: {parsed.scheme!r}"

        base = f"{parsed.scheme}://{parsed.netloc}"

        # robots.txt check
        if not _robots_allowed(base, parsed.path or "/"):
            return "", None, "Disallowed by robots.txt"

        client = await self._client()
        try:
            response = await client.get(url)
            response.raise_for_status()
        except httpx.TimeoutException:
            msg = f"Timeout fetching {url}"
            logger.warning(msg)
            return "", None, msg
        except httpx.HTTPStatusError as exc:
            msg = f"HTTP {exc.response.status_code} for {url}"
            logger.warning(msg)
            return "", None, msg
        except httpx.RequestError as exc:
            msg = f"Request error for {url}: {exc}"
            logger.warning(msg)
            return "", None, msg
        except Exception as exc:
            msg = f"Unexpected error fetching {url}: {exc}"
            logger.error(msg)
            return "", None, msg

        content_type = response.headers.get("content-type", "")
        if "text/html" not in content_type and "text/plain" not in content_type:
            return "", None, f"Non-HTML content-type: {content_type!r}"

        try:
            soup = BeautifulSoup(response.text, "lxml")
        except Exception as exc:
            msg = f"Parse error for {url}: {exc}"
            logger.warning(msg)
            return "", None, msg

        content = _clean_text(soup)
        if not content:
            return "", None, "Page parsed but no readable content extracted"

        published_at = _extract_published_at(soup)
        return content, published_at, None

    async def evaluate_claim(self, claim: str) -> list[WebEvidence]:
        """
        Full evidence-discovery pipeline for a single claim.

        Steps:

        1. Generate search queries from *claim*.
        2. Execute searches; collect up to ``max_results`` unique URLs.
        3. Fetch and parse up to ``max_pages`` of those URLs.
        4. Return a list of :class:`WebEvidence` objects (both successful
           and failed fetches are included so callers can log / audit them).

        Parameters
        ----------
        claim : str
            Raw user claim string.

        Returns
        -------
        list[WebEvidence]
            Evidence objects in the order they were processed.
            Never raises; failures are captured inside each object.
        """
        if not claim or not claim.strip():
            logger.warning("evaluate_claim called with empty claim.")
            return []

        search_results = await self.search_claim(claim)
        if not search_results:
            logger.info("No search results found for claim: %r", claim[:80])
            return []

        evidence_list: list[WebEvidence] = []
        fetched = 0

        for item in search_results:
            if fetched >= self._max_pages:
                # Record remaining results as unfetched stubs.
                ev = WebEvidence(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    source_name=_domain(item.get("url", "")),
                    content=item.get("snippet", ""),
                    search_query=item.get("query", ""),
                    fetched_successfully=False,
                    error="Page fetch skipped (max_pages limit reached); snippet only.",
                )
                evidence_list.append(ev)
                continue

            url = item.get("url", "")
            title = item.get("title", "")
            query = item.get("query", "")
            snippet = item.get("snippet", "")

            content, published_at, error = await self.fetch_page(url)

            if error:
                ev = WebEvidence(
                    title=title,
                    url=url,
                    source_name=_domain(url),
                    content=snippet,  # fall back to search snippet
                    search_query=query,
                    fetched_successfully=False,
                    published_at=None,
                    error=error,
                )
            else:
                ev = WebEvidence(
                    title=title,
                    url=url,
                    source_name=_domain(url),
                    content=content,
                    search_query=query,
                    fetched_successfully=True,
                    published_at=published_at,
                    error=None,
                )
                fetched += 1

            evidence_list.append(ev)

        logger.info(
            "evaluate_claim: %d evidence objects (%d fetched, %d from snippet only).",
            len(evidence_list),
            fetched,
            len(evidence_list) - fetched,
        )
        return evidence_list

    # ------------------------------------------------------------------
    # Internal search dispatch
    # ------------------------------------------------------------------

    async def _execute_search(self, query: str) -> list[dict]:
        """
        Dispatch to Serper (if key configured) or DDG HTML fallback.

        Parameters
        ----------
        query : str
            Search query string.

        Returns
        -------
        list[dict]
            Raw result dicts with keys: ``title``, ``url``, ``snippet``, ``query``.
        """
        if self._serper_key:
            return await self._serper_search(query)
        return await self._ddg_search(query)

    async def _serper_search(self, query: str) -> list[dict]:
        """Call the Serper API and return parsed results."""
        client = await self._client()
        payload = {"q": query, "num": self._max_results}
        headers = {
            "X-API-KEY": self._serper_key,
            "Content-Type": "application/json",
        }
        response = await client.post(
            SERPER_ENDPOINT,
            json=payload,
            headers=headers,
        )
        response.raise_for_status()
        data = response.json()
        results = _parse_serper_response(data, query)
        logger.debug("Serper: %d results for %r", len(results), query)
        return results

    async def _ddg_search(self, query: str) -> list[dict]:
        """
        Scrape DuckDuckGo HTML results as a zero-config fallback.

        Rate-limited by DDG — suitable for development / testing only.
        """
        client = await self._client()
        response = await client.post(
            DDG_ENDPOINT,
            data={"q": query, "kl": "us-en"},
            headers={
                "User-Agent": USER_AGENT,
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "text/html",
            },
        )
        response.raise_for_status()
        results = _parse_ddg_html(response.text, query)
        # Respect DDG by not hammering; tiny polite delay.
        time.sleep(0.5)
        logger.debug("DDG: %d results for %r", len(results), query)
        return results[: self._max_results]
