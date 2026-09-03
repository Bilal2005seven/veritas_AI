"""
test_web_evaluator.py
=====================
Unit tests for WebEvaluatorService using mocked HTML / mocked network calls.

Run from the backend/ directory:
    python tests/test_web_evaluator.py

For the optional live integration test (requires network access):
    python tests/test_web_evaluator.py --live
"""

from __future__ import annotations

import asyncio
import sys
import textwrap
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# ---------------------------------------------------------------------------
# Make 'app' importable from the backend/ directory.
# ---------------------------------------------------------------------------
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.web_evaluator import (
    MAX_CONTENT_CHARS,
    WebEvidence,
    WebEvaluatorService,
    _build_queries,
    _clean_text,
    _domain,
    _extract_published_at,
    _parse_ddg_html,
    _parse_serper_response,
    _robots_allowed,
)
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Sample HTML fixtures
# ---------------------------------------------------------------------------

SAMPLE_ARTICLE_HTML = textwrap.dedent("""\
    <!DOCTYPE html>
    <html>
    <head>
      <title>Two cars crash near post office in Damoh</title>
      <meta property="article:published_time" content="2024-03-15T10:30:00Z" />
    </head>
    <body>
      <header>Site navigation</header>
      <nav>Menu | Home | About</nav>
      <article>
        <h1>Two cars crash near post office in Damoh</h1>
        <p>
          Two vehicles collided near the central post office in Damoh district
          on Thursday morning. Police confirmed both drivers were taken to
          hospital with minor injuries. An eyewitness said the accident
          occurred around 8 a.m. due to dense fog.
        </p>
        <p>
          Local authorities have warned motorists to exercise caution during
          fog-prone winter months.
        </p>
      </article>
      <footer>Copyright 2024 Example News</footer>
      <script>alert("noise")</script>
      <style>.hidden { display: none; }</style>
    </body>
    </html>
""")

EMPTY_BODY_HTML = textwrap.dedent("""\
    <!DOCTYPE html>
    <html><head><title>Empty</title></head>
    <body><script>window.onload = function(){}</script></body>
    </html>
""")

SAMPLE_DDG_HTML = textwrap.dedent("""\
    <html><body>
    <div class="result">
      <a class="result__a" href="/?uddg=https%3A%2F%2Fwww.example.com%2Fnews%2Fdamoh-crash">
        Damoh crash report
      </a>
      <a class="result__snippet">Two cars crashed near post office.</a>
    </div>
    <div class="result">
      <a class="result__a" href="/?uddg=https%3A%2F%2Fwww.another.com%2Farticle">
        Another article
      </a>
      <a class="result__snippet">Fog causes pile-up in Madhya Pradesh.</a>
    </div>
    </body></html>
""")

SAMPLE_SERPER_JSON = {
    "organic": [
        {
            "title": "Damoh road accident — Times of India",
            "link": "https://timesofindia.com/city/bhopal/damoh-crash",
            "snippet": "Two cars collided near post office in Damoh.",
        },
        {
            "title": "Fog-related accident in Damoh",
            "link": "https://ndtv.com/india/damoh-fog-crash",
            "snippet": "Police report accident near Damoh post office on Thursday.",
        },
    ]
}


# ---------------------------------------------------------------------------
# Helper to run async tests in a simple synchronous test runner.
# ---------------------------------------------------------------------------
def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ===========================================================================
# Unit tests
# ===========================================================================

class TestDomainHelper(unittest.TestCase):
    def test_valid_url(self):
        assert _domain("https://www.bbc.co.uk/news/article") == "www.bbc.co.uk"

    def test_no_scheme(self):
        # urlparse gracefully handles this; netloc will be empty.
        result = _domain("not-a-url")
        assert isinstance(result, str)

    def test_empty_string(self):
        assert _domain("") == ""


class TestBuildQueries(unittest.TestCase):
    def test_returns_list(self):
        q = _build_queries("Two cars crashed in Damoh")
        assert isinstance(q, list)
        assert len(q) >= 1

    def test_verbatim_first(self):
        claim = "Two cars crashed near the post office in Damoh."
        q = _build_queries(claim)
        assert q[0] == claim

    def test_fact_check_variant(self):
        q = _build_queries("The moon is made of cheese")
        assert any("fact check" in x for x in q)

    def test_news_variant(self):
        q = _build_queries("Yesterday two cars crashed near the post office in Damoh")
        assert any("news" in x for x in q)

    def test_empty_claim(self):
        q = _build_queries("")
        # Empty string still produces at least one entry.
        assert isinstance(q, list) and len(q) >= 1
        assert q[0] == ""  # verbatim is always first

    def test_no_duplicate_queries(self):
        q = _build_queries("A")
        assert len(q) == len(set(q))


class TestCleanText(unittest.TestCase):
    def test_removes_script_and_style(self):
        soup = BeautifulSoup(SAMPLE_ARTICLE_HTML, "lxml")
        text = _clean_text(soup)
        assert "alert" not in text
        assert "hidden" not in text

    def test_removes_nav_footer_header(self):
        soup = BeautifulSoup(SAMPLE_ARTICLE_HTML, "lxml")
        text = _clean_text(soup)
        assert "Site navigation" not in text
        assert "Copyright" not in text
        assert "Menu | Home | About" not in text

    def test_preserves_article_content(self):
        soup = BeautifulSoup(SAMPLE_ARTICLE_HTML, "lxml")
        text = _clean_text(soup)
        assert "post office in Damoh" in text
        assert "minor injuries" in text

    def test_truncation(self):
        long_html = "<html><body><p>" + "x" * (MAX_CONTENT_CHARS + 1000) + "</p></body></html>"
        soup = BeautifulSoup(long_html, "lxml")
        text = _clean_text(soup)
        assert len(text) <= MAX_CONTENT_CHARS

    def test_empty_body(self):
        soup = BeautifulSoup(EMPTY_BODY_HTML, "lxml")
        text = _clean_text(soup)
        assert text.strip() == "" or len(text) < 10  # only noise removed


class TestExtractPublishedAt(unittest.TestCase):
    def test_article_published_time(self):
        soup = BeautifulSoup(SAMPLE_ARTICLE_HTML, "lxml")
        result = _extract_published_at(soup)
        assert result == "2024-03-15T10:30:00Z"

    def test_time_tag_fallback(self):
        html = '<html><body><time datetime="2024-01-10">Jan 10</time></body></html>'
        soup = BeautifulSoup(html, "lxml")
        result = _extract_published_at(soup)
        assert result == "2024-01-10"

    def test_no_date(self):
        soup = BeautifulSoup("<html><body><p>No date here</p></body></html>", "lxml")
        result = _extract_published_at(soup)
        assert result is None


class TestParseSerperResponse(unittest.TestCase):
    def test_extracts_results(self):
        results = _parse_serper_response(SAMPLE_SERPER_JSON, query="damoh crash")
        assert len(results) == 2
        assert results[0]["title"] == "Damoh road accident — Times of India"
        assert results[0]["url"] == "https://timesofindia.com/city/bhopal/damoh-crash"
        assert results[0]["snippet"] == "Two cars collided near post office in Damoh."
        assert results[0]["query"] == "damoh crash"

    def test_skips_items_without_url(self):
        data = {"organic": [{"title": "No link here", "snippet": "..."}]}
        results = _parse_serper_response(data, query="test")
        assert results == []

    def test_empty_organic(self):
        results = _parse_serper_response({}, query="test")
        assert results == []


class TestParseDdgHtml(unittest.TestCase):
    def test_extracts_results(self):
        results = _parse_ddg_html(SAMPLE_DDG_HTML, query="damoh crash")
        assert len(results) == 2
        assert results[0]["url"] == "https://www.example.com/news/damoh-crash"
        assert "Damoh crash" in results[0]["title"]

    def test_skips_non_http_links(self):
        html = '<html><body><div class="result"><a class="result__a" href="/relative">Title</a></div></body></html>'
        results = _parse_ddg_html(html, query="test")
        assert results == []


class TestWebEvidence(unittest.TestCase):
    def test_fields(self):
        ev = WebEvidence(
            title="Test Article",
            url="https://example.com/test",
            source_name="example.com",
            content="Some content here about the claim.",
            search_query="test query",
            fetched_successfully=True,
            published_at="2024-01-01",
        )
        assert ev.title == "Test Article"
        assert ev.fetched_successfully is True
        assert ev.error is None
        assert ev.published_at == "2024-01-01"

    def test_snippet_truncation(self):
        content = "x" * 500
        ev = WebEvidence(
            title="T", url="https://e.com", source_name="e.com",
            content=content, search_query="q", fetched_successfully=True,
        )
        assert len(ev.snippet) == 300

    def test_error_evidence(self):
        ev = WebEvidence(
            title="Failed",
            url="https://fail.com",
            source_name="fail.com",
            content="",
            search_query="q",
            fetched_successfully=False,
            error="HTTP 403",
        )
        assert ev.fetched_successfully is False
        assert ev.error == "HTTP 403"


class TestFetchPage(unittest.TestCase):
    """Tests for WebEvaluatorService.fetch_page using mocked HTTP."""

    def _make_service(self) -> WebEvaluatorService:
        return WebEvaluatorService(serper_api_key="", max_results=5, max_pages=3)

    def test_successful_fetch(self):
        svc = self._make_service()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/html; charset=utf-8"}
        mock_response.text = SAMPLE_ARTICLE_HTML
        mock_response.raise_for_status = MagicMock()

        async def run():
            client_mock = MagicMock()
            client_mock.get = AsyncMock(return_value=mock_response)

            with patch.object(svc, "_client", AsyncMock(return_value=client_mock)), \
                 patch("app.services.web_evaluator._robots_allowed", return_value=True):
                content, published_at, error = await svc.fetch_page(
                    "https://example.com/news/damoh"
                )
            return content, published_at, error

        content, published_at, error = _run(run())

        assert error is None, f"Expected no error, got: {error!r}"
        assert "post office in Damoh" in content
        assert published_at == "2024-03-15T10:30:00Z"
        assert len(content) <= MAX_CONTENT_CHARS

    def test_robots_disallowed(self):
        svc = self._make_service()

        async def run():
            with patch(
                "app.services.web_evaluator._robots_allowed", return_value=False
            ):
                content, published_at, error = await svc.fetch_page(
                    "https://blocked.com/secret"
                )
            return content, published_at, error

        content, published_at, error = _run(run())
        assert content == ""
        assert error is not None
        assert "robots" in error.lower()

    def test_unsupported_scheme(self):
        svc = self._make_service()
        content, published_at, error = _run(
            svc.fetch_page("ftp://example.com/file")
        )
        assert content == ""
        assert "scheme" in error.lower()

    def test_timeout_error(self):
        import httpx

        svc = self._make_service()

        async def run():
            client_mock = MagicMock()
            client_mock.get = AsyncMock(
                side_effect=httpx.TimeoutException("timed out")
            )

            with patch.object(svc, "_client", AsyncMock(return_value=client_mock)), \
                 patch("app.services.web_evaluator._robots_allowed", return_value=True):
                content, published_at, error = await svc.fetch_page(
                    "https://slow.example.com/page"
                )
            return content, published_at, error

        content, published_at, error = _run(run())
        assert content == ""
        assert "timeout" in error.lower()

    def test_http_error(self):
        import httpx

        svc = self._make_service()

        async def run():
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_response.raise_for_status = MagicMock(
                side_effect=httpx.HTTPStatusError(
                    "Not found",
                    request=MagicMock(),
                    response=mock_response,
                )
            )
            mock_response.headers = {}

            client_mock = MagicMock()
            client_mock.get = AsyncMock(return_value=mock_response)

            with patch.object(svc, "_client", AsyncMock(return_value=client_mock)), \
                 patch("app.services.web_evaluator._robots_allowed", return_value=True):
                content, published_at, error = await svc.fetch_page(
                    "https://example.com/not-found"
                )
            return content, published_at, error

        content, published_at, error = _run(run())
        assert content == ""
        assert "404" in error

    def test_non_html_content_type(self):
        svc = self._make_service()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/pdf"}
        mock_response.raise_for_status = MagicMock()

        async def run():
            client_mock = MagicMock()
            client_mock.get = AsyncMock(return_value=mock_response)

            with patch.object(svc, "_client", AsyncMock(return_value=client_mock)), \
                 patch("app.services.web_evaluator._robots_allowed", return_value=True):
                content, published_at, error = await svc.fetch_page(
                    "https://example.com/doc.pdf"
                )
            return content, published_at, error

        content, published_at, error = _run(run())
        assert content == ""
        assert "content-type" in error.lower()


class TestSearchClaim(unittest.TestCase):
    def test_empty_claim_returns_empty(self):
        svc = WebEvaluatorService()

        async def run():
            return await svc.search_claim("")

        result = _run(run())
        assert result == []

    def test_deduplication(self):
        """Same URL appearing in multiple query results is only included once."""
        svc = WebEvaluatorService()

        dup_results = [
            {"title": "A", "url": "https://dup.com/page", "snippet": "s", "query": "q1"},
            {"title": "A", "url": "https://dup.com/page", "snippet": "s", "query": "q2"},
            {"title": "B", "url": "https://unique.com/page", "snippet": "s", "query": "q1"},
        ]

        async def _fake_execute(query):
            return dup_results

        async def run():
            svc._execute_search = _fake_execute  # type: ignore[assignment]
            return await svc.search_claim("any claim")

        results = _run(run())
        urls = [r["url"] for r in results]
        assert len(urls) == len(set(urls)), "Duplicate URLs slipped through"
        assert "https://dup.com/page" in urls
        assert "https://unique.com/page" in urls


class TestEvaluateClaim(unittest.TestCase):
    def test_empty_claim_returns_empty(self):
        svc = WebEvaluatorService()

        async def run():
            return await svc.evaluate_claim("")

        result = _run(run())
        assert result == []

    def test_full_pipeline_mocked(self):
        """
        End-to-end evaluate_claim with mocked search + mocked fetch.
        Verifies WebEvidence objects are returned with correct structure.
        """
        svc = WebEvaluatorService(max_pages=2)

        search_results = [
            {
                "title": "Damoh crash report",
                "url": "https://example.com/damoh-crash",
                "snippet": "Two cars crashed near post office.",
                "query": "two cars crashed Damoh",
            },
            {
                "title": "Fog accident Madhya Pradesh",
                "url": "https://news.example.org/fog-accident",
                "snippet": "Fog causes accident in Damoh.",
                "query": "fact check two cars crashed Damoh",
            },
            {
                "title": "Overflow result",
                "url": "https://overflow.example.com/article",
                "snippet": "Should become snippet-only evidence.",
                "query": "crashed cars news",
            },
        ]

        async def _fake_search(claim):
            return search_results

        async def _fake_fetch(url):
            # Return real parsed content for first two URLs.
            if "damoh" in url:
                return "Damoh collision content.", "2024-03-15", None
            if "fog" in url:
                return "Fog accident content.", None, None
            return "", None, "Simulated error"

        async def run():
            svc.search_claim = _fake_search   # type: ignore[assignment]
            svc.fetch_page = _fake_fetch       # type: ignore[assignment]
            return await svc.evaluate_claim(
                "Two cars crashed near the post office in Damoh."
            )

        results = _run(run())

        assert len(results) == 3, f"Expected 3, got {len(results)}"
        assert all(isinstance(ev, WebEvidence) for ev in results)

        # First two are fetched.
        assert results[0].fetched_successfully is True
        assert results[0].content == "Damoh collision content."
        assert results[0].published_at == "2024-03-15"
        assert results[0].source_name == "example.com"

        assert results[1].fetched_successfully is True
        assert results[1].content == "Fog accident content."

        # Third exceeds max_pages — should be snippet-only.
        assert results[2].fetched_successfully is False
        assert "max_pages" in results[2].error.lower() or "skipped" in results[2].error.lower()
        assert results[2].content == "Should become snippet-only evidence."


# ===========================================================================
# Integration / manual test (opt-in, requires network)
# ===========================================================================

async def _live_integration_test() -> None:
    """
    Fetch a single well-known, stable public page and verify content extraction.
    Run only when --live flag is passed.
    """
    print("\n" + "=" * 60)
    print("LIVE INTEGRATION TEST")
    print("=" * 60)

    svc = WebEvaluatorService(max_pages=1)
    # Wikipedia main page — stable, no paywall, permissive robots.txt.
    test_url = "https://en.wikipedia.org/wiki/Fact-checking"

    print(f"Fetching: {test_url}")
    content, published_at, error = await svc.fetch_page(test_url)

    if error:
        print(f"[FAIL] Error: {error}")
    else:
        print(f"[PASS] Content length : {len(content)} chars")
        print(f"       Published at   : {published_at}")
        print(f"       Snippet        : {content[:200]!r}")
        assert len(content) > 100, "Expected substantial content from Wikipedia"
        assert len(content) <= MAX_CONTENT_CHARS
        print("[PASS] Wikipedia fetch integration test passed.")

    await svc.close()

    print("\n--- Live claim evaluate_claim (DDG fallback, no API key) ---")
    svc2 = WebEvaluatorService(max_pages=2)
    claim = "Two cars crashed near the post office in Damoh"
    print(f"Claim: {claim!r}")
    results = await svc2.evaluate_claim(claim)
    print(f"Evidence objects returned: {len(results)}")
    for i, ev in enumerate(results, 1):
        status = "OK" if ev.fetched_successfully else "SNIPPET"
        print(f"  [{i}] [{status}] {ev.url}")
        print(f"       title  : {ev.title[:70]}")
        print(f"       source : {ev.source_name}")
        print(f"       date   : {ev.published_at}")
        print(f"       chars  : {len(ev.content)}")
        if ev.error:
            print(f"       error  : {ev.error}")
    await svc2.close()
    print("\n[PASS] Live integration test complete.")


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    live = "--live" in sys.argv
    if live:
        # Remove --live so unittest doesn't see it as a test name.
        sys.argv.remove("--live")
        asyncio.get_event_loop().run_until_complete(_live_integration_test())
    else:
        print("Running unit tests (mocked)…")
        print("Tip: pass --live to also run the live network integration test.\n")
        unittest.main(verbosity=2)
