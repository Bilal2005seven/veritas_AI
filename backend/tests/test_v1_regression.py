"""
test_v1_regression.py
=====================
Focused regression tests for the V1 backend cleanup:

  A. Blocked social domain  -- Facebook/social URLs are excluded before reaching
                               the Evidence Engine.
  B. Low relevance gate     -- Evidence below DEFAULT_MIN_RELEVANCE is not sent
                               to NLI.
  C. NEUTRAL stance         -- NEUTRAL evidence contributes zero support or
                               contradiction in the Credibility Engine.
  D. Genuine contradiction  -- A high-relevance CONTRADICTION item still lands
                               in contradicting_evidence.
  E. Genuine entailment     -- A high-relevance ENTAILMENT item still lands in
                               supporting_evidence.

All ML services are mocked; no network or model downloads are required.

Run from backend/:
    python -m pytest tests/test_v1_regression.py -v
"""

from __future__ import annotations

import asyncio
import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from app.services.web_evaluator import (
    BLOCKED_DOMAINS,
    WebEvaluatorService,
    _is_blocked_domain,
)
from app.services.evidence_engine import (
    DEFAULT_MIN_RELEVANCE,
    EvidenceEngine,
    EvidenceAnalysis,
    EvidenceEngineResult,
)
from app.services.credibility_engine import CredibilityEngine
from app.models.schemas import TransformerResult
from app.models.transformer import (
    LABEL_CONTRADICTION,
    LABEL_ENTAILMENT,
    LABEL_NEUTRAL,
)
from app.services.web_evaluator import WebEvidence
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Small helpers shared across tests
# ---------------------------------------------------------------------------

def _run(coro):
    """Run a coroutine synchronously (test-only helper)."""
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_web_evidence(
    url="https://example.com/article",
    content="Some evidence text about the claim.",
    title="Test Article",
    source_name="example.com",
):
    return WebEvidence(
        title=title,
        url=url,
        source_name=source_name,
        content=content,
        search_query="test query",
        fetched_successfully=True,
    )


def _make_nli_result(
    label,
    entailment=0.1,
    contradiction=0.1,
    neutral=0.8,
):
    return TransformerResult(
        label=label,
        score=max(entailment, contradiction, neutral),
        entailment_score=entailment,
        contradiction_score=contradiction,
        neutral_score=neutral,
    )


def _make_engine(relevance_scores, nli_results=None, min_relevance=None):
    """
    Build an EvidenceEngine with mocked embedder and NLI service.
    Uses DEFAULT_MIN_RELEVANCE when min_relevance is None, so tests
    exercise the actual configured threshold.
    """
    mock_embedder = MagicMock()
    mock_nli = MagicMock()

    claim_vec = np.ones(384, dtype=np.float32) / np.sqrt(384)
    ev_vecs = [np.ones(384, dtype=np.float32) * s for s in relevance_scores]
    mock_embedder.encode_text.side_effect = [claim_vec] + ev_vecs
    mock_embedder.similarity.side_effect = [
        np.array([s], dtype=np.float32) for s in relevance_scores
    ]

    if nli_results is not None:
        mock_nli.analyze.side_effect = nli_results

    threshold = DEFAULT_MIN_RELEVANCE if min_relevance is None else min_relevance

    engine = EvidenceEngine(
        embedding_service=mock_embedder,
        transformer_service=mock_nli,
        min_relevance=threshold,
        max_nli_items=20,
        top_k=0,
    )
    return engine, mock_embedder, mock_nli


def _make_evidence_analysis(
    url="https://example.com/a",
    content="Evidence text.",
    relevance_score=0.80,
    nli_label=LABEL_NEUTRAL,
    entailment_score=0.1,
    contradiction_score=0.1,
    neutral_score=0.8,
):
    return EvidenceAnalysis(
        title="Article",
        url=url,
        source_name=url.split("/")[2],
        content=content,
        relevance_score=relevance_score,
        nli_label=nli_label,
        entailment_score=entailment_score,
        contradiction_score=contradiction_score,
        neutral_score=neutral_score,
    )


def _make_engine_result(items, claim="Test claim."):
    result = EvidenceEngineResult(claim=claim, total_processed=len(items))
    result.analyzed_evidence = list(items)
    for item in items:
        if item.nli_label == LABEL_ENTAILMENT:
            result.supporting_evidence.append(item)
        elif item.nli_label == LABEL_CONTRADICTION:
            result.contradicting_evidence.append(item)
        else:
            result.neutral_evidence.append(item)
    return result


# ===========================================================================
# A. Blocked social domain
# ===========================================================================

class TestBlockedSocialDomain(unittest.TestCase):
    FACEBOOK_URL = "https://www.facebook.com/posts/something-123"
    INSTAGRAM_URL = "https://instagram.com/p/abc123"
    TIKTOK_URL = "https://tiktok.com/@user/video/123"
    TWITTER_URL = "https://twitter.com/user/status/9999"
    X_URL = "https://x.com/user/status/9999"
    REDDIT_URL = "https://www.reddit.com/r/worldnews/comments/abc/post"
    LEGIT_URL = "https://reuters.com/world/earth-orbits-sun"

    def test_facebook_is_blocked(self):
        assert _is_blocked_domain(self.FACEBOOK_URL)

    def test_instagram_is_blocked(self):
        assert _is_blocked_domain(self.INSTAGRAM_URL)

    def test_tiktok_is_blocked(self):
        assert _is_blocked_domain(self.TIKTOK_URL)

    def test_twitter_is_blocked(self):
        assert _is_blocked_domain(self.TWITTER_URL)

    def test_x_com_is_blocked(self):
        assert _is_blocked_domain(self.X_URL)

    def test_reddit_is_blocked(self):
        assert _is_blocked_domain(self.REDDIT_URL)

    def test_reuters_is_not_blocked(self):
        assert not _is_blocked_domain(self.LEGIT_URL)

    def test_empty_url_not_blocked(self):
        assert not _is_blocked_domain("")

    def test_blocked_domains_constant_nonempty(self):
        assert len(BLOCKED_DOMAINS) >= 5

    def test_facebook_excluded_from_search_claim(self):
        svc = WebEvaluatorService()
        fake_results = [
            {"title": "FB joke", "url": self.FACEBOOK_URL, "snippet": "lol", "query": "q"},
            {"title": "Reuters", "url": self.LEGIT_URL, "snippet": "real news", "query": "q"},
        ]

        async def _fake_execute(query):
            return fake_results

        async def run():
            svc._execute_search = _fake_execute
            return await svc.search_claim("Earth revolves around the Sun")

        results = _run(run())
        urls = [r["url"] for r in results]
        assert self.FACEBOOK_URL not in urls, f"Facebook must be filtered; got: {urls}"
        assert self.LEGIT_URL in urls

    def test_all_social_blocked_only_legit_passes(self):
        svc = WebEvaluatorService()
        fake_results = [
            {"title": "FB", "url": self.FACEBOOK_URL, "snippet": "s", "query": "q"},
            {"title": "IG", "url": self.INSTAGRAM_URL, "snippet": "s", "query": "q"},
            {"title": "TT", "url": self.TIKTOK_URL, "snippet": "s", "query": "q"},
            {"title": "TW", "url": self.TWITTER_URL, "snippet": "s", "query": "q"},
            {"title": "X", "url": self.X_URL, "snippet": "s", "query": "q"},
            {"title": "RD", "url": self.REDDIT_URL, "snippet": "s", "query": "q"},
            {"title": "Reuters", "url": self.LEGIT_URL, "snippet": "s", "query": "q"},
        ]

        async def _fake_execute(query):
            return fake_results

        async def run():
            svc._execute_search = _fake_execute
            return await svc.search_claim("Earth revolves around the Sun")

        results = _run(run())
        urls = [r["url"] for r in results]
        assert len(results) == 1, f"Only Reuters should pass; got {len(results)}: {urls}"
        assert urls[0] == self.LEGIT_URL


# ===========================================================================
# B. Low-relevance evidence does not reach NLI
# ===========================================================================

class TestLowRelevanceGate(unittest.TestCase):

    def test_below_threshold_nli_not_called(self):
        low_relevance = 0.05
        assert low_relevance < DEFAULT_MIN_RELEVANCE, (
            f"Test assumes {low_relevance} < DEFAULT_MIN_RELEVANCE ({DEFAULT_MIN_RELEVANCE})"
        )
        engine, _, mock_nli = _make_engine(relevance_scores=[low_relevance])
        ev = _make_web_evidence(content="This is completely unrelated content.")
        engine.analyze("The Earth revolves around the Sun.", [ev])
        mock_nli.analyze.assert_not_called()

    def test_below_threshold_item_labelled_neutral(self):
        low_relevance = 0.05
        engine, _, _ = _make_engine(relevance_scores=[low_relevance])
        ev = _make_web_evidence(content="Unrelated content.")
        result = engine.analyze("The Earth revolves around the Sun.", [ev])
        assert result.analyzed_evidence[0].nli_label == LABEL_NEUTRAL
        assert result.analyzed_evidence[0].nli_error is not None

    def test_at_threshold_nli_is_called(self):
        """Relevance == DEFAULT_MIN_RELEVANCE: threshold check is '<', so it passes."""
        at_threshold = DEFAULT_MIN_RELEVANCE
        engine, _, mock_nli = _make_engine(
            relevance_scores=[at_threshold],
            nli_results=[_make_nli_result(LABEL_NEUTRAL)],
        )
        ev = _make_web_evidence(content="Borderline relevant content.")
        engine.analyze("The Earth revolves around the Sun.", [ev])
        mock_nli.analyze.assert_called_once()

    def test_above_threshold_nli_is_called(self):
        high_relevance = 0.80
        assert high_relevance > DEFAULT_MIN_RELEVANCE
        engine, _, mock_nli = _make_engine(
            relevance_scores=[high_relevance],
            nli_results=[_make_nli_result(LABEL_NEUTRAL)],
        )
        ev = _make_web_evidence(content="Highly relevant evidence about Earth orbit.")
        engine.analyze("The Earth revolves around the Sun.", [ev])
        mock_nli.analyze.assert_called_once()


# ===========================================================================
# C. NEUTRAL contributes zero stance in CredibilityEngine
# ===========================================================================

class TestNeutralZeroStance(unittest.TestCase):

    def test_single_neutral_zero_scores(self):
        engine = CredibilityEngine()
        neutral_item = _make_evidence_analysis(
            relevance_score=0.90,
            nli_label=LABEL_NEUTRAL,
        )
        result = _make_engine_result([neutral_item])
        assessment = engine.assess(result)
        assert assessment.support_score == 0.0
        assert assessment.contradiction_score == 0.0

    def test_all_neutral_zero_scores(self):
        engine = CredibilityEngine()
        items = [
            _make_evidence_analysis(
                url=f"https://source{i}.com/a",
                relevance_score=0.85,
                nli_label=LABEL_NEUTRAL,
            )
            for i in range(5)
        ]
        result = _make_engine_result(items)
        assessment = engine.assess(result)
        assert assessment.support_score == 0.0
        assert assessment.contradiction_score == 0.0

    def test_neutral_does_not_raise_support_alongside_entailment(self):
        """Adding a NEUTRAL item must not raise support_score."""
        entailment_item = _make_evidence_analysis(
            url="https://reuters.com/a",
            relevance_score=0.85,
            nli_label=LABEL_ENTAILMENT,
            entailment_score=0.92,
            contradiction_score=0.04,
            neutral_score=0.04,
        )
        neutral_item = _make_evidence_analysis(
            url="https://example.com/b",
            relevance_score=0.85,
            nli_label=LABEL_NEUTRAL,
        )
        result_with = _make_engine_result([entailment_item, neutral_item])
        result_without = _make_engine_result([entailment_item])
        score_with = CredibilityEngine().assess(result_with).support_score
        score_without = CredibilityEngine().assess(result_without).support_score
        assert score_with <= score_without + 1e-6, (
            f"Adding NEUTRAL raised support_score: {score_with} > {score_without}"
        )


# ===========================================================================
# D. Genuine contradiction still classified correctly
# ===========================================================================

class TestGenuineContradiction(unittest.TestCase):

    def test_high_relevance_contradiction_in_evidence_engine(self):
        high_relevance = 0.85
        assert high_relevance > DEFAULT_MIN_RELEVANCE
        engine, _, _ = _make_engine(
            relevance_scores=[high_relevance],
            nli_results=[_make_nli_result(LABEL_CONTRADICTION, 0.03, 0.94, 0.03)],
        )
        ev = _make_web_evidence(content="The Earth does not move; the Sun revolves around it.")
        result = engine.analyze("The Earth revolves around the Sun.", [ev])
        assert len(result.contradicting_evidence) == 1
        assert len(result.supporting_evidence) == 0
        assert result.contradicting_evidence[0].nli_label == LABEL_CONTRADICTION

    def test_high_relevance_contradiction_nonzero_credibility_score(self):
        contra_item = _make_evidence_analysis(
            url="https://skeptic.com/flat",
            relevance_score=0.85,
            nli_label=LABEL_CONTRADICTION,
            entailment_score=0.03,
            contradiction_score=0.94,
            neutral_score=0.03,
        )
        result = _make_engine_result([contra_item])
        assessment = CredibilityEngine().assess(result)
        assert assessment.contradiction_score > 0.0
        assert assessment.support_score == 0.0


# ===========================================================================
# E. Genuine entailment still classified correctly
# ===========================================================================

class TestGenuineEntailment(unittest.TestCase):

    def test_high_relevance_entailment_in_evidence_engine(self):
        high_relevance = 0.85
        assert high_relevance > DEFAULT_MIN_RELEVANCE
        engine, _, _ = _make_engine(
            relevance_scores=[high_relevance],
            nli_results=[_make_nli_result(LABEL_ENTAILMENT, 0.93, 0.03, 0.04)],
        )
        ev = _make_web_evidence(
            content="Earth orbits the Sun once per year at approximately 29.8 km/s."
        )
        result = engine.analyze("The Earth revolves around the Sun.", [ev])
        assert len(result.supporting_evidence) == 1
        assert len(result.contradicting_evidence) == 0
        assert result.supporting_evidence[0].nli_label == LABEL_ENTAILMENT

    def test_high_relevance_entailment_nonzero_credibility_score(self):
        support_item = _make_evidence_analysis(
            url="https://nasa.gov/orbit",
            relevance_score=0.85,
            nli_label=LABEL_ENTAILMENT,
            entailment_score=0.93,
            contradiction_score=0.03,
            neutral_score=0.04,
        )
        result = _make_engine_result([support_item])
        assessment = CredibilityEngine().assess(result)
        assert assessment.support_score > 0.0
        assert assessment.contradiction_score == 0.0


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    print("Running V1 regression tests (all mocked)...\n")
    unittest.main(verbosity=2)
