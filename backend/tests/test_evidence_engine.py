"""
test_evidence_engine.py
=======================
Unit tests for EvidenceEngine.

All dependencies (EmbeddingService, TransformerService, WebEvidence) are
mocked so tests run instantly with no model downloads or network calls.

Run from backend/:
    python tests/test_evidence_engine.py
"""

from __future__ import annotations

import sys
import os
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from app.services.evidence_engine import (
    DEFAULT_MIN_RELEVANCE,
    DEFAULT_TOP_K,
    EvidenceAnalysis,
    EvidenceEngine,
    EvidenceEngineResult,
)
from app.models.schemas import TransformerResult
from app.models.transformer import (
    LABEL_CONTRADICTION,
    LABEL_ENTAILMENT,
    LABEL_NEUTRAL,
)
from app.services.web_evaluator import WebEvidence


# ---------------------------------------------------------------------------
# Fixtures / factories
# ---------------------------------------------------------------------------

def _make_evidence(
    url: str = "https://example.com/article",
    title: str = "Test Article",
    content: str = "Some relevant evidence text about the claim.",
    source_name: str = "example.com",
    fetched_successfully: bool = True,
    error: str | None = None,
    published_at: str | None = "2024-01-01",
) -> WebEvidence:
    return WebEvidence(
        url=url,
        title=title,
        content=content,
        source_name=source_name,
        search_query="test query",
        fetched_successfully=fetched_successfully,
        error=error,
        published_at=published_at,
    )


def _make_nli_result(
    label: str = LABEL_NEUTRAL,
    entailment: float = 0.1,
    contradiction: float = 0.1,
    neutral: float = 0.8,
) -> TransformerResult:
    return TransformerResult(
        label=label,
        score=max(entailment, contradiction, neutral),
        entailment_score=entailment,
        contradiction_score=contradiction,
        neutral_score=neutral,
    )


def _make_engine(
    relevance_scores: list[float] | None = None,
    nli_results: list[TransformerResult] | None = None,
    min_relevance: float = 0.0,   # 0.0 so NLI always runs in tests
    max_nli_items: int = 20,
    top_k: int = 0,               # no cap by default in tests
) -> tuple[EvidenceEngine, MagicMock, MagicMock]:
    """
    Build an EvidenceEngine with mocked embedder and NLI service.

    relevance_scores: per-item cosine similarity returned by the mock embedder.
    nli_results: per-item NLI TransformerResult returned by the mock NLI.
    """
    mock_embedder = MagicMock()
    mock_nli = MagicMock()

    # Claim embedding — fixed 384-dim unit vector.
    claim_vec = np.ones(384, dtype=np.float32) / np.sqrt(384)
    mock_embedder.encode_text.return_value = claim_vec

    if relevance_scores is not None:
        # Return different vectors for each evidence call.
        # We fake similarity by returning the desired score directly from
        # the similarity() method.
        ev_vectors = [np.ones(384, dtype=np.float32) * s for s in relevance_scores]
        mock_embedder.encode_text.side_effect = [claim_vec] + ev_vectors

        # Make similarity() return the per-item score.
        sim_results = [np.array([s], dtype=np.float32) for s in relevance_scores]
        mock_embedder.similarity.side_effect = sim_results

    if nli_results is not None:
        mock_nli.analyze.side_effect = nli_results

    engine = EvidenceEngine(
        embedding_service=mock_embedder,
        transformer_service=mock_nli,
        min_relevance=min_relevance,
        max_nli_items=max_nli_items,
        top_k=top_k,
    )
    return engine, mock_embedder, mock_nli


# ===========================================================================
# Tests: EvidenceAnalysis schema
# ===========================================================================

class TestEvidenceAnalysisSchema(unittest.TestCase):

    def test_required_fields(self):
        ea = EvidenceAnalysis(
            title="T", url="https://u.com", source_name="u.com",
            content="c", relevance_score=0.5,
            nli_label=LABEL_ENTAILMENT,
            entailment_score=0.9, contradiction_score=0.05, neutral_score=0.05,
        )
        assert ea.title == "T"
        assert ea.nli_label == LABEL_ENTAILMENT
        assert ea.relevance_score == 0.5

    def test_optional_fields_default(self):
        ea = EvidenceAnalysis(
            title="T", url="https://u.com", source_name="u.com",
            content="c", relevance_score=0.0,
            nli_label=LABEL_NEUTRAL,
            entailment_score=0.0, contradiction_score=0.0, neutral_score=1.0,
        )
        assert ea.published_at is None
        assert ea.fetch_error is None
        assert ea.nli_error is None

    def test_snippet_property(self):
        ea = EvidenceAnalysis(
            title="T", url="https://u.com", source_name="u.com",
            content="x" * 500, relevance_score=0.0,
            nli_label=LABEL_NEUTRAL,
            entailment_score=0.0, contradiction_score=0.0, neutral_score=1.0,
        )
        assert len(ea.snippet) == 200

    def test_was_fetched_true(self):
        ea = EvidenceAnalysis(
            title="T", url="u", source_name="s", content="valid",
            relevance_score=0.5, nli_label=LABEL_NEUTRAL,
            entailment_score=0.0, contradiction_score=0.0, neutral_score=1.0,
        )
        assert ea.was_fetched is True

    def test_was_fetched_false_when_error(self):
        ea = EvidenceAnalysis(
            title="T", url="u", source_name="s", content="",
            relevance_score=0.0, nli_label="SKIPPED",
            entailment_score=0.0, contradiction_score=0.0, neutral_score=0.0,
            fetch_error="HTTP 404",
        )
        assert ea.was_fetched is False


# ===========================================================================
# Tests: EvidenceEngineResult schema
# ===========================================================================

class TestEvidenceEngineResultSchema(unittest.TestCase):

    def test_empty_result(self):
        r = EvidenceEngineResult(claim="Test claim.")
        assert r.claim == "Test claim."
        assert r.analyzed_evidence == []
        assert r.supporting_evidence == []
        assert r.contradicting_evidence == []
        assert r.neutral_evidence == []
        assert r.total_processed == 0
        assert r.total_skipped == 0


# ===========================================================================
# Tests: input validation
# ===========================================================================

class TestInputValidation(unittest.TestCase):

    def _engine(self) -> EvidenceEngine:
        e, _, _ = _make_engine()
        return e

    def test_non_string_claim_raises_type_error(self):
        with self.assertRaises(TypeError):
            self._engine().analyze(123, [])  # type: ignore

    def test_empty_claim_raises_value_error(self):
        with self.assertRaises(ValueError):
            self._engine().analyze("", [])

    def test_whitespace_claim_raises_value_error(self):
        with self.assertRaises(ValueError):
            self._engine().analyze("   ", [])


# ===========================================================================
# Tests: empty evidence
# ===========================================================================

class TestEmptyEvidence(unittest.TestCase):

    def test_empty_list_returns_empty_result(self):
        engine, _, _ = _make_engine()
        result = engine.analyze("Test claim.", [])
        assert isinstance(result, EvidenceEngineResult)
        assert result.analyzed_evidence == []
        assert result.total_processed == 0

    def test_evidence_with_empty_content_is_skipped(self):
        engine, mock_emb, _ = _make_engine()
        # encode_text for claim only (evidence skipped)
        mock_emb.encode_text.side_effect = None
        mock_emb.encode_text.return_value = np.ones(384, dtype=np.float32)

        ev = _make_evidence(content="")
        result = engine.analyze("Test claim.", [ev])

        assert result.total_skipped == 1
        assert len(result.analyzed_evidence) == 1
        assert result.analyzed_evidence[0].nli_label == "SKIPPED"
        # Empty content lands in neutral bucket.
        assert len(result.neutral_evidence) == 1


# ===========================================================================
# Tests: deduplication
# ===========================================================================

class TestDeduplication(unittest.TestCase):

    def test_duplicate_urls_deduplicated(self):
        engine, mock_emb, mock_nli = _make_engine(
            relevance_scores=[0.8],
            nli_results=[_make_nli_result(LABEL_NEUTRAL)],
        )

        ev1 = _make_evidence(url="https://same.com/page", content="First copy.")
        ev2 = _make_evidence(url="https://same.com/page", content="Second copy.")

        result = engine.analyze("Test claim.", [ev1, ev2])

        # Only first occurrence kept; second counted as skipped.
        assert result.total_processed == 2
        assert result.total_skipped == 1
        assert len(result.analyzed_evidence) == 1
        assert result.analyzed_evidence[0].url == "https://same.com/page"

    def test_different_urls_both_kept(self):
        engine, mock_emb, mock_nli = _make_engine(
            relevance_scores=[0.8, 0.6],
            nli_results=[
                _make_nli_result(LABEL_NEUTRAL),
                _make_nli_result(LABEL_NEUTRAL),
            ],
        )

        ev1 = _make_evidence(url="https://a.com/1", content="Evidence A.")
        ev2 = _make_evidence(url="https://b.com/2", content="Evidence B.")

        result = engine.analyze("Test claim.", [ev1, ev2])
        assert result.total_skipped == 0
        assert len(result.analyzed_evidence) == 2


# ===========================================================================
# Tests: NLI classification and partitioning
# ===========================================================================

class TestNLIClassification(unittest.TestCase):

    def _run(
        self,
        nli_label: str,
        relevance: float = 0.8,
    ) -> EvidenceEngineResult:
        ent = 0.9 if nli_label == LABEL_ENTAILMENT else 0.05
        con = 0.9 if nli_label == LABEL_CONTRADICTION else 0.05
        neu = 0.9 if nli_label == LABEL_NEUTRAL else 0.05

        engine, mock_emb, mock_nli = _make_engine(
            relevance_scores=[relevance],
            nli_results=[_make_nli_result(nli_label, ent, con, neu)],
        )
        ev = _make_evidence(content="Evidence text about the claim.")
        return engine.analyze("Test claim.", [ev])

    def test_entailment_goes_to_supporting(self):
        result = self._run(LABEL_ENTAILMENT)
        assert len(result.supporting_evidence) == 1
        assert result.supporting_evidence[0].nli_label == LABEL_ENTAILMENT
        assert len(result.contradicting_evidence) == 0
        assert len(result.neutral_evidence) == 0

    def test_contradiction_goes_to_contradicting(self):
        result = self._run(LABEL_CONTRADICTION)
        assert len(result.contradicting_evidence) == 1
        assert result.contradicting_evidence[0].nli_label == LABEL_CONTRADICTION
        assert len(result.supporting_evidence) == 0

    def test_neutral_goes_to_neutral(self):
        result = self._run(LABEL_NEUTRAL)
        assert len(result.neutral_evidence) == 1
        assert result.neutral_evidence[0].nli_label == LABEL_NEUTRAL
        assert len(result.supporting_evidence) == 0
        assert len(result.contradicting_evidence) == 0

    def test_neutral_does_not_become_support(self):
        """Core requirement: NEUTRAL must never be treated as SUPPORT."""
        result = self._run(LABEL_NEUTRAL, relevance=0.95)
        # Even with 0.95 relevance, NEUTRAL stays in neutral bucket.
        assert len(result.supporting_evidence) == 0
        assert len(result.neutral_evidence) == 1

    def test_scores_preserved(self):
        nli = _make_nli_result(
            LABEL_ENTAILMENT,
            entailment=0.88,
            contradiction=0.07,
            neutral=0.05,
        )
        engine, mock_emb, mock_nli = _make_engine(
            relevance_scores=[0.75],
            nli_results=[nli],
        )
        ev = _make_evidence(content="Evidence about the claim.")
        result = engine.analyze("Claim.", [ev])

        a = result.analyzed_evidence[0]
        assert abs(a.entailment_score - 0.88) < 1e-4
        assert abs(a.contradiction_score - 0.07) < 1e-4
        assert abs(a.neutral_score - 0.05) < 1e-4
        assert abs(a.relevance_score - 0.75) < 1e-4


# ===========================================================================
# Tests: evidence ranking
# ===========================================================================

class TestEvidenceRanking(unittest.TestCase):

    def test_ranked_by_relevance_descending(self):
        """Items must be sorted highest relevance first."""
        engine, mock_emb, mock_nli = _make_engine(
            relevance_scores=[0.3, 0.9, 0.6],
            nli_results=[
                _make_nli_result(LABEL_NEUTRAL),
                _make_nli_result(LABEL_NEUTRAL),
                _make_nli_result(LABEL_NEUTRAL),
            ],
        )
        evs = [
            _make_evidence(url=f"https://e{i}.com", content=f"Evidence {i}.")
            for i in range(3)
        ]
        result = engine.analyze("Test claim.", evs)

        scores = [a.relevance_score for a in result.analyzed_evidence]
        assert scores == sorted(scores, reverse=True), (
            f"Expected descending order, got {scores}"
        )

    def test_top_k_limits_output(self):
        """top_k must cap the number of items in analyzed_evidence."""
        engine, mock_emb, mock_nli = _make_engine(
            relevance_scores=[0.9, 0.8, 0.7, 0.6, 0.5],
            nli_results=[_make_nli_result(LABEL_NEUTRAL)] * 5,
            top_k=3,
        )
        evs = [
            _make_evidence(url=f"https://e{i}.com", content=f"Evidence {i}.")
            for i in range(5)
        ]
        result = engine.analyze("Test claim.", evs)
        assert len(result.analyzed_evidence) <= 3

    def test_higher_relevance_ranked_first(self):
        engine, mock_emb, mock_nli = _make_engine(
            relevance_scores=[0.2, 0.95],
            nli_results=[
                _make_nli_result(LABEL_NEUTRAL),
                _make_nli_result(LABEL_ENTAILMENT, 0.9, 0.05, 0.05),
            ],
        )
        ev_low = _make_evidence(url="https://low.com", content="Low relevance text.")
        ev_high = _make_evidence(url="https://high.com", content="High relevance text.")
        result = engine.analyze("Test claim.", [ev_low, ev_high])

        assert result.analyzed_evidence[0].url == "https://high.com"
        assert result.analyzed_evidence[1].url == "https://low.com"


# ===========================================================================
# Tests: failed web fetches
# ===========================================================================

class TestFailedFetches(unittest.TestCase):

    def test_failed_fetch_with_snippet_still_processed(self):
        """
        Evidence from a failed fetch but with a non-empty snippet (content)
        should still be embedded and NLI'd.
        """
        engine, mock_emb, mock_nli = _make_engine(
            relevance_scores=[0.6],
            nli_results=[_make_nli_result(LABEL_NEUTRAL)],
        )
        ev = _make_evidence(
            content="Search snippet about the claim.",
            fetched_successfully=False,
            error="HTTP 403",
        )
        result = engine.analyze("Test claim.", [ev])

        assert len(result.analyzed_evidence) == 1
        assert result.analyzed_evidence[0].fetch_error == "HTTP 403"
        # Content from snippet is still processed.
        assert result.analyzed_evidence[0].nli_label == LABEL_NEUTRAL

    def test_failed_fetch_empty_content_skipped(self):
        """Failed fetch with no content → skipped, goes to neutral bucket."""
        engine, mock_emb, _ = _make_engine()
        mock_emb.encode_text.return_value = np.ones(384, dtype=np.float32)

        ev = _make_evidence(
            content="",
            fetched_successfully=False,
            error="Timeout",
        )
        result = engine.analyze("Test claim.", [ev])

        assert result.total_skipped == 1
        assert result.analyzed_evidence[0].nli_label == "SKIPPED"
        assert len(result.neutral_evidence) == 1


# ===========================================================================
# Tests: NLI error handling
# ===========================================================================

class TestNLIErrorHandling(unittest.TestCase):

    def test_nli_exception_captured_gracefully(self):
        """If NLI raises, item is returned as NEUTRAL with nli_error set."""
        mock_emb = MagicMock()
        mock_nli = MagicMock()

        claim_vec = np.ones(384, dtype=np.float32)
        ev_vec = np.ones(384, dtype=np.float32) * 0.7
        mock_emb.encode_text.side_effect = [claim_vec, ev_vec]
        mock_emb.similarity.return_value = np.array([0.7], dtype=np.float32)
        mock_nli.analyze.side_effect = RuntimeError("Model not loaded")

        engine = EvidenceEngine(
            embedding_service=mock_emb,
            transformer_service=mock_nli,
            min_relevance=0.0,
        )
        ev = _make_evidence(content="Some evidence content.")
        result = engine.analyze("Test claim.", [ev])

        assert len(result.analyzed_evidence) == 1
        a = result.analyzed_evidence[0]
        assert a.nli_label == LABEL_NEUTRAL
        assert "Model not loaded" in (a.nli_error or "")
        # Goes to neutral bucket.
        assert len(result.neutral_evidence) == 1

    def test_embedding_error_sets_relevance_zero(self):
        """If embedding raises for an evidence item, relevance → 0."""
        mock_emb = MagicMock()
        mock_nli = MagicMock()

        claim_vec = np.ones(384, dtype=np.float32)
        # First call: claim embedding succeeds.
        # Second call: evidence embedding fails.
        mock_emb.encode_text.side_effect = [claim_vec, RuntimeError("OOM")]
        mock_nli.analyze.return_value = _make_nli_result(LABEL_NEUTRAL)

        engine = EvidenceEngine(
            embedding_service=mock_emb,
            transformer_service=mock_nli,
            min_relevance=0.0,
        )
        ev = _make_evidence(content="Evidence content here.")
        result = engine.analyze("Test claim.", [ev])

        assert result.analyzed_evidence[0].relevance_score == 0.0


# ===========================================================================
# Tests: below-threshold items
# ===========================================================================

class TestRelevanceThreshold(unittest.TestCase):

    def test_below_threshold_skips_nli(self):
        """Items below min_relevance should have NLI skipped."""
        mock_emb = MagicMock()
        mock_nli = MagicMock()

        claim_vec = np.ones(384, dtype=np.float32)
        ev_vec = np.ones(384, dtype=np.float32) * 0.01
        mock_emb.encode_text.side_effect = [claim_vec, ev_vec]
        mock_emb.similarity.return_value = np.array([0.01], dtype=np.float32)

        engine = EvidenceEngine(
            embedding_service=mock_emb,
            transformer_service=mock_nli,
            min_relevance=0.5,   # High threshold
        )
        ev = _make_evidence(content="Barely related evidence.")
        result = engine.analyze("Test claim.", [ev])

        # NLI should NOT have been called.
        mock_nli.analyze.assert_not_called()
        # Item kept as NEUTRAL with nli_error explaining why.
        assert result.analyzed_evidence[0].nli_label == LABEL_NEUTRAL
        assert result.analyzed_evidence[0].nli_error is not None


# ===========================================================================
# Tests: max_nli_items cap
# ===========================================================================

class TestMaxNLIItems(unittest.TestCase):

    def test_items_beyond_max_nli_not_sent_to_nli(self):
        """Items beyond max_nli_items should not invoke NLI."""
        n = 5
        mock_emb = MagicMock()
        mock_nli = MagicMock()

        claim_vec = np.ones(384, dtype=np.float32)
        ev_vecs = [np.ones(384, dtype=np.float32) * (0.9 - i * 0.1) for i in range(n)]
        mock_emb.encode_text.side_effect = [claim_vec] + ev_vecs
        mock_emb.similarity.side_effect = [
            np.array([0.9 - i * 0.1], dtype=np.float32) for i in range(n)
        ]
        mock_nli.analyze.return_value = _make_nli_result(LABEL_NEUTRAL)

        engine = EvidenceEngine(
            embedding_service=mock_emb,
            transformer_service=mock_nli,
            min_relevance=0.0,
            max_nli_items=3,   # Only top-3 go through NLI
            top_k=0,
        )
        evs = [
            _make_evidence(url=f"https://e{i}.com", content=f"Evidence {i}.")
            for i in range(n)
        ]
        result = engine.analyze("Test claim.", evs)

        # NLI called at most 3 times.
        assert mock_nli.analyze.call_count <= 3
        # All 5 items still appear in the result (2 as SKIPPED).
        assert len(result.analyzed_evidence) == 5


# ===========================================================================
# Tests: mixed evidence scenarios
# ===========================================================================

class TestMixedEvidence(unittest.TestCase):

    def test_multiple_labels_partition_correctly(self):
        """Three items with different NLI labels partition into correct buckets."""
        engine, mock_emb, mock_nli = _make_engine(
            relevance_scores=[0.9, 0.8, 0.7],
            nli_results=[
                _make_nli_result(LABEL_ENTAILMENT, 0.9, 0.05, 0.05),
                _make_nli_result(LABEL_CONTRADICTION, 0.05, 0.9, 0.05),
                _make_nli_result(LABEL_NEUTRAL, 0.05, 0.05, 0.9),
            ],
        )
        evs = [
            _make_evidence(url=f"https://e{i}.com", content=f"Evidence {i}.")
            for i in range(3)
        ]
        result = engine.analyze("Test claim.", evs)

        assert len(result.supporting_evidence) == 1
        assert len(result.contradicting_evidence) == 1
        assert len(result.neutral_evidence) == 1
        assert result.supporting_evidence[0].nli_label == LABEL_ENTAILMENT
        assert result.contradicting_evidence[0].nli_label == LABEL_CONTRADICTION
        assert result.neutral_evidence[0].nli_label == LABEL_NEUTRAL

    def test_all_metadata_preserved(self):
        """All WebEvidence fields flow through to EvidenceAnalysis."""
        engine, mock_emb, mock_nli = _make_engine(
            relevance_scores=[0.75],
            nli_results=[_make_nli_result(LABEL_NEUTRAL)],
        )
        ev = _make_evidence(
            url="https://test.com/article",
            title="Test Article Title",
            source_name="test.com",
            published_at="2024-06-15",
            content="Content about the claim.",
        )
        result = engine.analyze("Claim.", [ev])

        a = result.analyzed_evidence[0]
        assert a.url == "https://test.com/article"
        assert a.title == "Test Article Title"
        assert a.source_name == "test.com"
        assert a.published_at == "2024-06-15"


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    print("Running EvidenceEngine unit tests (all mocked)...\n")
    unittest.main(verbosity=2)
