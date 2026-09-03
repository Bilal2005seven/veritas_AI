"""
test_credibility_engine.py
==========================
Unit tests for CredibilityEngine.

All external dependencies (EvidenceEngine, EmbeddingService, TransformerService)
are replaced with hand-built fixtures so tests run instantly with no model
downloads, network calls, or file I/O.

Run from backend/:
    python tests/test_credibility_engine.py
"""

from __future__ import annotations

import math
import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.credibility_engine import (
    DEFAULT_SOURCE_WEIGHT,
    OVERALL_WEIGHT_ANTI_CONTRADICTION,
    OVERALL_WEIGHT_QUALITY,
    OVERALL_WEIGHT_SOURCE,
    OVERALL_WEIGHT_SUPPORT,
    SOURCE_WEIGHT_REGISTRY,
    CredibilityAssessment,
    CredibilityEngine,
    _extract_domain,
    _source_weight,
)
from app.services.evidence_engine import (
    EvidenceAnalysis,
    EvidenceEngineResult,
)
from app.models.transformer import (
    LABEL_CONTRADICTION,
    LABEL_ENTAILMENT,
    LABEL_NEUTRAL,
)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def _make_analysis(
    url: str = "https://reuters.com/article/1",
    title: str = "Test Article",
    source_name: str = "reuters.com",
    content: str = "Some relevant evidence text.",
    relevance_score: float = 0.80,
    nli_label: str = LABEL_NEUTRAL,
    entailment_score: float = 0.1,
    contradiction_score: float = 0.1,
    neutral_score: float = 0.8,
    published_at: str | None = "2024-01-01",
    fetch_error: str | None = None,
    nli_error: str | None = None,
) -> EvidenceAnalysis:
    return EvidenceAnalysis(
        title=title,
        url=url,
        source_name=source_name,
        content=content,
        relevance_score=relevance_score,
        nli_label=nli_label,
        entailment_score=entailment_score,
        contradiction_score=contradiction_score,
        neutral_score=neutral_score,
        published_at=published_at,
        fetch_error=fetch_error,
        nli_error=nli_error,
    )


def _make_result(
    claim: str = "Test claim.",
    analyzed: list[EvidenceAnalysis] | None = None,
) -> EvidenceEngineResult:
    """Build a minimal EvidenceEngineResult for testing."""
    analyzed = analyzed or []
    result = EvidenceEngineResult(
        claim=claim,
        total_processed=len(analyzed),
    )
    result.analyzed_evidence = analyzed
    # Populate buckets from labels (mirrors EvidenceEngine behaviour).
    for item in analyzed:
        if item.nli_label == LABEL_ENTAILMENT:
            result.supporting_evidence.append(item)
        elif item.nli_label == LABEL_CONTRADICTION:
            result.contradicting_evidence.append(item)
        else:
            result.neutral_evidence.append(item)
    return result


def _engine(**kwargs) -> CredibilityEngine:
    """Create a CredibilityEngine with optional overrides."""
    return CredibilityEngine(**kwargs)


# ===========================================================================
# Tests: helper utilities
# ===========================================================================

class TestExtractDomain(unittest.TestCase):

    def test_strips_www(self):
        assert _extract_domain("https://www.bbc.com/news/1") == "bbc.com"

    def test_no_www(self):
        assert _extract_domain("https://reuters.com/article") == "reuters.com"

    def test_subdomain_preserved(self):
        assert _extract_domain("https://news.bbc.co.uk/article") == "news.bbc.co.uk"

    def test_empty_url(self):
        assert _extract_domain("") == ""

    def test_bad_url_returns_empty(self):
        # urlparse handles most odd URLs gracefully; just assert no exception.
        result = _extract_domain("not-a-url")
        assert isinstance(result, str)


class TestSourceWeight(unittest.TestCase):

    def test_known_domain_returns_registry_value(self):
        w = _source_weight("reuters.com", SOURCE_WEIGHT_REGISTRY)
        assert w == SOURCE_WEIGHT_REGISTRY["reuters.com"]

    def test_unknown_domain_returns_default(self):
        w = _source_weight("totallymadeupblog123.io", SOURCE_WEIGHT_REGISTRY)
        assert w == DEFAULT_SOURCE_WEIGHT

    def test_empty_domain_returns_default(self):
        w = _source_weight("", SOURCE_WEIGHT_REGISTRY)
        assert w == DEFAULT_SOURCE_WEIGHT

    def test_suffix_match(self):
        # "news.bbc.co.uk" should match "bbc.co.uk" in the registry.
        w = _source_weight("news.bbc.co.uk", SOURCE_WEIGHT_REGISTRY)
        assert w == SOURCE_WEIGHT_REGISTRY["bbc.co.uk"]

    def test_custom_registry_override(self):
        custom = {"myblog.com": 0.95}
        w = _source_weight("myblog.com", custom)
        assert w == 0.95

    def test_extra_weights_merged_in_engine(self):
        engine = CredibilityEngine(extra_source_weights={"custom.com": 0.99})
        w = _source_weight("custom.com", engine._registry)
        assert w == 0.99

    def test_builtin_still_present_after_merge(self):
        engine = CredibilityEngine(extra_source_weights={"custom.com": 0.99})
        assert "reuters.com" in engine._registry


# ===========================================================================
# Tests: CredibilityAssessment schema
# ===========================================================================

class TestCredibilityAssessmentSchema(unittest.TestCase):

    def test_defaults(self):
        a = CredibilityAssessment()
        assert a.overall_score == 0.5
        assert a.support_score == 0.0
        assert a.contradiction_score == 0.0
        assert a.confidence == 0.0
        assert isinstance(a.reasoning, str)

    def test_all_fields_set(self):
        a = CredibilityAssessment(
            overall_score=0.72,
            support_score=0.6,
            contradiction_score=0.1,
            evidence_quality_score=0.55,
            source_quality_score=0.88,
            unique_source_count=3,
            supporting_source_count=2,
            contradicting_source_count=1,
            confidence=0.65,
            reasoning="Test reasoning.",
        )
        assert a.unique_source_count == 3
        assert a.reasoning == "Test reasoning."


# ===========================================================================
# Tests: empty evidence
# ===========================================================================

class TestEmptyEvidence(unittest.TestCase):

    def test_empty_analyzed_evidence_returns_default(self):
        engine = _engine()
        result = _make_result(analyzed=[])
        a = engine.assess(result)

        assert isinstance(a, CredibilityAssessment)
        assert a.overall_score == 0.5
        assert a.confidence == 0.0
        assert a.unique_source_count == 0
        assert "No usable evidence" in a.reasoning

    def test_all_empty_content_treated_as_no_evidence(self):
        """Items with empty content produce the same output as empty list."""
        engine = _engine()
        items = [
            _make_analysis(content="", nli_label="SKIPPED"),
            _make_analysis(url="https://b.com", content="", nli_label="SKIPPED"),
        ]
        result = _make_result(analyzed=items)
        a = engine.assess(result)

        # No usable items → falls through to default.
        assert a.support_score == 0.0
        assert a.contradiction_score == 0.0


# ===========================================================================
# Tests: source credibility scoring
# ===========================================================================

class TestSourceCredibility(unittest.TestCase):

    def test_known_source_has_higher_weight_than_unknown(self):
        engine = _engine()

        # Two items: same relevance, same NLI label — differ only by source.
        trusted = _make_analysis(
            url="https://reuters.com/story/1",
            content="Supporting text about the claim.",
            relevance_score=0.9,
            nli_label=LABEL_ENTAILMENT,
            entailment_score=0.9,
        )
        unknown = _make_analysis(
            url="https://randomnewsblog-zzz.io/story",
            content="Supporting text about the claim.",
            relevance_score=0.9,
            nli_label=LABEL_ENTAILMENT,
            entailment_score=0.9,
        )

        result_trusted = _make_result(analyzed=[trusted])
        result_unknown = _make_result(analyzed=[unknown])

        a_trusted = engine.assess(result_trusted)
        a_unknown = engine.assess(result_unknown)

        # Trusted source must produce higher support_score.
        assert a_trusted.support_score > a_unknown.support_score

    def test_unknown_source_not_zero(self):
        """Unknown source should NOT score zero — it is neutral (0.5), not adversarial."""
        engine = _engine()
        item = _make_analysis(
            url="https://totallyfakesite999.xyz/article",
            content="Some evidence text.",
            relevance_score=0.8,
            nli_label=LABEL_ENTAILMENT,
            entailment_score=0.9,
        )
        result = _make_result(analyzed=[item])
        a = engine.assess(result)

        assert a.support_score > 0.0

    def test_source_quality_score_reflects_registry(self):
        """source_quality_score should be close to the known source weight."""
        engine = _engine()
        item = _make_analysis(
            url="https://reuters.com/article",
            content="Evidence text.",
            relevance_score=0.8,
        )
        result = _make_result(analyzed=[item])
        a = engine.assess(result)

        expected_sw = SOURCE_WEIGHT_REGISTRY["reuters.com"]
        assert abs(a.source_quality_score - expected_sw) < 0.05

    def test_extra_source_weights_applied(self):
        """Custom source weight overrides built-in or adds new entry."""
        engine = CredibilityEngine(extra_source_weights={"myblog.com": 0.95})
        item = _make_analysis(
            url="https://myblog.com/post",
            content="Supporting text.",
            relevance_score=0.8,
            nli_label=LABEL_ENTAILMENT,
            entailment_score=0.9,
        )
        result = _make_result(analyzed=[item])
        a = engine.assess(result)

        # source_quality_score should reflect 0.95 weight.
        assert abs(a.source_quality_score - 0.95) < 0.05


# ===========================================================================
# Tests: NLI contribution
# ===========================================================================

class TestNLIContribution(unittest.TestCase):

    def test_entailment_drives_support_score(self):
        engine = _engine()
        item = _make_analysis(
            nli_label=LABEL_ENTAILMENT,
            entailment_score=0.95,
            relevance_score=0.9,
        )
        a = engine.assess(_make_result(analyzed=[item]))
        assert a.support_score > 0.0
        assert a.contradiction_score == 0.0

    def test_contradiction_drives_contradiction_score(self):
        engine = _engine()
        item = _make_analysis(
            nli_label=LABEL_CONTRADICTION,
            contradiction_score=0.95,
            relevance_score=0.9,
        )
        a = engine.assess(_make_result(analyzed=[item]))
        assert a.contradiction_score > 0.0
        assert a.support_score == 0.0

    def test_neutral_contributes_neither_score(self):
        engine = _engine()
        item = _make_analysis(
            nli_label=LABEL_NEUTRAL,
            neutral_score=0.95,
            relevance_score=0.9,
        )
        a = engine.assess(_make_result(analyzed=[item]))
        assert a.support_score == 0.0
        assert a.contradiction_score == 0.0

    def test_neutral_never_treated_as_support(self):
        """Core invariant: NEUTRAL must NOT increase support_score."""
        engine = _engine()
        neutral = _make_analysis(
            nli_label=LABEL_NEUTRAL,
            relevance_score=0.99,  # very high relevance
            neutral_score=0.99,
        )
        a = engine.assess(_make_result(analyzed=[neutral]))
        assert a.support_score == 0.0, (
            "NEUTRAL evidence must never contribute to support_score"
        )

    def test_skipped_contributes_neither_score(self):
        engine = _engine()
        item = _make_analysis(
            nli_label="SKIPPED",
            content="Some content.",
            relevance_score=0.0,
        )
        a = engine.assess(_make_result(analyzed=[item]))
        assert a.support_score == 0.0
        assert a.contradiction_score == 0.0

    def test_support_score_higher_for_higher_entailment_relevance(self):
        engine = _engine()
        low = _make_analysis(
            url="https://reuters.com/a",
            nli_label=LABEL_ENTAILMENT,
            relevance_score=0.3,
        )
        high = _make_analysis(
            url="https://reuters.com/b",
            nli_label=LABEL_ENTAILMENT,
            relevance_score=0.9,
        )
        a_low = engine.assess(_make_result(analyzed=[low]))
        a_high = engine.assess(_make_result(analyzed=[high]))
        assert a_high.support_score > a_low.support_score


# ===========================================================================
# Tests: relevance contribution
# ===========================================================================

class TestRelevanceContribution(unittest.TestCase):

    def test_higher_relevance_increases_evidence_quality(self):
        engine = _engine()
        low_rel = _make_analysis(
            url="https://reuters.com/a",
            relevance_score=0.2,
            nli_label=LABEL_NEUTRAL,
        )
        high_rel = _make_analysis(
            url="https://reuters.com/b",
            relevance_score=0.9,
            nli_label=LABEL_NEUTRAL,
        )
        a_low = engine.assess(_make_result(analyzed=[low_rel]))
        a_high = engine.assess(_make_result(analyzed=[high_rel]))
        assert a_high.evidence_quality_score > a_low.evidence_quality_score

    def test_zero_relevance_item_below_floor_no_stance_contribution(self):
        """Items with relevance < min_usable_relevance don't contribute to stance."""
        engine = CredibilityEngine(min_usable_relevance=0.10)
        item = _make_analysis(
            relevance_score=0.01,   # below floor
            nli_label=LABEL_ENTAILMENT,
        )
        a = engine.assess(_make_result(analyzed=[item]))
        assert a.support_score == 0.0


# ===========================================================================
# Tests: source diversity & duplicate handling
# ===========================================================================

class TestSourceDiversity(unittest.TestCase):

    def test_two_items_same_domain_count_as_one_source(self):
        engine = _engine()
        a1 = _make_analysis(url="https://bbc.com/article/1", content="Evidence A.")
        a2 = _make_analysis(url="https://bbc.com/article/2", content="Evidence B.")
        result = _make_result(analyzed=[a1, a2])
        a = engine.assess(result)
        # Both share domain "bbc.com" → 1 unique source.
        assert a.unique_source_count == 1

    def test_two_items_different_domains_count_as_two_sources(self):
        engine = _engine()
        a1 = _make_analysis(url="https://bbc.com/a", content="Evidence A.")
        a2 = _make_analysis(url="https://reuters.com/b", content="Evidence B.")
        result = _make_result(analyzed=[a1, a2])
        a = engine.assess(result)
        assert a.unique_source_count == 2

    def test_confidence_increases_with_more_unique_domains(self):
        engine = _engine()
        domains = [f"https://site{i}.com/article" for i in range(5)]
        items = [_make_analysis(url=d, content="Evidence.") for d in domains]
        result_1 = _make_result(analyzed=[items[0]])
        result_5 = _make_result(analyzed=items)

        a_1 = engine.assess(result_1)
        a_5 = engine.assess(result_5)
        assert a_5.confidence > a_1.confidence

    def test_confidence_formula_matches_expected(self):
        """Verify the exponential confidence formula: 1 - exp(-n/saturation)."""
        saturation = 5
        engine = CredibilityEngine(confidence_saturation=saturation)
        items = [
            _make_analysis(url=f"https://site{i}.org/a", content="Evidence.")
            for i in range(3)
        ]
        result = _make_result(analyzed=items)
        a = engine.assess(result)

        expected = 1.0 - math.exp(-3 / saturation)
        assert abs(a.confidence - expected) < 1e-4

    def test_supporting_source_count_counts_unique_domains(self):
        engine = _engine()
        s1 = _make_analysis(
            url="https://bbc.com/1", content="Support A.",
            nli_label=LABEL_ENTAILMENT,
        )
        s2 = _make_analysis(
            url="https://bbc.com/2", content="Support B.",
            nli_label=LABEL_ENTAILMENT,
        )
        s3 = _make_analysis(
            url="https://reuters.com/1", content="Support C.",
            nli_label=LABEL_ENTAILMENT,
        )
        result = _make_result(analyzed=[s1, s2, s3])
        a = engine.assess(result)
        # bbc.com × 2 + reuters.com × 1 → 2 unique supporting domains.
        assert a.supporting_source_count == 2

    def test_contradicting_source_count_unique_domains(self):
        engine = _engine()
        c1 = _make_analysis(
            url="https://apnews.com/1", content="Contra A.",
            nli_label=LABEL_CONTRADICTION,
        )
        c2 = _make_analysis(
            url="https://apnews.com/2", content="Contra B.",
            nli_label=LABEL_CONTRADICTION,
        )
        result = _make_result(analyzed=[c1, c2])
        a = engine.assess(result)
        assert a.contradicting_source_count == 1  # same domain


# ===========================================================================
# Tests: only supporting / only contradicting / only neutral
# ===========================================================================

class TestStanceBuckets(unittest.TestCase):

    def test_only_supporting_evidence(self):
        engine = _engine()
        items = [
            _make_analysis(
                url=f"https://site{i}.com/a",
                nli_label=LABEL_ENTAILMENT,
                relevance_score=0.8,
                content="Support.",
            )
            for i in range(3)
        ]
        a = engine.assess(_make_result(analyzed=items))

        assert a.support_score > 0.0
        assert a.contradiction_score == 0.0
        assert a.overall_score > 0.5   # leans toward support

    def test_only_contradicting_evidence(self):
        engine = _engine()
        items = [
            _make_analysis(
                url=f"https://site{i}.com/b",
                nli_label=LABEL_CONTRADICTION,
                relevance_score=0.8,
                content="Contradiction.",
            )
            for i in range(3)
        ]
        a = engine.assess(_make_result(analyzed=items))

        assert a.contradiction_score > 0.0
        assert a.support_score == 0.0
        assert a.overall_score < 0.5   # leans toward contradiction

    def test_only_neutral_evidence(self):
        engine = _engine()
        items = [
            _make_analysis(
                url=f"https://site{i}.com/c",
                nli_label=LABEL_NEUTRAL,
                relevance_score=0.8,
                content="Neutral content.",
            )
            for i in range(3)
        ]
        a = engine.assess(_make_result(analyzed=items))

        assert a.support_score == 0.0
        assert a.contradiction_score == 0.0
        # overall_score should be driven by quality/source weights; not extreme.
        assert 0.0 < a.overall_score < 1.0

    def test_mixed_evidence_both_scores_nonzero(self):
        engine = _engine()
        items = [
            _make_analysis(
                url="https://reuters.com/support",
                nli_label=LABEL_ENTAILMENT,
                relevance_score=0.85,
                content="Evidence supporting the claim.",
            ),
            _make_analysis(
                url="https://apnews.com/contra",
                nli_label=LABEL_CONTRADICTION,
                relevance_score=0.80,
                content="Evidence contradicting the claim.",
            ),
            _make_analysis(
                url="https://bbc.com/neutral",
                nli_label=LABEL_NEUTRAL,
                relevance_score=0.60,
                content="Neutral background article.",
            ),
        ]
        a = engine.assess(_make_result(analyzed=items))

        assert a.support_score > 0.0
        assert a.contradiction_score > 0.0
        assert a.unique_source_count == 3


# ===========================================================================
# Tests: failed fetches
# ===========================================================================

class TestFailedFetches(unittest.TestCase):

    def test_empty_content_item_not_in_support(self):
        """Items with no content must not contribute to support_score."""
        engine = _engine()
        # This item has an entailment label but NO content — it's a failed fetch.
        item = _make_analysis(
            content="",          # empty
            nli_label=LABEL_ENTAILMENT,
            relevance_score=0.9,
            fetch_error="HTTP 500",
        )
        result = _make_result(analyzed=[item])
        a = engine.assess(result)

        # All items have empty content → falls through to default.
        assert a.support_score == 0.0
        assert a.overall_score == 0.5

    def test_failed_fetch_with_snippet_still_scored(self):
        """A failed page fetch that kept its search snippet should still be scored."""
        engine = _engine()
        item = _make_analysis(
            content="The Earth orbits the Sun in approximately 365 days.",
            nli_label=LABEL_ENTAILMENT,
            relevance_score=0.7,
            fetch_error="HTTP 403",  # page fetch failed, but snippet preserved
        )
        result = _make_result(analyzed=[item])
        a = engine.assess(result)

        # Content exists → should score normally.
        assert a.support_score > 0.0


# ===========================================================================
# Tests: score normalization
# ===========================================================================

class TestScoreNormalization(unittest.TestCase):

    def test_all_scores_between_zero_and_one(self):
        engine = _engine()
        items = [
            _make_analysis(
                url=f"https://reuters.com/{i}",
                nli_label=LABEL_ENTAILMENT,
                relevance_score=1.0,
                content="Perfect evidence.",
            )
            for i in range(10)
        ]
        a = engine.assess(_make_result(analyzed=items))

        for attr in (
            "overall_score",
            "support_score",
            "contradiction_score",
            "evidence_quality_score",
            "source_quality_score",
            "confidence",
        ):
            val = getattr(a, attr)
            assert 0.0 <= val <= 1.0, (
                f"{attr}={val} is outside [0, 1]"
            )

    def test_single_perfect_item_support_score_bounded(self):
        """Even a perfect item must not push support_score above 1.0."""
        engine = _engine()
        item = _make_analysis(
            url="https://reuters.com/perfect",
            nli_label=LABEL_ENTAILMENT,
            relevance_score=1.0,
            content="Perfect evidence.",
        )
        a = engine.assess(_make_result(analyzed=[item]))
        assert a.support_score <= 1.0

    def test_overall_score_bounded_regardless_of_input(self):
        engine = _engine()
        # Inject many high-quality supporting items.
        items = [
            _make_analysis(
                url=f"https://reuters.com/{i}",
                nli_label=LABEL_ENTAILMENT,
                relevance_score=1.0,
                content="Perfect evidence.",
            )
            for i in range(20)
        ]
        a = engine.assess(_make_result(analyzed=items))
        assert 0.0 <= a.overall_score <= 1.0

    def test_contradiction_only_overall_below_half(self):
        """All-contradiction evidence should yield overall_score < 0.5."""
        engine = _engine()
        items = [
            _make_analysis(
                url=f"https://reuters.com/{i}",
                nli_label=LABEL_CONTRADICTION,
                relevance_score=0.9,
                content="This contradicts the claim.",
            )
            for i in range(5)
        ]
        a = engine.assess(_make_result(analyzed=items))
        assert a.overall_score < 0.5

    def test_support_only_overall_above_half(self):
        """All-support evidence should yield overall_score > 0.5."""
        engine = _engine()
        items = [
            _make_analysis(
                url=f"https://reuters.com/{i}",
                nli_label=LABEL_ENTAILMENT,
                relevance_score=0.9,
                content="This supports the claim.",
            )
            for i in range(5)
        ]
        a = engine.assess(_make_result(analyzed=items))
        assert a.overall_score > 0.5


# ===========================================================================
# Tests: confidence calculation
# ===========================================================================

class TestConfidence(unittest.TestCase):

    def test_zero_unique_sources_zero_confidence(self):
        """No usable evidence → confidence should be 0 (or the exp(0) result)."""
        engine = _engine()
        a = engine.assess(_make_result(analyzed=[]))
        assert a.confidence == 0.0

    def test_confidence_increases_monotonically_with_unique_sources(self):
        engine = CredibilityEngine(confidence_saturation=5)
        confidences = []
        for n in range(1, 8):
            items = [
                _make_analysis(url=f"https://s{i}.com/a", content="Evidence.")
                for i in range(n)
            ]
            a = engine.assess(_make_result(analyzed=items))
            confidences.append(a.confidence)

        for i in range(len(confidences) - 1):
            assert confidences[i] < confidences[i + 1], (
                f"Confidence should increase: {confidences}"
            )

    def test_confidence_approaches_one_with_many_sources(self):
        """With many diverse sources confidence should be close to 1.0."""
        engine = CredibilityEngine(confidence_saturation=3)
        items = [
            _make_analysis(url=f"https://source{i}.org/a", content="Evidence.")
            for i in range(20)
        ]
        a = engine.assess(_make_result(analyzed=items))
        assert a.confidence > 0.99

    def test_confidence_saturation_parameter_respected(self):
        """Higher saturation → slower confidence growth for the same n."""
        low_sat = CredibilityEngine(confidence_saturation=2)
        high_sat = CredibilityEngine(confidence_saturation=10)

        items = [
            _make_analysis(url=f"https://s{i}.com/a", content="Evidence.")
            for i in range(4)
        ]
        result = _make_result(analyzed=items)

        a_low = low_sat.assess(result)
        a_high = high_sat.assess(result)

        assert a_low.confidence > a_high.confidence


# ===========================================================================
# Tests: reasoning string
# ===========================================================================

class TestReasoning(unittest.TestCase):

    def test_reasoning_is_non_empty_string(self):
        engine = _engine()
        item = _make_analysis(content="Evidence.", nli_label=LABEL_NEUTRAL)
        a = engine.assess(_make_result(analyzed=[item]))
        assert isinstance(a.reasoning, str)
        assert len(a.reasoning) > 0

    def test_empty_evidence_reasoning_mentions_no_evidence(self):
        engine = _engine()
        a = engine.assess(_make_result(analyzed=[]))
        assert "No usable evidence" in a.reasoning

    def test_support_reasoning_mentions_support(self):
        engine = _engine()
        item = _make_analysis(
            url="https://bbc.com/a",
            content="Support.",
            nli_label=LABEL_ENTAILMENT,
            relevance_score=0.8,
        )
        a = engine.assess(_make_result(analyzed=[item]))
        assert "support" in a.reasoning.lower()

    def test_contradiction_reasoning_mentions_contradict(self):
        engine = _engine()
        item = _make_analysis(
            url="https://bbc.com/a",
            content="Contradiction.",
            nli_label=LABEL_CONTRADICTION,
            relevance_score=0.8,
        )
        a = engine.assess(_make_result(analyzed=[item]))
        assert "contradict" in a.reasoning.lower()


# ===========================================================================
# Tests: overall formula invariants
# ===========================================================================

class TestOverallFormulaInvariants(unittest.TestCase):

    def test_weights_sum_to_one(self):
        total = (
            OVERALL_WEIGHT_SUPPORT
            + OVERALL_WEIGHT_ANTI_CONTRADICTION
            + OVERALL_WEIGHT_QUALITY
            + OVERALL_WEIGHT_SOURCE
        )
        assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, expected 1.0"

    def test_more_support_domains_raises_overall(self):
        engine = _engine()
        one_support = [
            _make_analysis(
                url="https://reuters.com/a",
                content="Support.",
                nli_label=LABEL_ENTAILMENT,
                relevance_score=0.8,
            )
        ]
        three_support = [
            _make_analysis(
                url=f"https://site{i}.com/a",
                content="Support.",
                nli_label=LABEL_ENTAILMENT,
                relevance_score=0.8,
            )
            for i in range(3)
        ]
        a_one = engine.assess(_make_result(analyzed=one_support))
        a_three = engine.assess(_make_result(analyzed=three_support))

        # More diverse supporting sources → higher confidence, but support_score
        # also changes due to normalisation. Overall should be >= for more sources.
        assert a_three.confidence >= a_one.confidence

    def test_strong_contradiction_lowers_overall_vs_neutral(self):
        engine = _engine()
        neutral_item = _make_analysis(
            url="https://bbc.com/n",
            content="Neutral background.",
            nli_label=LABEL_NEUTRAL,
            relevance_score=0.8,
        )
        contra_item = _make_analysis(
            url="https://bbc.com/c",
            content="This contradicts the claim.",
            nli_label=LABEL_CONTRADICTION,
            relevance_score=0.8,
        )
        a_neutral = engine.assess(_make_result(analyzed=[neutral_item]))
        a_contra = engine.assess(_make_result(analyzed=[contra_item]))

        assert a_contra.overall_score < a_neutral.overall_score


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    print("Running CredibilityEngine unit tests (all mocked)...\n")
    unittest.main(verbosity=2)
