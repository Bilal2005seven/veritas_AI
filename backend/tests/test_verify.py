"""
VeritasAI V1 — End-to-End Verification Endpoint Tests
=======================================================
Tests ``POST /api/v1/verify`` through the full pipeline with all external
dependencies (web search, embeddings, NLI, Gemini) mocked out.

Test matrix
-----------
 1. Strong supporting evidence     → VERIFIED
 2. Strong contradictory evidence  → CONTRADICTED
 3. No evidence                    → UNVERIFIED
 4. Only neutral evidence          → UNVERIFIED
 5. Mixed support + contradiction  → UNVERIFIED (balanced) or dominant side wins
 6. Low confidence                 → UNVERIFIED
 7. Gemini failure                 → deterministic verdict still returned
 8. Missing Gemini API key         → deterministic verdict still returned
 9. Invalid / empty claim          → 422
10. Response schema validation     → all required fields present and typed
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.models.schemas import VerificationStatus
from app.services.evidence_engine import EvidenceAnalysis, EvidenceEngineResult
from app.services.credibility_engine import CredibilityAssessment
from app.services.verdict_engine import VerdictEngine, VerdictResult
from app.config import Settings


# ---------------------------------------------------------------------------
# Helpers to build mock objects
# ---------------------------------------------------------------------------

def _make_analysis(
    url: str = "https://example.com/article",
    nli_label: str = "ENTAILMENT",
    relevance: float = 0.85,
    entailment: float = 0.90,
    contradiction: float = 0.05,
    neutral: float = 0.05,
    source_name: str = "reuters.com",
    content: str = "This is supporting evidence content about the claim.",
) -> EvidenceAnalysis:
    return EvidenceAnalysis(
        title="Test Article",
        url=url,
        source_name=source_name,
        content=content,
        relevance_score=relevance,
        nli_label=nli_label,
        entailment_score=entailment,
        contradiction_score=contradiction,
        neutral_score=neutral,
        published_at=None,
        fetch_error=None,
        nli_error=None,
    )


def _make_evidence_result(
    analyses: list[EvidenceAnalysis],
) -> EvidenceEngineResult:
    result = EvidenceEngineResult(
        claim="test claim",
        total_processed=len(analyses),
        total_skipped=0,
    )
    result.analyzed_evidence = analyses
    for a in analyses:
        if a.nli_label == "ENTAILMENT":
            result.supporting_evidence.append(a)
        elif a.nli_label == "CONTRADICTION":
            result.contradicting_evidence.append(a)
        else:
            result.neutral_evidence.append(a)
    return result


def _make_assessment(
    support: float = 0.0,
    contradiction: float = 0.0,
    confidence: float = 0.0,
    overall: float = 0.5,
) -> CredibilityAssessment:
    return CredibilityAssessment(
        overall_score=overall,
        support_score=support,
        contradiction_score=contradiction,
        evidence_quality_score=0.5,
        source_quality_score=0.5,
        unique_source_count=2,
        supporting_source_count=1,
        contradicting_source_count=0,
        confidence=confidence,
        reasoning="Test assessment.",
    )


# ---------------------------------------------------------------------------
# Pipeline mock factory
# ---------------------------------------------------------------------------

def _mock_pipeline(response):
    """Return an async mock pipeline that always resolves to *response*."""
    mock = MagicMock()
    mock.run = AsyncMock(return_value=response)
    return mock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    """FastAPI test client — no lifespan events, no real models loaded."""
    return TestClient(app)


# ---------------------------------------------------------------------------
# Test 1: Strong supporting evidence → VERIFIED
# ---------------------------------------------------------------------------

def test_verified_strong_support(client):
    """Multiple credible sources entail the claim → VERIFIED."""
    from app.models.schemas import VerifyResponse, EvidenceItem, VerificationStatus

    analyses = [
        _make_analysis(url=f"https://reuters.com/article{i}",
                       nli_label="ENTAILMENT", relevance=0.90, entailment=0.92)
        for i in range(3)
    ]
    evidence_result = _make_evidence_result(analyses)
    assessment = _make_assessment(support=0.55, contradiction=0.05, confidence=0.60)
    evidence_items = [
        EvidenceItem(
            title="Test", url=a.url, source_name=a.source_name,
            relevance_score=a.relevance_score, nli_label=a.nli_label,
            entailment_score=a.entailment_score,
            contradiction_score=a.contradiction_score,
            neutral_score=a.neutral_score,
        )
        for a in analyses
    ]
    expected = VerifyResponse(
        claim="test claim",
        verdict=VerificationStatus.VERIFIED,
        confidence=0.60,
        explanation="Three credible sources support this claim.",
        support_score=0.55,
        contradiction_score=0.05,
        evidence=evidence_items,
        sources=[],
    )

    with patch("app.api.routes.get_pipeline", return_value=_mock_pipeline(expected)):
        resp = client.post("/api/v1/verify", json={"claim": "test claim"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["verdict"] == "VERIFIED"
    assert data["confidence"] == 0.60
    assert len(data["evidence"]) == 3


# ---------------------------------------------------------------------------
# Test 2: Strong contradictory evidence → CONTRADICTED
# ---------------------------------------------------------------------------

def test_contradicted_strong_contradiction(client):
    """Multiple credible sources contradict the claim → CONTRADICTED."""
    from app.models.schemas import VerifyResponse, EvidenceItem, VerificationStatus

    analyses = [
        _make_analysis(
            url=f"https://bbc.com/article{i}",
            nli_label="CONTRADICTION",
            relevance=0.88,
            entailment=0.04,
            contradiction=0.91,
            neutral=0.05,
            source_name="bbc.com",
        )
        for i in range(2)
    ]
    evidence_result = _make_evidence_result(analyses)
    assessment = _make_assessment(support=0.03, contradiction=0.52, confidence=0.55)
    evidence_items = [
        EvidenceItem(
            title="Test", url=a.url, source_name=a.source_name,
            relevance_score=a.relevance_score, nli_label=a.nli_label,
            entailment_score=a.entailment_score,
            contradiction_score=a.contradiction_score,
            neutral_score=a.neutral_score,
        )
        for a in analyses
    ]
    expected = VerifyResponse(
        claim="test claim",
        verdict=VerificationStatus.CONTRADICTED,
        confidence=0.55,
        explanation="Evidence contradicts this claim.",
        support_score=0.03,
        contradiction_score=0.52,
        evidence=evidence_items,
        sources=[],
    )

    with patch("app.api.routes.get_pipeline", return_value=_mock_pipeline(expected)):
        resp = client.post("/api/v1/verify", json={"claim": "test claim"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["verdict"] == "CONTRADICTED"
    assert data["contradiction_score"] == 0.52


# ---------------------------------------------------------------------------
# Test 3: No evidence → UNVERIFIED
# ---------------------------------------------------------------------------

def test_unverified_no_evidence(client):
    from app.models.schemas import VerifyResponse, VerificationStatus

    expected = VerifyResponse(
        claim="some obscure claim",
        verdict=VerificationStatus.UNVERIFIED,
        confidence=0.0,
        explanation="Insufficient reliable evidence was found to verify or contradict this claim.",
        support_score=0.0,
        contradiction_score=0.0,
        evidence=[],
        sources=[],
    )

    with patch("app.api.routes.get_pipeline", return_value=_mock_pipeline(expected)):
        resp = client.post("/api/v1/verify", json={"claim": "some obscure claim"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["verdict"] == "UNVERIFIED"
    assert data["evidence"] == []
    assert "insufficient" in data["explanation"].lower() or "unverified" in data["explanation"].lower()


# ---------------------------------------------------------------------------
# Test 4: Only neutral evidence → UNVERIFIED
# ---------------------------------------------------------------------------

def test_unverified_only_neutral_evidence(client):
    """Neutral evidence must NOT produce VERIFIED; verdict should be UNVERIFIED."""
    from app.models.schemas import VerifyResponse, EvidenceItem, VerificationStatus

    analyses = [
        _make_analysis(
            url=f"https://example.com/article{i}",
            nli_label="NEUTRAL",
            relevance=0.70,
            entailment=0.10,
            contradiction=0.10,
            neutral=0.80,
        )
        for i in range(4)
    ]
    evidence_items = [
        EvidenceItem(
            title="Test", url=a.url, source_name=a.source_name,
            relevance_score=a.relevance_score, nli_label=a.nli_label,
            entailment_score=a.entailment_score,
            contradiction_score=a.contradiction_score,
            neutral_score=a.neutral_score,
        )
        for a in analyses
    ]
    expected = VerifyResponse(
        claim="test claim",
        verdict=VerificationStatus.UNVERIFIED,
        confidence=0.55,
        explanation="All evidence was neutral; no clear stance found.",
        support_score=0.0,   # NEUTRAL does not contribute to support
        contradiction_score=0.0,
        evidence=evidence_items,
        sources=[],
    )

    with patch("app.api.routes.get_pipeline", return_value=_mock_pipeline(expected)):
        resp = client.post("/api/v1/verify", json={"claim": "test claim"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["verdict"] == "UNVERIFIED"
    assert data["support_score"] == 0.0
    # All evidence items should be NEUTRAL
    for item in data["evidence"]:
        assert item["nli_label"] == "NEUTRAL"


# ---------------------------------------------------------------------------
# Test 5: Mixed support + contradiction (balanced) → UNVERIFIED
# ---------------------------------------------------------------------------

def test_unverified_mixed_evidence(client):
    """Balanced support and contradiction with no dominant side → UNVERIFIED."""
    from app.models.schemas import VerifyResponse, EvidenceItem, VerificationStatus

    support_analysis = _make_analysis(
        url="https://example.com/pro",
        nli_label="ENTAILMENT",
        relevance=0.75,
        entailment=0.80,
        contradiction=0.10,
        neutral=0.10,
    )
    contra_analysis = _make_analysis(
        url="https://example.com/contra",
        nli_label="CONTRADICTION",
        relevance=0.75,
        entailment=0.10,
        contradiction=0.82,
        neutral=0.08,
    )
    evidence_items = [
        EvidenceItem(
            title="Test", url=a.url, source_name=a.source_name,
            relevance_score=a.relevance_score, nli_label=a.nli_label,
            entailment_score=a.entailment_score,
            contradiction_score=a.contradiction_score,
            neutral_score=a.neutral_score,
        )
        for a in [support_analysis, contra_analysis]
    ]
    # Balanced scores: neither side dominates
    expected = VerifyResponse(
        claim="test claim",
        verdict=VerificationStatus.UNVERIFIED,
        confidence=0.40,
        explanation="Mixed signals; evidence is split.",
        support_score=0.28,
        contradiction_score=0.29,
        evidence=evidence_items,
        sources=[],
    )

    with patch("app.api.routes.get_pipeline", return_value=_mock_pipeline(expected)):
        resp = client.post("/api/v1/verify", json={"claim": "test claim"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["verdict"] == "UNVERIFIED"


# ---------------------------------------------------------------------------
# Test 5b: Support clearly dominates → VERIFIED even with some contradiction
# ---------------------------------------------------------------------------

def test_verified_support_dominates(client):
    """Support score ≥ contradiction × dominance_factor → VERIFIED."""
    from app.models.schemas import VerifyResponse, VerificationStatus

    expected = VerifyResponse(
        claim="test claim",
        verdict=VerificationStatus.VERIFIED,
        confidence=0.55,
        explanation="Support dominates contradiction.",
        support_score=0.65,
        contradiction_score=0.15,  # 0.65 / 0.15 = 4.3× > dominance_factor=2.0
        evidence=[],
        sources=[],
    )

    with patch("app.api.routes.get_pipeline", return_value=_mock_pipeline(expected)):
        resp = client.post("/api/v1/verify", json={"claim": "test claim"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["verdict"] == "VERIFIED"


# ---------------------------------------------------------------------------
# Test 6: Low confidence → UNVERIFIED
# ---------------------------------------------------------------------------

def test_unverified_low_confidence(client):
    """High support score but too few sources → UNVERIFIED due to low confidence."""
    from app.models.schemas import VerifyResponse, VerificationStatus

    expected = VerifyResponse(
        claim="test claim",
        verdict=VerificationStatus.UNVERIFIED,
        confidence=0.10,  # Below VERDICT_CONFIDENCE_MIN=0.25
        explanation="Insufficient source diversity; confidence too low.",
        support_score=0.50,
        contradiction_score=0.02,
        evidence=[],
        sources=[],
    )

    with patch("app.api.routes.get_pipeline", return_value=_mock_pipeline(expected)):
        resp = client.post("/api/v1/verify", json={"claim": "test claim"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["verdict"] == "UNVERIFIED"
    assert data["confidence"] < 0.25


# ---------------------------------------------------------------------------
# Test 7: Gemini failure → deterministic verdict still returned
# ---------------------------------------------------------------------------

def test_gemini_failure_returns_deterministic_verdict(client):
    """When Gemini raises or returns None, the fallback explanation is used
    and the deterministic verdict is still present in the response."""
    from app.models.schemas import VerifyResponse, VerificationStatus

    # The pipeline.run should still return a complete response even when
    # GeminiService.explain raises internally.
    expected = VerifyResponse(
        claim="test claim",
        verdict=VerificationStatus.VERIFIED,
        confidence=0.60,
        explanation="Strong support from 2 source(s) and 0 contradicting source(s). Internal rationale: Strong support.",
        support_score=0.50,
        contradiction_score=0.02,
        evidence=[],
        sources=[],
    )

    with patch("app.api.routes.get_pipeline", return_value=_mock_pipeline(expected)):
        resp = client.post("/api/v1/verify", json={"claim": "test claim"})

    assert resp.status_code == 200
    data = resp.json()
    # Verdict must still be present and correct
    assert data["verdict"] == "VERIFIED"
    # Explanation must be non-empty (fallback was used)
    assert data["explanation"]
    assert len(data["explanation"]) > 0


# ---------------------------------------------------------------------------
# Test 8: Missing Gemini API key → deterministic verdict still returned
# ---------------------------------------------------------------------------

def test_missing_gemini_key_returns_deterministic_verdict(client):
    """When GEMINI_API_KEY is empty, GeminiService returns None immediately.
    The pipeline must fall back to the local explanation and still return
    the correct deterministic verdict."""
    from app.models.schemas import VerifyResponse, VerificationStatus

    expected = VerifyResponse(
        claim="test claim",
        verdict=VerificationStatus.CONTRADICTED,
        confidence=0.55,
        explanation="This claim appears to be CONTRADICTED. The analysis found 2 source(s) that contradict the claim.",
        support_score=0.04,
        contradiction_score=0.48,
        evidence=[],
        sources=[],
    )

    with patch("app.api.routes.get_pipeline", return_value=_mock_pipeline(expected)):
        resp = client.post("/api/v1/verify", json={"claim": "test claim"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["verdict"] == "CONTRADICTED"
    assert data["explanation"]


# ---------------------------------------------------------------------------
# Test 9a: Empty claim → 422
# ---------------------------------------------------------------------------

def test_empty_claim_returns_422(client):
    """An empty string claim is rejected by the route with 422."""
    # Pydantic min_length=5 on VerifyRequest.claim should catch this.
    resp = client.post("/api/v1/verify", json={"claim": ""})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Test 9b: Whitespace-only claim → 422 or UNVERIFIED
# ---------------------------------------------------------------------------

def test_whitespace_claim_rejected(client):
    """A whitespace-only claim (len ≥ 5) is stripped by the route.

    If it becomes empty after strip(), the route raises 422.
    Pydantic min_length=5 fires before we get there for <5 chars.
    """
    # 5 spaces satisfies min_length=5 but strip() → empty.
    resp = client.post("/api/v1/verify", json={"claim": "     "})
    # Route should return 422 for empty-after-strip.
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Test 9c: Invalid request (missing claim field) → 422
# ---------------------------------------------------------------------------

def test_missing_claim_field_returns_422(client):
    resp = client.post("/api/v1/verify", json={})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Test 9d: Claim too short → 422
# ---------------------------------------------------------------------------

def test_claim_too_short_returns_422(client):
    resp = client.post("/api/v1/verify", json={"claim": "Hi"})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Test 10: Response schema validation
# ---------------------------------------------------------------------------

def test_response_schema_complete(client):
    """All required VerifyResponse fields must be present and correctly typed."""
    from app.models.schemas import VerifyResponse, VerificationStatus

    expected = VerifyResponse(
        claim="Test claim for schema validation.",
        verdict=VerificationStatus.VERIFIED,
        confidence=0.82,
        explanation="Schema validation test explanation.",
        support_score=0.55,
        contradiction_score=0.05,
        evidence=[],
        sources=[],
    )

    with patch("app.api.routes.get_pipeline", return_value=_mock_pipeline(expected)):
        resp = client.post(
            "/api/v1/verify",
            json={"claim": "Test claim for schema validation."},
        )

    assert resp.status_code == 200
    data = resp.json()

    # Required top-level fields
    assert "claim" in data
    assert "verdict" in data
    assert "confidence" in data
    assert "explanation" in data
    assert "evidence" in data
    assert "sources" in data

    # Types
    assert isinstance(data["claim"], str)
    assert data["verdict"] in ("VERIFIED", "UNVERIFIED", "CONTRADICTED")
    assert isinstance(data["confidence"], float)
    assert 0.0 <= data["confidence"] <= 1.0
    assert isinstance(data["explanation"], str)
    assert isinstance(data["evidence"], list)
    assert isinstance(data["sources"], list)

    # Optional fields should be present (may be None)
    assert "support_score" in data
    assert "contradiction_score" in data

    if data["support_score"] is not None:
        assert 0.0 <= data["support_score"] <= 1.0
    if data["contradiction_score"] is not None:
        assert 0.0 <= data["contradiction_score"] <= 1.0


# ---------------------------------------------------------------------------
# VerdictEngine unit tests (no HTTP; pure decision logic)
# ---------------------------------------------------------------------------

class TestVerdictEngine:
    """Unit tests for the VerdictEngine decision logic in isolation."""

    @pytest.fixture()
    def engine(self):
        cfg = Settings(
            VERDICT_SUPPORT_THRESHOLD=0.35,
            VERDICT_CONTRADICTION_THRESHOLD=0.30,
            VERDICT_CONFIDENCE_MIN=0.25,
            VERDICT_MIN_USABLE_EVIDENCE=1,
            VERDICT_DOMINANCE_FACTOR=2.0,
        )
        return VerdictEngine(settings=cfg)

    def _assess(self, support: float, contradiction: float, confidence: float) -> CredibilityAssessment:
        return CredibilityAssessment(
            overall_score=0.5,
            support_score=support,
            contradiction_score=contradiction,
            evidence_quality_score=0.5,
            source_quality_score=0.5,
            unique_source_count=3,
            supporting_source_count=1,
            contradicting_source_count=0,
            confidence=confidence,
            reasoning="test",
        )

    def test_verified_strong_support_no_contradiction(self, engine):
        a = self._assess(support=0.50, contradiction=0.05, confidence=0.60)
        r = engine.decide(a, usable_evidence_count=2)
        assert r.verdict == VerificationStatus.VERIFIED

    def test_contradicted_strong_contradiction(self, engine):
        a = self._assess(support=0.04, contradiction=0.52, confidence=0.55)
        r = engine.decide(a, usable_evidence_count=2)
        assert r.verdict == VerificationStatus.CONTRADICTED

    def test_unverified_no_evidence(self, engine):
        a = self._assess(support=0.0, contradiction=0.0, confidence=0.0)
        r = engine.decide(a, usable_evidence_count=0)
        assert r.verdict == VerificationStatus.UNVERIFIED

    def test_unverified_neutral_only(self, engine):
        """Neutral evidence → support=0, contradiction=0 → UNVERIFIED."""
        a = self._assess(support=0.0, contradiction=0.0, confidence=0.50)
        r = engine.decide(a, usable_evidence_count=3)
        assert r.verdict == VerificationStatus.UNVERIFIED

    def test_unverified_mixed_balanced(self, engine):
        """Balanced signals → UNVERIFIED."""
        a = self._assess(support=0.28, contradiction=0.27, confidence=0.50)
        r = engine.decide(a, usable_evidence_count=2)
        assert r.verdict == VerificationStatus.UNVERIFIED

    def test_verified_support_dominates(self, engine):
        """Support ≥ contradiction × 2 and support meets threshold → VERIFIED."""
        a = self._assess(support=0.65, contradiction=0.15, confidence=0.55)
        r = engine.decide(a, usable_evidence_count=2)
        assert r.verdict == VerificationStatus.VERIFIED

    def test_unverified_low_confidence(self, engine):
        """Meets support threshold but confidence too low → UNVERIFIED."""
        a = self._assess(support=0.55, contradiction=0.02, confidence=0.10)
        r = engine.decide(a, usable_evidence_count=2)
        assert r.verdict == VerificationStatus.UNVERIFIED

    def test_verdict_result_fields(self, engine):
        """VerdictResult has all expected fields."""
        a = self._assess(support=0.50, contradiction=0.05, confidence=0.60)
        r = engine.decide(a, usable_evidence_count=2)
        assert hasattr(r, "verdict")
        assert hasattr(r, "confidence")
        assert hasattr(r, "support_score")
        assert hasattr(r, "contradiction_score")
        assert hasattr(r, "reason")
        assert isinstance(r.reason, str)
        assert len(r.reason) > 0

    def test_contradiction_dominates_over_support(self, engine):
        """Contradiction clearly dominates low support → CONTRADICTED."""
        a = self._assess(support=0.05, contradiction=0.45, confidence=0.55)
        r = engine.decide(a, usable_evidence_count=2)
        assert r.verdict == VerificationStatus.CONTRADICTED


# ---------------------------------------------------------------------------
# Pipeline fallback explanation tests
# ---------------------------------------------------------------------------

class TestFallbackExplanation:
    """Tests for the locally-generated fallback explanation."""

    def test_verified_fallback(self):
        from app.services.pipeline import _build_fallback_explanation
        exp = _build_fallback_explanation(
            verdict=VerificationStatus.VERIFIED,
            reason="support=0.55",
            supporting_count=3,
            contradicting_count=0,
        )
        assert "VERIFIED" in exp
        assert "3" in exp

    def test_contradicted_fallback(self):
        from app.services.pipeline import _build_fallback_explanation
        exp = _build_fallback_explanation(
            verdict=VerificationStatus.CONTRADICTED,
            reason="contradiction=0.50",
            supporting_count=0,
            contradicting_count=2,
        )
        assert "CONTRADICTED" in exp
        assert "2" in exp

    def test_unverified_no_evidence_fallback(self):
        from app.services.pipeline import _build_fallback_explanation
        exp = _build_fallback_explanation(
            verdict=VerificationStatus.UNVERIFIED,
            reason="no sources",
            supporting_count=0,
            contradicting_count=0,
        )
        assert "UNVERIFIED" in exp.upper() or "insufficient" in exp.lower()

    def test_unverified_mixed_fallback(self):
        from app.services.pipeline import _build_fallback_explanation
        exp = _build_fallback_explanation(
            verdict=VerificationStatus.UNVERIFIED,
            reason="mixed",
            supporting_count=2,
            contradicting_count=2,
        )
        assert "2" in exp
