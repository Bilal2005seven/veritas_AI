"""
VeritasAI V1 — Verification Pipeline Orchestrator
===================================================
Wires all independent services into the end-to-end verification pipeline for
``POST /api/v1/verify``.

Pipeline stages (in order):
  1. Claim normalisation (lightweight; ClaimExtractorService is unimplemented,
     so we use the raw claim directly for V1).
  2. WebEvaluatorService  → list[WebEvidence]
  3. EvidenceEngine       → EvidenceEngineResult  (embedding + NLI)
  4. CredibilityEngine    → CredibilityAssessment
  5. VerdictEngine        → VerdictResult          (VERIFIED/UNVERIFIED/CONTRADICTED)
  6. GeminiService        → explanation string     (may fail gracefully)
  7. Build VerifyResponse

This module is intentionally I/O-aware only at the WebEvaluator and Gemini
stages.  All other stages are synchronous CPU operations.

Responsibilities NOT in this module:
  * Fetching evidence          → WebEvaluatorService
  * Semantic similarity        → EmbeddingService (inside EvidenceEngine)
  * NLI classification         → TransformerService (inside EvidenceEngine)
  * Evidence/NLI combination   → EvidenceEngine
  * Source credibility scoring → CredibilityEngine
  * Verdict decision logic     → VerdictEngine
  * Explanation generation     → GeminiService
"""

from __future__ import annotations

import logging
from typing import List

from app.config import Settings, settings as _default_settings
from app.models.schemas import (
    CredibilityResult,
    EvidenceItem,
    SourceReference,
    VerificationStatus,
    VerifyResponse,
)
from app.models.transformer import TransformerService
from app.rag.embeddings import EmbeddingService
from app.services.credibility_engine import CredibilityEngine
from app.services.evidence_engine import EvidenceEngine, EvidenceEngineResult
from app.services.gemini_service import GeminiService
from app.services.verdict_engine import VerdictEngine
from app.services.web_evaluator import WebEvaluatorService

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fallback explanation builder (used when Gemini is unavailable / fails)
# ---------------------------------------------------------------------------

def _build_fallback_explanation(
    verdict: VerificationStatus,
    reason: str,
    supporting_count: int,
    contradicting_count: int,
) -> str:
    """
    Generate a local, deterministic explanation when Gemini is unavailable.

    Parameters
    ----------
    verdict : VerificationStatus
    reason : str
        Short internal reason string from VerdictEngine.
    supporting_count : int
        Number of supporting evidence items.
    contradicting_count : int
        Number of contradicting evidence items.

    Returns
    -------
    str
        A plain-language explanation suitable for end-users.
    """
    if verdict == VerificationStatus.VERIFIED:
        base = (
            f"This claim appears to be VERIFIED. "
            f"The analysis found {supporting_count} supporting source(s) and "
            f"{contradicting_count} contradicting source(s). "
        )
    elif verdict == VerificationStatus.CONTRADICTED:
        base = (
            f"This claim appears to be CONTRADICTED. "
            f"The analysis found {contradicting_count} source(s) that contradict "
            f"the claim and {supporting_count} source(s) that support it. "
        )
    else:
        if supporting_count == 0 and contradicting_count == 0:
            return (
                "Insufficient reliable evidence was found to verify or contradict "
                "this claim. The claim remains UNVERIFIED."
            )
        base = (
            f"The evidence was insufficient or mixed to reach a clear verdict. "
            f"Found {supporting_count} supporting and {contradicting_count} "
            f"contradicting source(s). The claim is UNVERIFIED."
        )

    return base + f"Internal rationale: {reason}"


# ---------------------------------------------------------------------------
# Evidence → EvidenceItem schema conversion
# ---------------------------------------------------------------------------

def _analysis_to_evidence_item(analysis) -> EvidenceItem:  # type: ignore[no-untyped-def]
    """
    Convert an :class:`~app.services.evidence_engine.EvidenceAnalysis`
    into the public :class:`~app.models.schemas.EvidenceItem` schema.

    Internal fields (``content``, ``fetch_error``, ``nli_error``) are dropped.
    """
    return EvidenceItem(
        title=analysis.title or "",
        url=analysis.url or "",
        source_name=analysis.source_name or "",
        relevance_score=round(float(analysis.relevance_score), 6),
        nli_label=analysis.nli_label,
        entailment_score=round(float(analysis.entailment_score), 6),
        contradiction_score=round(float(analysis.contradiction_score), 6),
        neutral_score=round(float(analysis.neutral_score), 6),
        published_at=analysis.published_at,
        content_snippet=analysis.snippet[:200] if analysis.content else None,
    )


# ---------------------------------------------------------------------------
# Main orchestrator class
# ---------------------------------------------------------------------------

class VerificationPipeline:
    """
    Orchestrates the full claim verification pipeline.

    All services are injected at construction time so they can be replaced
    with mocks during testing.

    Parameters
    ----------
    web_evaluator : WebEvaluatorService
    evidence_engine : EvidenceEngine
    credibility_engine : CredibilityEngine
    verdict_engine : VerdictEngine
    gemini_service : GeminiService
    settings : Settings
    """

    def __init__(
        self,
        web_evaluator: WebEvaluatorService,
        evidence_engine: EvidenceEngine,
        credibility_engine: CredibilityEngine,
        verdict_engine: VerdictEngine,
        gemini_service: GeminiService,
        settings: Settings,
    ) -> None:
        self._web = web_evaluator
        self._evidence = evidence_engine
        self._credibility = credibility_engine
        self._verdict = verdict_engine
        self._gemini = gemini_service
        self._settings = settings

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, claim: str) -> VerifyResponse:
        """
        Execute the full pipeline for *claim* and return a :class:`VerifyResponse`.

        This method never raises.  All errors are caught, logged, and result
        in a graceful UNVERIFIED response or partial response.

        Parameters
        ----------
        claim : str
            Raw user claim (already validated by Pydantic at the route level).

        Returns
        -------
        VerifyResponse
        """
        logger.info("Pipeline.run: starting for claim=%r", claim[:80])

        # ── Stage 1: Web evidence retrieval ───────────────────────────────────
        try:
            web_evidence = await self._web.evaluate_claim(claim)
        except Exception as exc:
            logger.error("WebEvaluator failed: %s", exc)
            web_evidence = []

        if not web_evidence:
            logger.info("Pipeline: no web evidence found.")
            return self._no_evidence_response(claim)

        # ── Stage 2: Evidence Engine (embedding + NLI) ────────────────────────
        try:
            evidence_result: EvidenceEngineResult = self._evidence.analyze(
                claim, web_evidence
            )
        except Exception as exc:
            logger.error("EvidenceEngine failed: %s", exc)
            return self._no_evidence_response(claim)

        # Count items that were actually processed (not just skipped stubs).
        usable_count = len(
            [
                e for e in evidence_result.analyzed_evidence
                if e.content.strip() and e.nli_label not in ("SKIPPED",)
            ]
        )

        # ── Stage 3: Credibility Engine ───────────────────────────────────────
        try:
            assessment = self._credibility.assess(evidence_result)
        except Exception as exc:
            logger.error("CredibilityEngine failed: %s", exc)
            return self._no_evidence_response(claim)

        # ── Stage 4: Verdict decision ─────────────────────────────────────────
        verdict_result = self._verdict.decide(assessment, usable_count)

        logger.info(
            "Pipeline: verdict=%s confidence=%.3f support=%.3f contradiction=%.3f",
            verdict_result.verdict.value,
            verdict_result.confidence,
            verdict_result.support_score,
            verdict_result.contradiction_score,
        )

        # ── Stage 5: Gemini explanation ───────────────────────────────────────
        explanation: str | None = None
        try:
            explanation = await self._gemini.explain(
                claim=claim,
                verdict=verdict_result.verdict.value,
                confidence=verdict_result.confidence,
                assessment=assessment,
                evidence_result=evidence_result,
            )
        except Exception as exc:
            logger.warning("GeminiService raised unexpectedly: %s", exc)
            explanation = None

        if not explanation:
            explanation = _build_fallback_explanation(
                verdict=verdict_result.verdict,
                reason=verdict_result.reason,
                supporting_count=len(evidence_result.supporting_evidence),
                contradicting_count=len(evidence_result.contradicting_evidence),
            )

        # ── Stage 6: Build response ───────────────────────────────────────────
        evidence_items: List[EvidenceItem] = [
            _analysis_to_evidence_item(a)
            for a in evidence_result.analyzed_evidence
        ]

        # Build source references from evidence (deduplicated by URL).
        seen_source_urls: set[str] = set()
        sources: List[SourceReference] = []
        for a in evidence_result.analyzed_evidence:
            if a.url and a.url not in seen_source_urls:
                seen_source_urls.add(a.url)
                sources.append(
                    SourceReference(
                        url=a.url,
                        title=a.title or None,
                        domain=a.source_name or None,
                        published_at=a.published_at,
                    )
                )

        credibility_out = CredibilityResult(
            overall_score=round(assessment.overall_score, 6),
            component_scores={
                "support_score": round(assessment.support_score, 6),
                "contradiction_score": round(assessment.contradiction_score, 6),
                "evidence_quality_score": round(assessment.evidence_quality_score, 6),
                "source_quality_score": round(assessment.source_quality_score, 6),
                "confidence": round(assessment.confidence, 6),
            },
        )

        return VerifyResponse(
            claim=claim,
            verdict=verdict_result.verdict,
            confidence=round(verdict_result.confidence, 6),
            explanation=explanation,
            support_score=round(verdict_result.support_score, 6),
            contradiction_score=round(verdict_result.contradiction_score, 6),
            evidence=evidence_items,
            sources=sources,
            credibility=credibility_out,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _no_evidence_response(claim: str) -> VerifyResponse:
        """Return a graceful UNVERIFIED response when evidence is unavailable."""
        return VerifyResponse(
            claim=claim,
            verdict=VerificationStatus.UNVERIFIED,
            confidence=0.0,
            explanation=(
                "Insufficient reliable evidence was found to verify or "
                "contradict this claim."
            ),
            support_score=0.0,
            contradiction_score=0.0,
            evidence=[],
            sources=[],
            credibility=None,
        )


# ---------------------------------------------------------------------------
# Singleton factory — creates a shared pipeline instance per process
# ---------------------------------------------------------------------------

_pipeline_instance: VerificationPipeline | None = None


def get_pipeline(cfg: Settings = _default_settings) -> VerificationPipeline:
    """
    Return the process-global :class:`VerificationPipeline`, creating it on
    first call.

    All heavy services (embedding model, NLI model) are initialised lazily
    inside their respective classes, so the first *request* pays the load cost,
    not import time.

    Parameters
    ----------
    cfg : Settings
        Settings to use when constructing the pipeline.  Defaults to the
        process-wide singleton.  Pass a custom Settings object in tests.
    """
    global _pipeline_instance  # noqa: PLW0603
    if _pipeline_instance is None:
        _pipeline_instance = _build_pipeline(cfg)
    return _pipeline_instance


def _build_pipeline(cfg: Settings) -> VerificationPipeline:
    """Construct a fresh :class:`VerificationPipeline` from *cfg*."""
    web_evaluator = WebEvaluatorService(
        serper_api_key=cfg.SERPER_API_KEY,
        max_results=cfg.WEB_SEARCH_MAX_RESULTS,
    )
    embedding_service = EmbeddingService(model_name=cfg.EMBEDDING_MODEL_NAME)
    transformer_service = TransformerService(model_name=cfg.TRANSFORMER_MODEL_NAME)
    evidence_engine = EvidenceEngine(
        embedding_service=embedding_service,
        transformer_service=transformer_service,
    )
    credibility_engine = CredibilityEngine()
    verdict_engine = VerdictEngine(settings=cfg)
    gemini_service = GeminiService(
        api_key=cfg.GEMINI_API_KEY,
        model=cfg.GEMINI_MODEL,
        timeout=cfg.GEMINI_TIMEOUT,
    )
    return VerificationPipeline(
        web_evaluator=web_evaluator,
        evidence_engine=evidence_engine,
        credibility_engine=credibility_engine,
        verdict_engine=verdict_engine,
        gemini_service=gemini_service,
        settings=cfg,
    )
