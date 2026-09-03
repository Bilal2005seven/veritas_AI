"""
VeritasAI V1 — API Routes
"""

import logging

from fastapi import APIRouter, HTTPException, status

from app.models.schemas import VerifyRequest, VerifyResponse, VerificationStatus
from app.services.pipeline import get_pipeline

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/verify",
    response_model=VerifyResponse,
    status_code=status.HTTP_200_OK,
    summary="Verify a claim",
    tags=["Verification"],
)
async def verify_claim(request: VerifyRequest) -> VerifyResponse:
    """
    Accept a free-text claim and return a structured verification result.

    Pipeline (executed inside VerificationPipeline.run):
      1. WebEvaluatorService   — live web evidence retrieval
      2. EvidenceEngine        — MiniLM semantic relevance + NLI stance
      3. CredibilityEngine     — source-weighted credibility assessment
      4. VerdictEngine         — deterministic VERIFIED/UNVERIFIED/CONTRADICTED
      5. GeminiService         — human-readable explanation (graceful fallback)

    Errors at any stage are caught and return a graceful UNVERIFIED response;
    the endpoint itself only raises HTTP 422 for invalid input (handled by
    Pydantic) or HTTP 500 for truly unexpected top-level failures.
    """
    claim = request.claim.strip()

    if not claim:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Claim must not be empty or whitespace-only.",
        )

    try:
        pipeline = get_pipeline()
        result = await pipeline.run(claim)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected pipeline error for claim=%r: %s", claim[:80], exc)
        # Return a safe fallback rather than a 500 so the client always gets
        # a structured VerifyResponse.
        return VerifyResponse(
            claim=claim,
            verdict=VerificationStatus.UNVERIFIED,
            confidence=0.0,
            explanation=(
                "An unexpected error occurred during verification. "
                "Please try again later."
            ),
            support_score=0.0,
            contradiction_score=0.0,
            evidence=[],
            sources=[],
            credibility=None,
        )

    return result
