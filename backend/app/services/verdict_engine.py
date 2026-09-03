"""
VeritasAI V1 — Verdict Engine
==============================
Pure-function decision layer that maps a :class:`CredibilityAssessment` and
evidence counts to one of three final verdicts:

    VERIFIED | UNVERIFIED | CONTRADICTED

This module has **no** responsibility for:
* Fetching evidence (WebEvaluatorService)
* Embedding or NLI (EmbeddingService / TransformerService)
* Combining evidence signals (EvidenceEngine)
* Scoring source credibility (CredibilityEngine)
* Generating explanations (GeminiService)

Threshold rationale
-------------------
All thresholds live in :mod:`app.config` (Settings) and are documented there.
Here we only document the *logic* that uses them.

VERIFIED
    • support_score  ≥  VERDICT_SUPPORT_THRESHOLD        (meaningful support)
    • confidence     ≥  VERDICT_CONFIDENCE_MIN            (enough sources)
    • contradiction_score  <  VERDICT_CONTRADICTION_THRESHOLD   (no strong counter)
    • OR support_score ≥ contradiction_score × VERDICT_DOMINANCE_FACTOR
      (support clearly dominates even if absolute threshold not met)

CONTRADICTED
    • contradiction_score  ≥  VERDICT_CONTRADICTION_THRESHOLD   (meaningful counter)
    • confidence           ≥  VERDICT_CONFIDENCE_MIN
    • support does NOT dominate (support_score < contradiction_score × dominance)

UNVERIFIED (default)
    Everything else: insufficient evidence, low confidence, mixed signals.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.models.schemas import VerificationStatus

if TYPE_CHECKING:
    from app.config import Settings
    from app.services.credibility_engine import CredibilityAssessment

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VerdictResult:
    """
    Output of the Verdict Engine.

    Attributes
    ----------
    verdict : VerificationStatus
        One of VERIFIED, UNVERIFIED, or CONTRADICTED.
    confidence : float
        Passed through from the CredibilityAssessment.
    support_score : float
        Passed through from the CredibilityAssessment.
    contradiction_score : float
        Passed through from the CredibilityAssessment.
    reason : str
        Short internal reasoning string (for logging and fallback explanations).
    """

    verdict: VerificationStatus
    confidence: float
    support_score: float
    contradiction_score: float
    reason: str


class VerdictEngine:
    """
    Maps a :class:`~app.services.credibility_engine.CredibilityAssessment`
    to a final :class:`VerificationStatus` verdict.

    Parameters
    ----------
    settings : Settings
        Application settings object supplying all threshold constants.

    Examples
    --------
    >>> engine = VerdictEngine(settings)
    >>> result = engine.decide(assessment, usable_evidence_count=3)
    >>> result.verdict
    <VerificationStatus.VERIFIED: 'VERIFIED'>
    """

    def __init__(self, settings: "Settings") -> None:
        self._support_threshold = settings.VERDICT_SUPPORT_THRESHOLD
        self._contradiction_threshold = settings.VERDICT_CONTRADICTION_THRESHOLD
        self._confidence_min = settings.VERDICT_CONFIDENCE_MIN
        self._min_usable = settings.VERDICT_MIN_USABLE_EVIDENCE
        self._dominance = settings.VERDICT_DOMINANCE_FACTOR

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decide(
        self,
        assessment: "CredibilityAssessment",
        usable_evidence_count: int,
    ) -> VerdictResult:
        """
        Produce a deterministic verdict from the credibility assessment.

        Parameters
        ----------
        assessment : CredibilityAssessment
            Full output from :class:`~app.services.credibility_engine.CredibilityEngine`.
        usable_evidence_count : int
            Number of evidence items that had non-empty content and were
            processed by the NLI stage (used to gate the minimum-evidence guard).

        Returns
        -------
        VerdictResult
            Always returns a result; never raises.
        """
        sup = assessment.support_score
        con = assessment.contradiction_score
        conf = assessment.confidence

        logger.debug(
            "VerdictEngine.decide: support=%.3f contradiction=%.3f "
            "confidence=%.3f usable=%d",
            sup, con, conf, usable_evidence_count,
        )

        # ── Guard: insufficient evidence ──────────────────────────────────────
        if usable_evidence_count < self._min_usable:
            return VerdictResult(
                verdict=VerificationStatus.UNVERIFIED,
                confidence=conf,
                support_score=sup,
                contradiction_score=con,
                reason=(
                    f"Insufficient usable evidence "
                    f"(found {usable_evidence_count}, need ≥{self._min_usable})."
                ),
            )

        # ── Guard: confidence too low ─────────────────────────────────────────
        if conf < self._confidence_min:
            return VerdictResult(
                verdict=VerificationStatus.UNVERIFIED,
                confidence=conf,
                support_score=sup,
                contradiction_score=con,
                reason=(
                    f"Confidence too low ({conf:.2f} < {self._confidence_min:.2f}); "
                    "insufficient source diversity."
                ),
            )

        # ── VERIFIED check ────────────────────────────────────────────────────
        # Primary path: support meets threshold AND no strong contradiction.
        support_strong = sup >= self._support_threshold
        contradiction_weak = con < self._contradiction_threshold
        support_dominates = (
            con > 0 and sup >= con * self._dominance
        ) or (con == 0 and sup >= self._support_threshold)

        if support_strong and contradiction_weak:
            return VerdictResult(
                verdict=VerificationStatus.VERIFIED,
                confidence=conf,
                support_score=sup,
                contradiction_score=con,
                reason=(
                    f"Strong support (score={sup:.2f}) with minimal contradiction "
                    f"(score={con:.2f}); confidence={conf:.2f}."
                ),
            )

        # Secondary path: support strongly dominates even with some contradiction.
        if support_dominates and support_strong:
            return VerdictResult(
                verdict=VerificationStatus.VERIFIED,
                confidence=conf,
                support_score=sup,
                contradiction_score=con,
                reason=(
                    f"Support (score={sup:.2f}) dominates contradiction "
                    f"(score={con:.2f}) by ≥{self._dominance}× factor; "
                    f"confidence={conf:.2f}."
                ),
            )

        # ── CONTRADICTED check ────────────────────────────────────────────────
        # Contradiction meets threshold AND support does not dominate it.
        contradiction_strong = con >= self._contradiction_threshold
        contradiction_dominates = (
            sup > 0 and con >= sup * self._dominance
        ) or (sup == 0 and con >= self._contradiction_threshold)

        if contradiction_strong and not support_dominates:
            return VerdictResult(
                verdict=VerificationStatus.CONTRADICTED,
                confidence=conf,
                support_score=sup,
                contradiction_score=con,
                reason=(
                    f"Strong contradiction (score={con:.2f}) with insufficient "
                    f"counter-support (score={sup:.2f}); confidence={conf:.2f}."
                ),
            )

        # Also catch: contradiction dominates even if absolute threshold not met.
        if contradiction_dominates and con >= self._contradiction_threshold * 0.7:
            return VerdictResult(
                verdict=VerificationStatus.CONTRADICTED,
                confidence=conf,
                support_score=sup,
                contradiction_score=con,
                reason=(
                    f"Contradiction (score={con:.2f}) dominates support "
                    f"(score={sup:.2f}) by ≥{self._dominance}× factor."
                ),
            )

        # ── Default: UNVERIFIED ───────────────────────────────────────────────
        return VerdictResult(
            verdict=VerificationStatus.UNVERIFIED,
            confidence=conf,
            support_score=sup,
            contradiction_score=con,
            reason=(
                f"Mixed or weak signals: support={sup:.2f}, "
                f"contradiction={con:.2f}, confidence={conf:.2f}."
            ),
        )
