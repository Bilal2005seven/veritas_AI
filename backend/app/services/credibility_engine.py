"""
VeritasAI V1 — Credibility Engine
=================================

Converts a structured EvidenceEngineResult into a multi-signal
credibility assessment.

Responsibilities
----------------
* Source credibility — weight evidence by source/domain reliability.
* NLI stance — ENTAILMENT contributes support, CONTRADICTION contributes
  contradiction, NEUTRAL contributes neither.
* Evidence relevance — use relevance scores produced by EvidenceEngine.
* Source diversity — unique domains increase confidence.
* Evidence quantity — more independent sources increase confidence.

Important
---------
This engine DOES NOT produce the final VERIFIED / UNVERIFIED /
CONTRADICTED verdict.

The VerdictEngine is responsible for the final verdict.

Design
------
The stance scores are based on the balance of actual stance-bearing
evidence rather than dividing support by the total number of evidence
items.

This is important because a neutral article should not mathematically
cancel a genuinely supporting article.

However, stance strength is still multiplied by overall evidence quality,
so weak / low-relevance evidence cannot automatically become strong
support merely because there is no contradiction.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

from app.models.transformer import (
    LABEL_CONTRADICTION,
    LABEL_ENTAILMENT,
    LABEL_NEUTRAL,
)
from app.services.evidence_engine import EvidenceAnalysis, EvidenceEngineResult


logger = logging.getLogger(__name__)


# ============================================================================
# SOURCE CREDIBILITY REGISTRY
# ============================================================================

SOURCE_WEIGHT_REGISTRY: dict[str, float] = {
    # ------------------------------------------------------------------
    # Tier A — international wire services / major public broadcasters
    # ------------------------------------------------------------------
    "reuters.com": 0.92,
    "apnews.com": 0.92,
    "bbc.com": 0.90,
    "bbc.co.uk": 0.90,
    "npr.org": 0.90,
    "pbs.org": 0.88,
    "dw.com": 0.88,
    "france24.com": 0.87,
    "aljazeera.com": 0.86,

    # ------------------------------------------------------------------
    # Major Indian sources
    # ------------------------------------------------------------------
    "thehindu.com": 0.85,
    "ndtv.com": 0.82,
    "hindustantimes.com": 0.80,
    "timesofindia.com": 0.80,
    "theprint.in": 0.78,
    "scroll.in": 0.76,

    # ------------------------------------------------------------------
    # Tier B — established newspapers / publications
    # ------------------------------------------------------------------
    "nytimes.com": 0.88,
    "washingtonpost.com": 0.88,
    "theguardian.com": 0.87,
    "economist.com": 0.87,
    "ft.com": 0.87,
    "bloomberg.com": 0.86,
    "wsj.com": 0.85,
    "latimes.com": 0.84,
    "usatoday.com": 0.82,
    "telegraph.co.uk": 0.82,
    "independent.co.uk": 0.81,
    "thetimes.co.uk": 0.82,

    # ------------------------------------------------------------------
    # Tier C — established digital media
    # ------------------------------------------------------------------
    "politico.com": 0.78,
    "theatlantic.com": 0.78,
    "vox.com": 0.74,
    "axios.com": 0.76,
    "businessinsider.com": 0.72,
    "huffpost.com": 0.70,
    "slate.com": 0.70,
    "salon.com": 0.68,

    # ------------------------------------------------------------------
    # Fact-check / reference sources
    # ------------------------------------------------------------------
    "wikipedia.org": 0.60,
    "snopes.com": 0.82,
    "factcheck.org": 0.82,
    "politifact.com": 0.82,
    "altfacts.in": 0.75,

    # ------------------------------------------------------------------
    # Government / institutional / scientific sources
    # ------------------------------------------------------------------
    "who.int": 0.90,
    "cdc.gov": 0.90,
    "nih.gov": 0.90,
    "nature.com": 0.92,
    "sciencemag.org": 0.92,
    "thelancet.com": 0.92,
}


# Unknown sources are not considered false.
# They receive a conservative neutral weight.
DEFAULT_SOURCE_WEIGHT: float = 0.50


# Number of unique domains after which the confidence curve starts
# approaching saturation.
CONFIDENCE_SATURATION_SOURCES: int = 5


# Overall credibility blend.
#
# These weights intentionally remain the same as the previous V1 engine
# so that the surrounding application behaviour is not unnecessarily
# changed.
OVERALL_WEIGHT_SUPPORT: float = 0.45
OVERALL_WEIGHT_ANTI_CONTRADICTION: float = 0.30
OVERALL_WEIGHT_QUALITY: float = 0.15
OVERALL_WEIGHT_SOURCE: float = 0.10


# NLI label → magnitude.
#
# The actual direction is determined by whether the label is
# ENTAILMENT or CONTRADICTION.
NLI_STANCE: dict[str, float] = {
    LABEL_ENTAILMENT: 1.0,
    LABEL_CONTRADICTION: 1.0,
    LABEL_NEUTRAL: 0.0,
    "SKIPPED": 0.0,
}


# ============================================================================
# OUTPUT SCHEMA
# ============================================================================

@dataclass
class CredibilityAssessment:
    """
    Multi-signal credibility assessment produced by CredibilityEngine.

    All scores are guaranteed to be in [0.0, 1.0].
    """

    overall_score: float = 0.5

    support_score: float = 0.0

    contradiction_score: float = 0.0

    evidence_quality_score: float = 0.0

    source_quality_score: float = 0.5

    unique_source_count: int = 0

    supporting_source_count: int = 0

    contradicting_source_count: int = 0

    confidence: float = 0.0

    reasoning: str = ""


# ============================================================================
# DOMAIN UTILITIES
# ============================================================================

def _extract_domain(url: str) -> str:
    """
    Extract the hostname/domain from a URL.

    Examples
    --------
    https://www.reuters.com/article/...
        -> reuters.com

    https://news.bbc.co.uk/story
        -> news.bbc.co.uk

    Returns an empty string if parsing fails.
    """

    if not isinstance(url, str) or not url.strip():
        return ""

    try:
        parsed = urlparse(url.strip())

        domain = parsed.netloc.lower()

        # Remove username/password if malformed URLs contain them.
        if "@" in domain:
            domain = domain.rsplit("@", 1)[-1]

        # Remove port.
        domain = domain.split(":", 1)[0]

        # Remove www prefix only.
        if domain.startswith("www."):
            domain = domain[4:]

        return domain

    except Exception:
        return ""


def _source_weight(
    domain: str,
    registry: dict[str, float],
) -> float:
    """
    Return the credibility weight for a domain.

    Uses:

        exact match
            ↓
        longest suffix match
            ↓
        DEFAULT_SOURCE_WEIGHT
    """

    if not domain:
        return DEFAULT_SOURCE_WEIGHT

    # Exact match.
    if domain in registry:
        return _clamp(registry[domain])

    # Longest suffix match.
    #
    # Example:
    #
    # news.bbc.co.uk
    #
    # checks:
    #   bbc.co.uk
    #   co.uk
    #   uk
    #
    # and therefore resolves to bbc.co.uk.
    parts = domain.split(".")

    for i in range(1, len(parts)):
        suffix = ".".join(parts[i:])

        if suffix in registry:
            return _clamp(registry[suffix])

    return DEFAULT_SOURCE_WEIGHT


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """
    Safely clamp a numeric value to [low, high].
    """

    try:
        value = float(value)
    except (TypeError, ValueError):
        return low

    if not math.isfinite(value):
        return low

    return max(low, min(high, value))


# ============================================================================
# ENGINE
# ============================================================================

class CredibilityEngine:
    """
    Converts EvidenceEngineResult into CredibilityAssessment.

    Parameters
    ----------
    extra_source_weights:
        Optional domain → credibility weight overrides.

    min_usable_relevance:
        Minimum relevance required for an evidence item to participate
        in stance scoring.

        Items below this threshold can still contribute to source
        diversity and evidence-quality calculations.

    confidence_saturation:
        Number of unique source domains at which the confidence curve
        begins to saturate.
    """

    def __init__(
        self,
        extra_source_weights: Optional[dict[str, float]] = None,
        min_usable_relevance: float = 0.20,
        confidence_saturation: int = CONFIDENCE_SATURATION_SOURCES,
    ) -> None:

        # Preserve the built-in registry.
        self._registry: dict[str, float] = {
            **SOURCE_WEIGHT_REGISTRY
        }

        # Allow deployment/test-specific overrides.
        if extra_source_weights:
            for domain, weight in extra_source_weights.items():

                if not isinstance(domain, str):
                    continue

                clean_domain = domain.strip().lower()

                if not clean_domain:
                    continue

                self._registry[clean_domain] = _clamp(weight)

        # Defensive configuration handling.
        self._min_usable_relevance = _clamp(
            min_usable_relevance
        )

        # Avoid division by zero in confidence calculation.
        try:
            saturation = int(confidence_saturation)
        except (TypeError, ValueError):
            saturation = CONFIDENCE_SATURATION_SOURCES

        self._confidence_saturation = max(1, saturation)

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def assess(
        self,
        result: EvidenceEngineResult,
    ) -> CredibilityAssessment:
        """
        Produce a credibility assessment.

        This method is intentionally defensive and should not crash the
        verification pipeline because of malformed evidence.
        """

        # ------------------------------------------------------------------
        # 1. Validate evidence container
        # ------------------------------------------------------------------

        if result is None:
            logger.warning(
                "CredibilityEngine: received None result."
            )

            return CredibilityAssessment(
                overall_score=0.5,
                confidence=0.0,
                reasoning=(
                    "No evidence result was available; "
                    "claim could not be assessed."
                ),
            )

        analyzed_evidence = getattr(
            result,
            "analyzed_evidence",
            None,
        )

        if not analyzed_evidence:
            logger.info(
                "CredibilityEngine: empty evidence."
            )

            return CredibilityAssessment(
                overall_score=0.5,
                confidence=0.0,
                reasoning=(
                    "No usable evidence found; "
                    "claim could not be assessed."
                ),
            )

        # ------------------------------------------------------------------
        # 2. Filter usable evidence
        # ------------------------------------------------------------------

        usable: list[EvidenceAnalysis] = []

        for item in analyzed_evidence:

            if item is None:
                continue

            # Empty fetched content cannot provide useful evidence.
            content = getattr(item, "content", "") or ""

            if not isinstance(content, str):
                content = str(content)

            if not content.strip():
                continue

            usable.append(item)

        if not usable:
            logger.info(
                "CredibilityEngine: all evidence items were empty."
            )

            return CredibilityAssessment(
                overall_score=0.5,
                confidence=0.0,
                reasoning=(
                    "All evidence items were empty or unfetchable; "
                    "claim could not be assessed."
                ),
            )

        # ------------------------------------------------------------------
        # 3. Initialize scoring accumulators
        # ------------------------------------------------------------------

        support_sum: float = 0.0

        contradiction_sum: float = 0.0

        # Total quality contribution:
        #
        #     relevance × source_weight
        #
        quality_weight_sum: float = 0.0

        # Denominator for weighted evidence quality.
        #
        # We use source credibility weights here so that a highly
        # credible source contributes more strongly to the quality average.
        quality_relevance_sum: float = 0.0

        # Unique source domains.
        all_domains: set[str] = set()

        support_domains: set[str] = set()

        contradiction_domains: set[str] = set()

        # ------------------------------------------------------------------
        # 4. Score each evidence item
        # ------------------------------------------------------------------

        for item in usable:

            # --------------------------------------------------------------
            # Domain
            # --------------------------------------------------------------

            url = getattr(item, "url", "") or ""

            domain = _extract_domain(url)

            source_weight = _source_weight(
                domain,
                self._registry,
            )

            # --------------------------------------------------------------
            # Relevance
            # --------------------------------------------------------------

            raw_relevance = getattr(
                item,
                "relevance_score",
                0.0,
            )

            relevance = _clamp(raw_relevance)

            # --------------------------------------------------------------
            # Track source diversity
            # --------------------------------------------------------------

            if domain:
                all_domains.add(domain)

            # --------------------------------------------------------------
            # Evidence quality
            #
            # Every usable evidence item contributes to quality,
            # regardless of NLI stance.
            # --------------------------------------------------------------

            quality_contribution = (
                relevance * source_weight
            )

            quality_weight_sum += quality_contribution

            quality_relevance_sum += source_weight

            # --------------------------------------------------------------
            # Low-relevance evidence
            #
            # It remains useful for diversity/quality calculations,
            # but should not be treated as meaningful stance evidence.
            # --------------------------------------------------------------

            if relevance < self._min_usable_relevance:
                continue

            # --------------------------------------------------------------
            # Stance contribution
            # --------------------------------------------------------------

            item_contribution = (
                relevance * source_weight
            )

            nli_label = getattr(
                item,
                "nli_label",
                LABEL_NEUTRAL,
            )

            # ENTAILMENT → support.
            if nli_label == LABEL_ENTAILMENT:

                support_sum += item_contribution

                if domain:
                    support_domains.add(domain)

            # CONTRADICTION → contradiction.
            elif nli_label == LABEL_CONTRADICTION:

                contradiction_sum += item_contribution

                if domain:
                    contradiction_domains.add(domain)

            # NEUTRAL / SKIPPED → no stance contribution.
            else:
                pass

        # ------------------------------------------------------------------
        # 5. Evidence quality
        # ------------------------------------------------------------------

        if quality_relevance_sum > 0.0:

            evidence_quality_score = (
                quality_weight_sum
                / quality_relevance_sum
            )

        else:
            evidence_quality_score = 0.0

        evidence_quality_score = _clamp(
            evidence_quality_score
        )

        # ------------------------------------------------------------------
        # 6. STANCE NORMALIZATION
        # ------------------------------------------------------------------
        #
        # THIS IS THE IMPORTANT FIX.
        #
        # Old implementation:
        #
        #     support_sum / number_of_all_items
        #
        # Problem:
        #
        #     3 supporting + 7 neutral
        #
        # became:
        #
        #     support / 10
        #
        # Neutral evidence therefore artificially destroyed support.
        #
        # New implementation:
        #
        #     stance_total = support_sum + contradiction_sum
        #
        #     support_balance =
        #         support_sum / stance_total
        #
        #     contradiction_balance =
        #         contradiction_sum / stance_total
        #
        # This measures the actual balance of evidence that has a
        # meaningful NLI stance.
        #
        # Then we multiply by evidence_quality_score.
        #
        # This prevents weak evidence from automatically becoming
        # strong merely because contradiction is absent.
        # ------------------------------------------------------------------

        stance_total = (
            support_sum
            + contradiction_sum
        )

        if stance_total > 0.0:

            support_balance = (
                support_sum / stance_total
            )

            contradiction_balance = (
                contradiction_sum / stance_total
            )

            support_score = (
                support_balance
                * evidence_quality_score
            )

            contradiction_score = (
                contradiction_balance
                * evidence_quality_score
            )

        else:

            # No meaningful stance evidence.
            support_score = 0.0
            contradiction_score = 0.0

        support_score = _clamp(
            support_score
        )

        contradiction_score = _clamp(
            contradiction_score
        )

        # ------------------------------------------------------------------
        # 7. Source quality
        # ------------------------------------------------------------------

        n_unique = len(all_domains)

        if n_unique > 0:

            source_quality_score = sum(
                _source_weight(
                    domain,
                    self._registry,
                )
                for domain in all_domains
            ) / n_unique

        else:

            source_quality_score = DEFAULT_SOURCE_WEIGHT

        source_quality_score = _clamp(
            source_quality_score
        )

        # ------------------------------------------------------------------
        # 8. Confidence
        # ------------------------------------------------------------------
        #
        # Confidence depends on source diversity.
        #
        # Formula:
        #
        #     confidence = 1 - exp(-n_unique / saturation)
        #
        # This means:
        #
        # 1 source  → lower confidence
        # 3 sources → moderate/high
        # 5 sources → strong
        # many      → approaches 1
        # ------------------------------------------------------------------

        confidence = (
            1.0
            - math.exp(
                -n_unique
                / self._confidence_saturation
            )
        )

        confidence = _clamp(
            confidence
        )

        # ------------------------------------------------------------------
        # 9. Overall score
        # ------------------------------------------------------------------
        #
        # overall =
        #
        #     support contribution
        #   + anti-contradiction contribution
        #   + evidence quality
        #   + source quality
        #
        # We preserve the existing V1 weights.
        # ------------------------------------------------------------------

        anti_contradiction = (
            1.0 - contradiction_score
        )

        overall_score = (
            OVERALL_WEIGHT_SUPPORT
            * support_score

            + OVERALL_WEIGHT_ANTI_CONTRADICTION
            * anti_contradiction

            + OVERALL_WEIGHT_QUALITY
            * evidence_quality_score

            + OVERALL_WEIGHT_SOURCE
            * source_quality_score
        )

        overall_score = _clamp(
            overall_score
        )

        # ------------------------------------------------------------------
        # 10. Human-readable reasoning
        # ------------------------------------------------------------------

        reasoning = self._build_reasoning(
            n_usable=len(usable),
            unique_sources=n_unique,
            support_domains=len(support_domains),
            contradiction_domains=len(
                contradiction_domains
            ),
            support_score=support_score,
            contradiction_score=contradiction_score,
            confidence=confidence,
        )

        # ------------------------------------------------------------------
        # 11. Final assessment
        # ------------------------------------------------------------------

        assessment = CredibilityAssessment(
            overall_score=round(
                overall_score,
                6,
            ),

            support_score=round(
                support_score,
                6,
            ),

            contradiction_score=round(
                contradiction_score,
                6,
            ),

            evidence_quality_score=round(
                evidence_quality_score,
                6,
            ),

            source_quality_score=round(
                source_quality_score,
                6,
            ),

            unique_source_count=n_unique,

            supporting_source_count=len(
                support_domains
            ),

            contradicting_source_count=len(
                contradiction_domains
            ),

            confidence=round(
                confidence,
                6,
            ),

            reasoning=reasoning,
        )

        logger.info(
            (
                "CredibilityEngine: "
                "overall=%.3f "
                "support=%.3f "
                "contra=%.3f "
                "quality=%.3f "
                "src_quality=%.3f "
                "confidence=%.3f "
                "unique_sources=%d"
            ),
            assessment.overall_score,
            assessment.support_score,
            assessment.contradiction_score,
            assessment.evidence_quality_score,
            assessment.source_quality_score,
            assessment.confidence,
            assessment.unique_source_count,
        )

        return assessment

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    @staticmethod
    def _build_reasoning(
        *,
        n_usable: int,
        unique_sources: int,
        support_domains: int,
        contradiction_domains: int,
        support_score: float,
        contradiction_score: float,
        confidence: float,
    ) -> str:
        """
        Build a concise explanation suitable for the API/UI.
        """

        if n_usable == 0:

            return (
                "No usable evidence found; "
                "claim could not be assessed."
            )

        # --------------------------------------------------------------
        # Confidence description
        # --------------------------------------------------------------

        if confidence >= 0.70:

            conf_desc = "high"

        elif confidence >= 0.40:

            conf_desc = "moderate"

        else:

            conf_desc = "low"

        # --------------------------------------------------------------
        # No stance evidence
        # --------------------------------------------------------------

        if (
            support_domains == 0
            and contradiction_domains == 0
        ):

            return (
                f"Found {n_usable} evidence item(s) "
                f"from {unique_sources} unique source(s), "
                f"but none provided a clear stance on the claim "
                f"(all were neutral or below the relevance threshold). "
                f"Confidence is {conf_desc}."
            )

        # --------------------------------------------------------------
        # Build stance summary
        # --------------------------------------------------------------

        parts: list[str] = []

        if support_domains > 0:

            parts.append(
                f"{support_domains} source(s) support the claim "
                f"(support score: {support_score:.2f})"
            )

        if contradiction_domains > 0:

            parts.append(
                f"{contradiction_domains} source(s) contradict it "
                f"(contradiction score: "
                f"{contradiction_score:.2f})"
            )

        stance_summary = "; ".join(parts)

        return (
            f"Assessed {n_usable} evidence item(s) "
            f"from {unique_sources} unique source(s): "
            f"{stance_summary}. "
            f"Confidence is {conf_desc} based on "
            f"source diversity."
        )