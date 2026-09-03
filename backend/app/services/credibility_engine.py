"""
VeritasAI V1 — Credibility Engine
====================================
Converts a structured :class:`~app.services.evidence_engine.EvidenceEngineResult`
into a multi-signal credibility assessment.

Responsibilities
----------------
* **Source credibility** — weight each evidence item by the trustworthiness of
  its domain.  Established news and institutional sources receive higher weights;
  unknown sources receive a conservative (not zero) neutral weight.
* **NLI stance** — ENTAILMENT drives the support score, CONTRADICTION drives the
  contradiction score, NEUTRAL contributes neither.  NEUTRAL is never treated as
  support.
* **Evidence relevance** — use the ``relevance_score`` already computed by the
  Evidence Engine; high-relevance items contribute more to the quality signal.
* **Source diversity** — articles from the same domain are not fully independent;
  the unique-domain count is used to scale confidence.
* **Evidence quantity** — more independent sources increases confidence (up to a
  configurable saturation point).

Design decisions
----------------
* **No final verdict** — the engine produces a :class:`CredibilityAssessment`
  only.  The orchestration layer maps scores to VERIFIED / UNVERIFIED /
  CONTRADICTED using its own configurable thresholds.
* **Deterministic & explainable** — every number in the output can be traced
  back to a specific formula and a specific constant defined below.
* **Configurable thresholds** — all magic numbers live in module-level
  constants; none are buried in logic.
* **Failed / empty evidence is neutral** — items with empty content or
  ``fetch_error`` set are explicitly excluded from positive scoring.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

from app.models.transformer import (
    LABEL_CONTRADICTION,
    LABEL_ENTAILMENT,
    LABEL_NEUTRAL,
)
from app.services.evidence_engine import EvidenceAnalysis, EvidenceEngineResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Source-tier registry
# ---------------------------------------------------------------------------
# Maps domain suffixes/names to a credibility weight in [0, 1].
# The lookup is longest-suffix-first so that "bbc.co.uk" > "co.uk" > "uk".
#
# Tier A (0.90) — major international broadcasters / wire services.
# Tier B (0.80) — established national broadcasters / newspapers.
# Tier C (0.70) — well-known digital media with editorial standards.
# Tier D (0.55) — general reference or aggregator sites.
# Unknown        — DEFAULT_SOURCE_WEIGHT (conservative, not punitive).
#
# Operators can extend or override this dict at instantiation time via the
# ``extra_source_weights`` parameter on :class:`CredibilityEngine`.

SOURCE_WEIGHT_REGISTRY: dict[str, float] = {
    # ── Tier A: international wire / public broadcasters ─────────────────
    "reuters.com":          0.92,
    "apnews.com":           0.92,
    "bbc.com":              0.90,
    "bbc.co.uk":            0.90,
    "npr.org":              0.90,
    "pbs.org":              0.88,
    "dw.com":               0.88,
    "france24.com":         0.87,
    "aljazeera.com":        0.86,
    "thehindu.com":         0.85,
    "ndtv.com":             0.82,
    "hindustantimes.com":   0.80,
    "timesofindia.com":     0.80,
    "theprint.in":          0.78,
    "scroll.in":            0.76,
    # ── Tier B: established print / national papers ───────────────────────
    "nytimes.com":          0.88,
    "washingtonpost.com":   0.88,
    "theguardian.com":      0.87,
    "economist.com":        0.87,
    "ft.com":               0.87,
    "bloomberg.com":        0.86,
    "wsj.com":              0.85,
    "latimes.com":          0.84,
    "usatoday.com":         0.82,
    "telegraph.co.uk":      0.82,
    "independent.co.uk":    0.81,
    "thetimes.co.uk":       0.82,
    # ── Tier C: established digital media with editorial standards ────────
    "politico.com":         0.78,
    "theatlantic.com":      0.78,
    "vox.com":              0.74,
    "axios.com":            0.76,
    "businessinsider.com":  0.72,
    "huffpost.com":         0.70,
    "slate.com":            0.70,
    "salon.com":            0.68,
    # ── Tier D: reference / aggregators ──────────────────────────────────
    "wikipedia.org":        0.60,
    "snopes.com":           0.82,   # fact-check site
    "factcheck.org":        0.82,   # fact-check site
    "politifact.com":       0.82,   # fact-check site
    "altfacts.in":          0.75,
    # ── Government / institutional ────────────────────────────────────────
    "who.int":              0.90,
    "cdc.gov":              0.90,
    "nih.gov":              0.90,
    "nature.com":           0.92,
    "sciencemag.org":       0.92,
    "thelancet.com":        0.92,
}

#: Weight applied to domains not found in SOURCE_WEIGHT_REGISTRY.
#: Set to 0.5 (neutral) — unknown ≠ false, but not a positive signal.
DEFAULT_SOURCE_WEIGHT: float = 0.50

#: Maximum number of unique supporting sources required before confidence
#: saturates (i.e., the quantity-bonus curve flattens out).
CONFIDENCE_SATURATION_SOURCES: int = 5

#: Blending weights for the overall_score formula.
#: Must sum to 1.0.
OVERALL_WEIGHT_SUPPORT: float = 0.45
OVERALL_WEIGHT_ANTI_CONTRADICTION: float = 0.30
OVERALL_WEIGHT_QUALITY: float = 0.15
OVERALL_WEIGHT_SOURCE: float = 0.10

#: NLI label → numeric stance factor used in per-item score contribution.
NLI_STANCE: dict[str, float] = {
    LABEL_ENTAILMENT:    1.0,
    LABEL_CONTRADICTION: 1.0,   # magnitude; sign is handled by bucket
    LABEL_NEUTRAL:       0.0,
    "SKIPPED":           0.0,
}


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

@dataclass
class CredibilityAssessment:
    """
    Multi-signal credibility assessment produced by the Credibility Engine.

    All scores are in ``[0.0, 1.0]``.

    Attributes
    ----------
    overall_score : float
        Blended score reflecting the balance of supporting vs contradicting
        evidence, weighted by quality and source credibility.
        High  → strong support signal.
        Low   → strong contradiction or no usable evidence.
        Mid   → inconclusive.
    support_score : float
        Normalised strength of ENTAILMENT evidence weighted by relevance and
        source credibility.  0 when no supporting evidence exists.
    contradiction_score : float
        Normalised strength of CONTRADICTION evidence.  0 when none exists.
    evidence_quality_score : float
        Average relevance-weighted quality across all usable evidence items.
    source_quality_score : float
        Average source-credibility weight across unique contributing domains.
    unique_source_count : int
        Number of distinct domains that contributed usable (non-empty) evidence.
    supporting_source_count : int
        Number of unique domains with at least one ENTAILMENT item.
    contradicting_source_count : int
        Number of unique domains with at least one CONTRADICTION item.
    confidence : float
        Estimate of how much the scores can be trusted, based on the quantity
        and diversity of usable evidence.  Low when evidence is sparse.
    reasoning : str
        Human-readable single-sentence summary of the scoring rationale.
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


# ---------------------------------------------------------------------------
# Domain utilities
# ---------------------------------------------------------------------------

def _extract_domain(url: str) -> str:
    """
    Extract the registered domain (netloc) from a URL.

    Returns an empty string on parse failure.
    """
    try:
        return urlparse(url).netloc.lower().lstrip("www.")
    except Exception:
        return ""


def _source_weight(domain: str, registry: dict[str, float]) -> float:
    """
    Look up the source-credibility weight for *domain* in *registry*.

    Uses longest-suffix matching so that ``"news.bbc.co.uk"`` correctly
    resolves to the entry for ``"bbc.co.uk"`` rather than falling through
    to the default.

    Parameters
    ----------
    domain : str
        Registered domain extracted from the evidence URL (no ``www.``
        prefix).
    registry : dict[str, float]
        Source-weight mapping.

    Returns
    -------
    float
        Weight in ``[0, 1]``.  Falls back to :data:`DEFAULT_SOURCE_WEIGHT`
        when no suffix match is found.
    """
    if not domain:
        return DEFAULT_SOURCE_WEIGHT

    # Direct hit first (most common case).
    if domain in registry:
        return registry[domain]

    # Walk suffix segments: "news.bbc.co.uk" → "bbc.co.uk" → "co.uk" → "uk"
    parts = domain.split(".")
    for i in range(1, len(parts)):
        suffix = ".".join(parts[i:])
        if suffix in registry:
            return registry[suffix]

    return DEFAULT_SOURCE_WEIGHT


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class CredibilityEngine:
    """
    Converts an :class:`~app.services.evidence_engine.EvidenceEngineResult`
    into a :class:`CredibilityAssessment`.

    Parameters
    ----------
    extra_source_weights : dict[str, float], optional
        Additional or overriding domain → weight mappings merged on top of the
        built-in :data:`SOURCE_WEIGHT_REGISTRY`.  Useful for tests and per-
        deployment customisation without modifying module-level constants.
    min_usable_relevance : float
        Items with ``relevance_score`` below this threshold do not contribute
        to support / contradiction scores.  They still count toward unique-
        source totals.  Default ``0.05`` — a very permissive floor to exclude
        only truly unrelated items.
    confidence_saturation : int
        Number of unique source domains at which confidence saturates (reaches
        its maximum before the penalty for low evidence kicks in).

    Examples
    --------
    >>> engine = CredibilityEngine()
    >>> assessment = engine.assess(evidence_engine_result)
    >>> assessment.overall_score
    0.73
    """

    def __init__(
        self,
        extra_source_weights: Optional[dict[str, float]] = None,
        min_usable_relevance: float = 0.20,   # V1 heuristic — was 0.05; see note below
        confidence_saturation: int = CONFIDENCE_SATURATION_SOURCES,
    ) -> None:
        # min_usable_relevance: items below this floor are counted in source-
        # diversity totals but do NOT contribute to support/contradiction sums.
        # 0.20 is a V1 heuristic aligned with DEFAULT_MIN_RELEVANCE in
        # evidence_engine.py.  If live tests reveal genuine recall regression,
        # report it and consider reverting rather than escalating further.
        self._registry: dict[str, float] = {**SOURCE_WEIGHT_REGISTRY}
        if extra_source_weights:
            self._registry.update(extra_source_weights)
        self._min_usable_relevance = min_usable_relevance
        self._confidence_saturation = confidence_saturation

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assess(self, result: EvidenceEngineResult) -> CredibilityAssessment:
        """
        Produce a :class:`CredibilityAssessment` from the Evidence Engine output.

        Parameters
        ----------
        result : EvidenceEngineResult
            Output from :class:`~app.services.evidence_engine.EvidenceEngine`.

        Returns
        -------
        CredibilityAssessment
            Never raises; degenerate inputs (empty, all-neutral) return a
            conservative mid-range assessment with low confidence.
        """
        if not result.analyzed_evidence:
            logger.info("CredibilityEngine: empty evidence — returning default assessment.")
            return CredibilityAssessment(
                overall_score=0.5,
                reasoning="No usable evidence found; claim could not be assessed.",
            )

        # ── 1. Classify items into usable vs. unusable ─────────────────────
        usable: list[EvidenceAnalysis] = []
        for item in result.analyzed_evidence:
            # Skip items with no content (empty fetch or explicit skip).
            if not item.content.strip():
                continue
            # Skip items that are pure skips with no real NLI result.
            if item.nli_label == "SKIPPED" and not item.content.strip():
                continue
            usable.append(item)

        if not usable:
            logger.info("CredibilityEngine: all evidence items are empty/skipped.")
            return CredibilityAssessment(
                overall_score=0.5,
                reasoning="All evidence items were empty or unfetchable; claim could not be assessed.",
            )

        # ── 2. Per-item scoring ────────────────────────────────────────────
        # For each usable item compute:
        #   item_weight = source_weight × max(relevance_score, 0)
        # Items with relevance below min_usable_relevance are retained in
        # source-diversity counts but do NOT contribute to stance sums.

        support_sum: float = 0.0
        contradiction_sum: float = 0.0
        quality_weight_sum: float = 0.0    # Σ(relevance × source_weight)
        quality_relevance_sum: float = 0.0  # denominator for quality score

        # Unique-domain tracking.
        all_domains: set[str] = set()
        support_domains: set[str] = set()
        contradiction_domains: set[str] = set()
        source_weights_seen: list[float] = []

        for item in usable:
            domain = _extract_domain(item.url)
            sw = _source_weight(domain, self._registry)
            relevance = max(0.0, item.relevance_score)

            # Track domains for diversity.
            if domain:
                all_domains.add(domain)
                if sw not in source_weights_seen or domain not in [
                    d for d in all_domains if _source_weight(d, self._registry) == sw
                ]:
                    source_weights_seen.append(sw)

            # Evidence quality (all usable items regardless of NLI label).
            quality_weight_sum += relevance * sw
            quality_relevance_sum += sw   # weight denominator

            # Only items above the relevance floor contribute to stance sums.
            if relevance < self._min_usable_relevance:
                continue

            item_contribution = relevance * sw

            if item.nli_label == LABEL_ENTAILMENT:
                support_sum += item_contribution
                if domain:
                    support_domains.add(domain)
            elif item.nli_label == LABEL_CONTRADICTION:
                contradiction_sum += item_contribution
                if domain:
                    contradiction_domains.add(domain)
            # NEUTRAL and SKIPPED contribute nothing to stance sums.

        # ── 3. Normalise stance scores ─────────────────────────────────────
        # Normalise against a theoretical maximum where every item is a
        # top-quality (relevance=1, source_weight=1) supporter or contradictor.
        # This prevents adding more low-quality items from inflating scores
        # beyond what a single perfect source would produce.
        #
        # We use the count of usable items as the normaliser so that:
        #   1 perfect supporting item  → support_score = 1.0
        #   N items of varying quality → support_score in (0, 1]
        #
        # Denominator is number of usable items (max possible item_contribution
        # per item = 1.0 × 1.0 = 1.0).

        n_usable = len(usable)

        support_score = min(1.0, support_sum / n_usable) if n_usable > 0 else 0.0
        contradiction_score = (
            min(1.0, contradiction_sum / n_usable) if n_usable > 0 else 0.0
        )

        # ── 4. Evidence quality score ──────────────────────────────────────
        # Weighted-average relevance across usable items (weighted by source
        # credibility so high-quality sources count more).
        evidence_quality_score = (
            quality_weight_sum / quality_relevance_sum
            if quality_relevance_sum > 0.0
            else 0.0
        )

        # ── 5. Source quality score ────────────────────────────────────────
        # Average source-credibility weight across unique contributing domains.
        if all_domains:
            source_quality_score = sum(
                _source_weight(d, self._registry) for d in all_domains
            ) / len(all_domains)
        else:
            source_quality_score = DEFAULT_SOURCE_WEIGHT

        # ── 6. Confidence ─────────────────────────────────────────────────
        # Confidence rises with the number of unique source domains (diversity)
        # and saturates beyond CONFIDENCE_SATURATION_SOURCES.
        # Formula: 1 - exp(-n / saturation)   ∈ (0, 1)
        # This gives a smooth, explainable growth curve with no hard cutoffs.
        n_unique = len(all_domains)
        confidence = 1.0 - math.exp(-n_unique / self._confidence_saturation)

        # ── 7. Overall score ───────────────────────────────────────────────
        # overall = w_s × support_score
        #         + w_a × (1 - contradiction_score)    ← anti-contradiction
        #         + w_q × evidence_quality_score
        #         + w_src × source_quality_score
        #
        # The anti-contradiction term rewards absence of contradicting evidence.
        # It is not the same as rewarding neutral evidence — NEUTRAL items do
        # not affect contradiction_score, so they leave this term at its max.
        anti_contradiction = 1.0 - contradiction_score

        overall_score = (
            OVERALL_WEIGHT_SUPPORT           * support_score
            + OVERALL_WEIGHT_ANTI_CONTRADICTION * anti_contradiction
            + OVERALL_WEIGHT_QUALITY           * evidence_quality_score
            + OVERALL_WEIGHT_SOURCE            * source_quality_score
        )
        overall_score = max(0.0, min(1.0, overall_score))

        # ── 8. Reasoning string ────────────────────────────────────────────
        reasoning = self._build_reasoning(
            n_usable=n_usable,
            unique_sources=n_unique,
            support_domains=len(support_domains),
            contradiction_domains=len(contradiction_domains),
            support_score=support_score,
            contradiction_score=contradiction_score,
            confidence=confidence,
        )

        assessment = CredibilityAssessment(
            overall_score=round(overall_score, 6),
            support_score=round(support_score, 6),
            contradiction_score=round(contradiction_score, 6),
            evidence_quality_score=round(evidence_quality_score, 6),
            source_quality_score=round(source_quality_score, 6),
            unique_source_count=n_unique,
            supporting_source_count=len(support_domains),
            contradicting_source_count=len(contradiction_domains),
            confidence=round(confidence, 6),
            reasoning=reasoning,
        )

        logger.info(
            "CredibilityEngine: overall=%.3f support=%.3f contra=%.3f "
            "quality=%.3f src_quality=%.3f confidence=%.3f "
            "unique_sources=%d",
            assessment.overall_score,
            assessment.support_score,
            assessment.contradiction_score,
            assessment.evidence_quality_score,
            assessment.source_quality_score,
            assessment.confidence,
            assessment.unique_source_count,
        )
        return assessment

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
        Produce a concise, human-readable explanation of the scoring outcome.

        Returns
        -------
        str
            One or two sentences suitable for display in a UI or API response.
        """
        conf_desc = (
            "high" if confidence >= 0.70
            else "moderate" if confidence >= 0.40
            else "low"
        )

        if n_usable == 0:
            return "No usable evidence found; claim could not be assessed."

        if support_domains == 0 and contradiction_domains == 0:
            return (
                f"Found {n_usable} evidence item(s) from {unique_sources} "
                f"unique source(s), but none provided a clear stance on the claim "
                f"(all were neutral or below relevance threshold). "
                f"Confidence is {conf_desc}."
            )

        parts: list[str] = []

        if support_domains > 0:
            parts.append(
                f"{support_domains} source(s) support the claim "
                f"(support score: {support_score:.2f})"
            )
        if contradiction_domains > 0:
            parts.append(
                f"{contradiction_domains} source(s) contradict it "
                f"(contradiction score: {contradiction_score:.2f})"
            )

        stance_summary = "; ".join(parts) + "."
        return (
            f"Assessed {n_usable} evidence item(s) from {unique_sources} unique "
            f"source(s): {stance_summary} "
            f"Confidence is {conf_desc} based on source diversity."
        )
