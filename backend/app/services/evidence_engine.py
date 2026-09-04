"""
VeritasAI V1 — Evidence Engine
================================
Combines web evidence, MiniLM semantic similarity, NLI classification,
and passage-level hybrid retrieval into structured, ranked evidence
analysis for a given claim.

Responsibilities
----------------
1. Accept a raw *claim* string and a list of :class:`~app.services.web_evaluator.WebEvidence`
   objects (already fetched by :class:`~app.services.web_evaluator.WebEvaluatorService`).
2. For every evidence item with usable content:
   a. Chunk the article text into ≈200-word passages
      (:func:`~app.rag.chunker.chunk_text`).
   b. Retrieve the most relevant passage(s) using hybrid BM25 + MiniLM
      retrieval (:func:`~app.rag.hybrid_retriever.hybrid_retrieve`).
   c. Compute a semantic relevance score for the best passage via
      :class:`~app.rag.embeddings.EmbeddingService`.
   d. Run NLI classification on the best passage via
      :class:`~app.models.transformer.TransformerService`.
   e. Combine into an :class:`EvidenceAnalysis` record.
3. Deduplicate by URL (keep first occurrence).
4. Skip items with empty/whitespace content and record them as SKIPPED.
5. Rank by ``relevance_score`` descending.
6. Partition into three buckets using NLI label only (no heuristics):
   - ``supporting_evidence``   — label == ENTAILMENT
   - ``contradicting_evidence`` — label == CONTRADICTION
   - ``neutral_evidence``      — label == NEUTRAL or SKIPPED

Design decisions
----------------
* **Passage-level NLI** — NLI now operates on the single best ≈200-word
  passage selected by hybrid retrieval instead of the full (noisy) article
  text.  This dramatically reduces false NEUTRAL/CONTRADICTION results caused
  by off-topic boilerplate surrounding the relevant sentence.
* **Hybrid retrieval reuses existing services** — BM25 (new: rank-bm25) and
  MiniLM embeddings (existing EmbeddingService) are fused with RRF.  No new
  embedding model is introduced.
* **No final verdict** — the engine emits structured analysis only.
  The Credibility Engine makes the VERIFIED/UNVERIFIED/CONTRADICTED call.
* **NEUTRAL ≠ SUPPORT** — a high embedding similarity with a NEUTRAL NLI
  label means topically related but not logically entailed.  The downstream
  stage must interpret both signals together.
* **Configurable thresholds** — ``min_relevance_threshold`` filters out
  items that are semantically unrelated before running the expensive NLI
  step, saving compute.  Defaults are conservative.
* **NLI errors are non-fatal** — if TransformerService raises (e.g. model
  not yet downloaded), the item is kept with label NEUTRAL and score 0
  so the pipeline does not crash on partial deployments.
* **Sync embedding + NLI** — both services are CPU-bound and synchronous.
  Evidence items are processed sequentially.  Async wrappers are not
  needed until a thread-pool is introduced.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from app.models.schemas import TransformerResult
from app.models.transformer import (
    LABEL_CONTRADICTION,
    LABEL_ENTAILMENT,
    LABEL_NEUTRAL,
    TransformerService,
)
from app.rag.chunker import chunk_text
from app.rag.embeddings import EmbeddingService
from app.rag.hybrid_retriever import hybrid_retrieve
from app.services.web_evaluator import WebEvidence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configurable defaults
# ---------------------------------------------------------------------------

#: Minimum cosine-similarity score [0, 1] required before running NLI.
#: Items below this threshold are kept but tagged NEUTRAL with score 0.0.
#:
#: V1 heuristic: 0.25 on all-MiniLM-L6-v2 consistently indicates topical
#: overlap.  Do NOT raise this blindly — if live integration tests show that
#: genuinely relevant evidence is being discarded, report the regression and
#: consider reverting to 0.10 rather than escalating further.
DEFAULT_MIN_RELEVANCE: float = 0.25

#: Maximum number of evidence items to run NLI on (most relevant first).
DEFAULT_MAX_NLI_ITEMS: int = 20

#: Maximum number of items to return in the final result.
DEFAULT_TOP_K: int = 10

#: How many top passages to retrieve per article via hybrid retrieval.
#: NLI then runs on the single best-ranked passage from this set.
DEFAULT_HYBRID_TOP_K: int = 7


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class EvidenceAnalysis:
    """
    Analysis of a single piece of evidence against the claim.

    Attributes
    ----------
    title : str
        Page or article title.
    url : str
        Source URL.
    source_name : str
        Human-readable domain name.
    published_at : Optional[str]
        Publication date if available.
    content : str
        Extracted page text (may be empty for failed fetches).
    relevance_score : float
        Cosine similarity between claim embedding and evidence embedding.
        Range ``[0.0, 1.0]``; clipped from ``[-1, 1]`` so callers can treat
        it as a non-negative relevance signal.
    nli_label : str
        NLI classification: ``ENTAILMENT``, ``CONTRADICTION``, or ``NEUTRAL``.
        ``SKIPPED`` when content was empty or NLI was not run.
    entailment_score : float
        Softmax probability for ENTAILMENT (0.0 when NLI not run).
    contradiction_score : float
        Softmax probability for CONTRADICTION (0.0 when NLI not run).
    neutral_score : float
        Softmax probability for NEUTRAL (0.0 when NLI not run).
    fetch_error : Optional[str]
        Error message from web fetcher, if any.
    nli_error : Optional[str]
        Error message from NLI inference, if any.
    """

    title: str
    url: str
    source_name: str
    content: str
    relevance_score: float
    nli_label: str
    entailment_score: float
    contradiction_score: float
    neutral_score: float
    published_at: Optional[str] = None
    fetch_error: Optional[str] = None
    nli_error: Optional[str] = None
    #: Number of passages the source article was split into during hybrid
    #: retrieval.  1 means the article was short enough to use as-is.
    passage_count: int = 1

    @property
    def snippet(self) -> str:
        """First 200 chars of content."""
        return self.content[:200]

    @property
    def was_fetched(self) -> bool:
        """``True`` if the page was fetched without error."""
        return self.fetch_error is None and bool(self.content.strip())


@dataclass
class EvidenceEngineResult:
    """
    Full output of the Evidence Engine for a single claim.

    Attributes
    ----------
    claim : str
        The original claim string passed to the engine.
    analyzed_evidence : list[EvidenceAnalysis]
        All evidence items ranked by ``relevance_score`` descending.
        Includes supporting, contradicting, and neutral items.
    supporting_evidence : list[EvidenceAnalysis]
        Items whose ``nli_label == "ENTAILMENT"``.
    contradicting_evidence : list[EvidenceAnalysis]
        Items whose ``nli_label == "CONTRADICTION"``.
    neutral_evidence : list[EvidenceAnalysis]
        Items whose ``nli_label == "NEUTRAL"`` or ``"SKIPPED"``.
    total_processed : int
        Total number of WebEvidence items the engine received.
    total_skipped : int
        Number of items skipped due to empty content or duplicate URL.
    """

    claim: str
    analyzed_evidence: list[EvidenceAnalysis] = field(default_factory=list)
    supporting_evidence: list[EvidenceAnalysis] = field(default_factory=list)
    contradicting_evidence: list[EvidenceAnalysis] = field(default_factory=list)
    neutral_evidence: list[EvidenceAnalysis] = field(default_factory=list)
    total_processed: int = 0
    total_skipped: int = 0


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class EvidenceEngine:
    """
    Orchestrates semantic scoring and NLI analysis across a set of web
    evidence items for a given claim.

    Parameters
    ----------
    embedding_service : EmbeddingService
        Pre-instantiated embedding service.  The model is loaded lazily on
        first use inside the service itself.
    transformer_service : TransformerService
        Pre-instantiated NLI service.  Loaded lazily on first use.
    min_relevance : float
        Cosine similarity threshold below which NLI is skipped.
        Items below this threshold are still returned, labelled NEUTRAL.
    max_nli_items : int
        Maximum items (by descending relevance) that run through NLI.
        Caps compute when many results are retrieved.
    top_k : int
        Maximum items to return in ``analyzed_evidence``.
        0 = no limit.

    Examples
    --------
    >>> from app.rag.embeddings import EmbeddingService
    >>> from app.models.transformer import TransformerService
    >>> engine = EvidenceEngine(EmbeddingService(), TransformerService())
    >>> result = engine.analyze("Two cars crashed in Damoh.", evidence_list)
    >>> result.supporting_evidence
    [...]
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        transformer_service: TransformerService,
        min_relevance: float = DEFAULT_MIN_RELEVANCE,
        max_nli_items: int = DEFAULT_MAX_NLI_ITEMS,
        top_k: int = DEFAULT_TOP_K,
        hybrid_top_k: int = DEFAULT_HYBRID_TOP_K,
    ) -> None:
        self._embedder = embedding_service
        self._nli = transformer_service
        self._min_relevance = min_relevance
        self._max_nli_items = max_nli_items
        self._top_k = top_k
        self._hybrid_top_k = hybrid_top_k

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        claim: str,
        evidence_list: list[WebEvidence],
    ) -> EvidenceEngineResult:
        """
        Run the full Evidence Engine pipeline.

        Steps:

        1. Validate and normalise inputs.
        2. Deduplicate evidence by URL.
        3. Embed the claim.
        4. For each evidence item with usable content:
           a. Embed the content.
           b. Compute cosine similarity → ``relevance_score``.
        5. Sort all items by ``relevance_score`` descending.
        6. Run NLI on the top ``max_nli_items`` items that meet
           ``min_relevance``.
        7. Partition into supporting / contradicting / neutral buckets.
        8. Return :class:`EvidenceEngineResult`.

        Parameters
        ----------
        claim : str
            Raw user claim.  Must be a non-empty string.
        evidence_list : list[WebEvidence]
            Evidence items from :class:`~app.services.web_evaluator.WebEvaluatorService`.
            May be empty.

        Returns
        -------
        EvidenceEngineResult
            Never raises; errors inside individual items are captured.

        Raises
        ------
        TypeError
            If *claim* is not a string.
        ValueError
            If *claim* is empty or whitespace-only.
        """
        if not isinstance(claim, str):
            raise TypeError(f"claim must be a str, got {type(claim).__name__!r}")
        if not claim.strip():
            raise ValueError("claim must not be empty or whitespace-only.")

        result = EvidenceEngineResult(
            claim=claim,
            total_processed=len(evidence_list),
        )

        if not evidence_list:
            logger.info("EvidenceEngine: received empty evidence list.")
            return result

        # Step 1 — deduplicate by URL.
        seen_urls: set[str] = set()
        unique_evidence: list[WebEvidence] = []
        for item in evidence_list:
            if item.url in seen_urls:
                result.total_skipped += 1
                logger.debug("Skipping duplicate URL: %s", item.url)
                continue
            seen_urls.add(item.url)
            unique_evidence.append(item)

        # Step 2 — embed the claim (1 × 384).
        try:
            claim_vec = self._embedder.encode_text(claim)
        except Exception as exc:
            logger.error("Failed to embed claim: %s", exc)
            # Cannot proceed without a claim vector.
            return result

        # Step 3 — embed each evidence item and score relevance.
        # Items with empty content are collected separately; they bypass
        # embedding and NLI but still appear in the final result.
        scored: list[tuple[float, WebEvidence]] = []
        empty_content_analyses: list[EvidenceAnalysis] = []
        for ev in unique_evidence:
            content = ev.content.strip()
            if not content:
                result.total_skipped += 1
                logger.debug("Skipping evidence with empty content: %s", ev.url)
                empty_content_analyses.append(
                    self._make_skipped(ev, reason="empty content")
                )
                continue

            try:
                ev_vec = self._embedder.encode_text(content)
                # similarity() returns shape (1,) when given a single doc vec.
                import numpy as np
                sim = float(
                    self._embedder.similarity(
                        claim_vec, ev_vec.reshape(1, -1)
                    )[0]
                )
                # Clip to [0, 1] — negative cosine similarity means
                # semantically opposite, treat as 0 relevance for ranking.
                relevance = max(0.0, sim)
            except Exception as exc:
                logger.warning(
                    "Embedding error for %s: %s — relevance set to 0.", ev.url, exc
                )
                relevance = 0.0

            scored.append((relevance, ev))

        # Step 4 — sort by descending relevance.
        scored.sort(key=lambda t: t[0], reverse=True)

        # Step 5 — split into NLI-eligible and overflow batches.
        scored_for_nli = scored[: self._max_nli_items]
        scored_rest = scored[self._max_nli_items :]

        # Step 6 — run NLI on eligible items.
        analyses: list[EvidenceAnalysis] = []
        for relevance, ev in scored_for_nli:
            analysis = self._run_nli(claim, ev, relevance)
            analyses.append(analysis)

        # Items beyond max_nli_items: keep with relevance score but no NLI.
        for relevance, ev in scored_rest:
            analyses.append(
                self._make_skipped(
                    ev,
                    reason="max_nli_items limit reached",
                    relevance=relevance,
                )
            )

        # Merge empty-content skipped items at the end (lowest relevance).
        # They already have relevance_score=0.0 so ordering is preserved.
        analyses.extend(empty_content_analyses)

        # Trim to top_k for the final output (0 means no limit).
        if self._top_k > 0:
            result.analyzed_evidence = analyses[: self._top_k]
        else:
            result.analyzed_evidence = analyses

        # Step 7 — partition into supporting / contradicting / neutral buckets.
        # Classification is based solely on NLI label; NEUTRAL and SKIPPED both
        # fall into neutral_evidence so that NEUTRAL is never treated as support.
        for analysis in result.analyzed_evidence:
            if analysis.nli_label == LABEL_ENTAILMENT:
                result.supporting_evidence.append(analysis)
            elif analysis.nli_label == LABEL_CONTRADICTION:
                result.contradicting_evidence.append(analysis)
            else:
                # Covers NEUTRAL, SKIPPED, and any unexpected label.
                result.neutral_evidence.append(analysis)

        logger.info(
            "EvidenceEngine: claim=%r | total=%d skipped=%d analyzed=%d "
            "support=%d contra=%d neutral=%d",
            claim[:60],
            result.total_processed,
            result.total_skipped,
            len(result.analyzed_evidence),
            len(result.supporting_evidence),
            len(result.contradicting_evidence),
            len(result.neutral_evidence),
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_nli(
        self,
        claim: str,
        ev: WebEvidence,
        relevance: float,
    ) -> EvidenceAnalysis:
        """
        Run NLI on the best passage extracted from *ev* for *claim*.

        Steps
        -----
        1. Chunk the article text into ≈200-word passages.
        2. If the article has multiple chunks, run hybrid BM25 + MiniLM
           retrieval to select the single most relevant passage.
        3. Recompute the relevance score for that passage (replaces the
           whole-article score used during the initial sort).
        4. Apply the relevance threshold gate.
        5. Pass the best passage as the NLI premise; claim as hypothesis.

        If any step raises, errors are captured and the item is returned
        with label NEUTRAL and the error recorded — same as before.
        """
        content = ev.content.strip()

        # ------------------------------------------------------------------
        # Step 1: chunk the article.
        # ------------------------------------------------------------------
        chunks = chunk_text(content)
        passage_count = len(chunks)

        # ------------------------------------------------------------------
        # Step 2: select best passage via hybrid retrieval.
        # ------------------------------------------------------------------
        best_passage = content  # fallback: use full content
        best_relevance = relevance  # fallback: keep original score

        if passage_count > 1:
            try:
                ranked = hybrid_retrieve(
                    claim=claim,
                    chunks=chunks,
                    embedding_service=self._embedder,
                    top_k=self._hybrid_top_k,
                )
                if ranked:
                    best_idx, _ = ranked[0]
                    best_passage = chunks[best_idx]
                    # Recompute relevance for the selected passage so that the
                    # score reflects what NLI actually received.
                    import numpy as np
                    claim_vec = self._embedder.encode_text(claim)
                    passage_vec = self._embedder.encode_text(best_passage)
                    sim = float(
                        self._embedder.similarity(
                            claim_vec, passage_vec.reshape(1, -1)
                        )[0]
                    )
                    best_relevance = max(0.0, sim)
                    logger.debug(
                        "Hybrid retrieval: %d chunks → best passage idx=%d "
                        "relevance=%.3f (was %.3f) for %s",
                        passage_count, best_idx, best_relevance, relevance, ev.url,
                    )
            except Exception as exc:
                logger.warning(
                    "Hybrid retrieval failed for %s (%s) — using full content.",
                    ev.url, exc,
                )
                # best_passage and best_relevance already set to fallback values.
        else:
            # Article was short enough to be a single chunk; use it directly.
            best_passage = chunks[0] if chunks else content

        # ------------------------------------------------------------------
        # Step 3: relevance threshold gate (unchanged logic, new score).
        # ------------------------------------------------------------------
        if best_relevance < self._min_relevance:
            logger.debug(
                "Skipping NLI for %s (relevance=%.3f < threshold=%.3f)",
                ev.url,
                best_relevance,
                self._min_relevance,
            )
            return EvidenceAnalysis(
                title=ev.title,
                url=ev.url,
                source_name=ev.source_name,
                content=best_passage,
                published_at=ev.published_at,
                fetch_error=ev.error,
                relevance_score=round(best_relevance, 6),
                nli_label=LABEL_NEUTRAL,
                entailment_score=0.0,
                contradiction_score=0.0,
                neutral_score=0.0,
                nli_error="Below relevance threshold — NLI skipped.",
                passage_count=passage_count,
            )

        # ------------------------------------------------------------------
        # Step 4: NLI on the best passage.
        # Premise = passage (evidence), Hypothesis = claim — ordering preserved.
        #
        # Prepend the article title to the passage so the NLI model benefits
        # from the headline context even when the selected chunk is indirect.
        # Falls back to best_passage alone when the title is absent.
        # ------------------------------------------------------------------
        article_title = ev.title.strip() if ev.title else ""
        if article_title:
            evidence_for_nli = f"{article_title}\n{best_passage}"
        else:
            evidence_for_nli = best_passage

        try:
            nli_result: TransformerResult = self._nli.analyze(
                claim=claim,
                evidence=evidence_for_nli,
            )
            return EvidenceAnalysis(
                title=ev.title,
                url=ev.url,
                source_name=ev.source_name,
                content=best_passage,
                published_at=ev.published_at,
                fetch_error=ev.error,
                relevance_score=round(best_relevance, 6),
                nli_label=nli_result.label,
                entailment_score=nli_result.entailment_score,
                contradiction_score=nli_result.contradiction_score,
                neutral_score=nli_result.neutral_score,
                passage_count=passage_count,
            )
        except Exception as exc:
            logger.warning("NLI error for %s: %s", ev.url, exc)
            return EvidenceAnalysis(
                title=ev.title,
                url=ev.url,
                source_name=ev.source_name,
                content=best_passage,
                published_at=ev.published_at,
                fetch_error=ev.error,
                relevance_score=round(best_relevance, 6),
                nli_label=LABEL_NEUTRAL,
                entailment_score=0.0,
                contradiction_score=0.0,
                neutral_score=0.0,
                nli_error=str(exc),
                passage_count=passage_count,
            )

    @staticmethod
    def _make_skipped(
        ev: WebEvidence,
        reason: str = "skipped",
        relevance: float = 0.0,
    ) -> EvidenceAnalysis:
        """Build a placeholder EvidenceAnalysis for items that were skipped."""
        return EvidenceAnalysis(
            title=ev.title,
            url=ev.url,
            source_name=ev.source_name,
            content=ev.content,
            published_at=ev.published_at,
            fetch_error=ev.error,
            relevance_score=round(relevance, 6),
            nli_label="SKIPPED",
            entailment_score=0.0,
            contradiction_score=0.0,
            neutral_score=0.0,
            nli_error=reason,
            passage_count=1,
        )
