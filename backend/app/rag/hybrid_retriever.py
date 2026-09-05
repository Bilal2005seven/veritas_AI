"""
VeritasAI V1 — Hybrid Passage Retriever
=========================================
Combines BM25 lexical retrieval with MiniLM semantic retrieval using
Reciprocal Rank Fusion (RRF) to select the most relevant passages from a
chunked article.

Design
------
* **BM25** catches exact/lexical matches (names, numbers, dates, quoted phrases).
* **MiniLM semantic similarity** catches paraphrased or conceptually related
  passages even when the exact words differ.
* **RRF** fuses both ranked lists without requiring score normalisation or
  weight tuning.

The :func:`hybrid_retrieve` function is the only public interface.  It is
intentionally a plain function (not a class) to avoid unnecessary state.

Dependencies
------------
* ``rank-bm25>=0.2.2``        — BM25Okapi implementation.
* ``app.rag.embeddings``       — the EXISTING EmbeddingService (reused, not duplicated).
* ``numpy``                    — already a project dependency.

Notes
-----
* This module does NOT introduce a new embedding model.
* BM25 tokenisation uses simple whitespace + lowercase splitting — no tokenizer
  library is required.
* The RRF constant ``k=60`` is a standard default; do NOT tune it.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

from app.rag.embeddings import EmbeddingService

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: RRF smoothing constant.  Standard value; do not change.
RRF_K: int = 60

#: Default number of top passages to return.
DEFAULT_TOP_K: int = 3


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """
    Tokenise *text* for BM25: lowercase and split on whitespace.

    This is intentionally simple — no stemming, no stopword removal.
    BM25 benefits from exact token overlap; aggressive preprocessing would
    harm recall for names, numbers, and quoted phrases.
    """
    return text.lower().split()


def _bm25_scores(query: str, chunks: List[str]) -> np.ndarray:
    """
    Compute BM25 scores for all *chunks* against *query*.

    Parameters
    ----------
    query : str
        The claim or question string.
    chunks : list[str]
        Text passages to score.

    Returns
    -------
    np.ndarray
        Shape ``(len(chunks),)`` of BM25 relevance scores.  Values are
        non-negative; higher is more relevant.
    """
    try:
        from rank_bm25 import BM25Okapi  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "rank-bm25 is not installed. Run: pip install rank-bm25"
        ) from exc

    tokenized_chunks = [_tokenize(c) for c in chunks]
    tokenized_query = _tokenize(query)

    bm25 = BM25Okapi(tokenized_chunks)
    scores: np.ndarray = bm25.get_scores(tokenized_query)
    return scores.astype(np.float32)


def _semantic_scores(
    claim: str,
    chunks: List[str],
    embedding_service: EmbeddingService,
) -> np.ndarray:
    """
    Compute MiniLM cosine-similarity scores for all *chunks* against *claim*.

    Reuses the existing :class:`~app.rag.embeddings.EmbeddingService` instance;
    does NOT create a new embedding model.

    Parameters
    ----------
    claim : str
        The claim string.
    chunks : list[str]
        Text passages to score.
    embedding_service : EmbeddingService
        The existing shared embedding service instance.

    Returns
    -------
    np.ndarray
        Shape ``(len(chunks),)`` of cosine-similarity scores in ``[-1, 1]``.
    """
    claim_vec = embedding_service.encode_text(claim)         # (384,)
    chunk_vecs = embedding_service.encode_documents(chunks)  # (n, 384)
    scores = embedding_service.similarity(claim_vec, chunk_vecs)  # (n,)
    return scores


def _rrf_fusion(
    bm25_scores: np.ndarray,
    semantic_scores: np.ndarray,
    k: int = RRF_K,
) -> np.ndarray:
    """
    Combine two score arrays using Reciprocal Rank Fusion.

    RRF formula: ``score(d) = 1/(k + rank_bm25(d)) + 1/(k + rank_sem(d))``

    Ranks are 1-indexed (rank 1 = best).  Higher RRF score = more relevant.

    Parameters
    ----------
    bm25_scores : np.ndarray
        Shape ``(n,)`` — raw BM25 scores (higher = better).
    semantic_scores : np.ndarray
        Shape ``(n,)`` — cosine similarity scores (higher = better).
    k : int
        RRF smoothing constant (default 60).

    Returns
    -------
    np.ndarray
        Shape ``(n,)`` — fused RRF scores.
    """
    n = len(bm25_scores)

    # Convert scores to ranks (rank 1 = best = highest score).
    # argsort ascending, then invert.
    bm25_ranks = np.empty(n, dtype=np.float32)
    bm25_ranks[np.argsort(bm25_scores)[::-1]] = np.arange(1, n + 1, dtype=np.float32)

    sem_ranks = np.empty(n, dtype=np.float32)
    sem_ranks[np.argsort(semantic_scores)[::-1]] = np.arange(1, n + 1, dtype=np.float32)

    rrf = 1.0 / (k + bm25_ranks) + 1.0 / (k + sem_ranks)
    return rrf.astype(np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def hybrid_retrieve(
    claim: str,
    chunks: List[str],
    embedding_service: EmbeddingService,
    top_k: int = DEFAULT_TOP_K,
) -> List[Tuple[int, float]]:
    """
    Retrieve the top-*k* most relevant passages from *chunks* for *claim*.

    Steps
    -----
    1. Compute BM25 scores for all chunks against the claim.
    2. Compute MiniLM cosine-similarity scores using the existing
       :class:`~app.rag.embeddings.EmbeddingService`.
    3. Fuse both ranked lists with Reciprocal Rank Fusion (RRF, k=60).
    4. Return the top-*k* ``(chunk_index, rrf_score)`` pairs sorted by
       descending RRF score.

    Parameters
    ----------
    claim : str
        The claim or question string.
    chunks : list[str]
        Text passages (already chunked from a source article).
    embedding_service : EmbeddingService
        The existing shared embedding service instance.  Must not be ``None``.
    top_k : int
        Maximum number of passages to return.

    Returns
    -------
    list[tuple[int, float]]
        ``(chunk_index, rrf_score)`` pairs, sorted descending by score.
        At most ``min(top_k, len(chunks))`` items are returned.
        Returns ``[]`` if *chunks* is empty.

    Raises
    ------
    TypeError
        If *claim* is not a string or *chunks* is not a list.
    """
    if not isinstance(claim, str):
        raise TypeError(f"claim must be a str, got {type(claim).__name__!r}")
    if not isinstance(chunks, list):
        raise TypeError(f"chunks must be a list, got {type(chunks).__name__!r}")

    if not chunks:
        return []

    n = len(chunks)
    effective_k = min(top_k, n)

    if n == 1:
        # Single chunk — no retrieval needed; return it directly.
        return [(0, 1.0)]

    try:
        bm25 = _bm25_scores(claim, chunks)
    except Exception as exc:
        logger.warning("BM25 scoring failed (%s) — falling back to semantic only.", exc)
        bm25 = np.zeros(n, dtype=np.float32)

    try:
        semantic = _semantic_scores(claim, chunks, embedding_service)
        # Clip negative cosine scores to 0 (semantically opposite → 0 relevance).
        semantic = np.clip(semantic, 0.0, 1.0)
    except Exception as exc:
        logger.warning(
            "Semantic scoring failed (%s) — falling back to BM25 only.", exc
        )
        semantic = np.zeros(n, dtype=np.float32)

    rrf = _rrf_fusion(bm25, semantic, k=RRF_K)

    # Pick top-k indices by descending RRF score.
    if effective_k < n:
        top_indices = np.argpartition(rrf, -effective_k)[-effective_k:]
    else:
        top_indices = np.arange(n)

    top_indices = top_indices[np.argsort(rrf[top_indices])[::-1]]

    result: List[Tuple[int, float]] = [
        (int(idx), float(rrf[idx])) for idx in top_indices
    ]

    logger.debug(
        "hybrid_retrieve: %d chunks → top-%d selected (top score=%.4f)",
        n, effective_k, result[0][1] if result else 0.0,
    )
    return result
