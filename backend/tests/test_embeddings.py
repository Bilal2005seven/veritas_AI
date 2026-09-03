"""
test_embeddings.py — Manual verification script for EmbeddingService.

Run from the backend/ directory:
    python tests/test_embeddings.py

Expected output (shapes and top-k indices; exact scores vary slightly by platform):
    Claim embedding shape : (384,)
    Docs  embedding shape : (3, 384)
    Similarity scores     : [0.xxxx  0.xxxx  0.xxxx]
    Top-2 results         : [(idx, score), (idx, score)]
    ✓ All assertions passed.
"""

import sys
import os

# Ensure the backend/app package is importable regardless of CWD.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from app.rag.embeddings import EmbeddingService, EMBEDDING_DIM


def main() -> None:
    # ── 1. Instantiate ──────────────────────────────────────────────────────
    print("Instantiating EmbeddingService …")
    svc = EmbeddingService()

    # ── 2. Encode one claim ─────────────────────────────────────────────────
    claim = (
        "A photograph circulating on social media shows politicians shaking "
        "hands at a summit that allegedly never took place."
    )
    print(f"\nClaim : {claim[:80]}…")
    claim_vec = svc.encode_text(claim)

    # ── 3. Encode 3 evidence documents ──────────────────────────────────────
    evidence_docs = [
        # Relevant — confirms the event happened
        (
            "Reuters fact-checkers verified that the bilateral summit did occur "
            "on 14 March 2024 and official photographs were released by both "
            "governments."
        ),
        # Partially relevant — mentions social-media misinformation context
        (
            "A 2023 study found that manipulated images spread six times faster "
            "on social-media platforms than corrections issued by fact-checkers."
        ),
        # Off-topic — unrelated news item
        (
            "The national cricket team won the series 3-1, with the captain "
            "scoring a century in the final match."
        ),
    ]
    print(f"\nEncoding {len(evidence_docs)} evidence documents …")
    docs_matrix = svc.encode_documents(evidence_docs)

    # ── 4. Verify shapes ─────────────────────────────────────────────────────
    print(f"\nClaim embedding shape : {claim_vec.shape}")
    print(f"Docs  embedding shape : {docs_matrix.shape}")

    assert claim_vec.shape == (EMBEDDING_DIM,), (
        f"Expected ({EMBEDDING_DIM},), got {claim_vec.shape}"
    )
    assert docs_matrix.shape == (3, EMBEDDING_DIM), (
        f"Expected (3, {EMBEDDING_DIM}), got {docs_matrix.shape}"
    )

    # ── 5. Verify dtype ──────────────────────────────────────────────────────
    assert claim_vec.dtype == np.float32, f"Expected float32, got {claim_vec.dtype}"
    assert docs_matrix.dtype == np.float32, f"Expected float32, got {docs_matrix.dtype}"

    # ── 6. Calculate similarities ────────────────────────────────────────────
    scores = svc.similarity(claim_vec, docs_matrix)
    print(f"\nSimilarity scores     : {np.array2string(scores, precision=4, floatmode='fixed')}")

    assert scores.shape == (3,), f"Expected (3,), got {scores.shape}"
    assert np.all(scores >= -1.0) and np.all(scores <= 1.0), (
        "Scores must be in [-1, 1]"
    )

    # ── 7. Retrieve top-2 ────────────────────────────────────────────────────
    top2 = svc.top_k(claim_vec, docs_matrix, k=2)
    print(f"\nTop-2 results         : {top2}")

    assert len(top2) == 2, f"Expected 2 results, got {len(top2)}"
    # Scores must be in descending order.
    assert top2[0][1] >= top2[1][1], "Top-k results must be sorted descending."

    # ── 8. Edge-case guards ───────────────────────────────────────────────────
    # k > n_docs should not crash — clamps to n_docs.
    top_all = svc.top_k(claim_vec, docs_matrix, k=100)
    assert len(top_all) == 3, f"Expected 3 (clamped), got {len(top_all)}"

    # Empty document list.
    empty_mat = svc.encode_documents([])
    assert empty_mat.shape == (0, EMBEDDING_DIM), (
        f"Expected (0, {EMBEDDING_DIM}), got {empty_mat.shape}"
    )

    # top_k on empty documents.
    top_empty = svc.top_k(claim_vec, empty_mat, k=3)
    assert top_empty == [], f"Expected [], got {top_empty}"

    # k = 0 should return empty.
    top_zero_k = svc.top_k(claim_vec, docs_matrix, k=0)
    assert top_zero_k == [], f"Expected [], got {top_zero_k}"

    # -- Summary ---------------------------------------------------------------
    print("\n" + "-" * 60)
    print(f"  Embedding dim         : {EMBEDDING_DIM}")
    print(f"  Claim shape           : {claim_vec.shape}")
    print(f"  Docs matrix shape     : {docs_matrix.shape}")
    print(f"  Similarity scores     : {scores.tolist()}")
    print(f"  Top-2 (index, score)  : {top2}")
    print("-" * 60)
    print("[PASS] All assertions passed.")


if __name__ == "__main__":
    main()
