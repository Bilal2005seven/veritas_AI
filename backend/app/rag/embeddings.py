"""
VeritasAI V1 — Embedding Service
=================================
Wraps ``sentence-transformers/all-MiniLM-L6-v2`` to produce 384-dimensional
dense text embeddings used for semantic similarity and evidence retrieval.

Design decisions
----------------
* **Lazy loading** — the SentenceTransformer model is only instantiated on the
  first call to :meth:`encode_text` or :meth:`encode_documents`.  Importing
  this module does *not* trigger a download or load.
* **NumPy cosine similarity** — no FAISS, no ChromaDB, no LangChain.
* **L2 normalisation** — embeddings are L2-normalised before similarity
  calculations so that cosine similarity reduces to a plain dot product and
  remains numerically stable even for near-zero vectors.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Public constant so callers can verify the expected dimension.
EMBEDDING_DIM: int = 384
_DEFAULT_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingService:
    """
    Provides text embedding and cosine-similarity utilities backed by
    ``sentence-transformers/all-MiniLM-L6-v2``.

    Attributes
    ----------
    model_name : str
        HuggingFace model identifier.
    embedding_dim : int
        Output dimensionality (384 for all-MiniLM-L6-v2).

    Examples
    --------
    >>> svc = EmbeddingService()
    >>> claim_vec = svc.encode_text("The earth is round.")
    >>> claim_vec.shape
    (384,)
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self.model_name: str = model_name
        self.embedding_dim: int = EMBEDDING_DIM
        self._model: Optional[object] = None  # loaded lazily

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load the SentenceTransformer model if it has not been loaded yet."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            ) from exc

        logger.info("Loading embedding model '%s' …", self.model_name)
        self._model = SentenceTransformer(self.model_name)
        logger.info("Embedding model loaded (dim=%d).", self.embedding_dim)

    @staticmethod
    def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
        """
        L2-normalise *vectors* along the last axis in-place-safe manner.

        Zero vectors are left as-is (their norm is 0 and dividing would
        produce NaN).

        Parameters
        ----------
        vectors : np.ndarray
            Shape ``(n, d)`` or ``(d,)``.

        Returns
        -------
        np.ndarray
            Unit-norm vectors of the same shape as *vectors*.
        """
        norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
        # Avoid division by zero for zero vectors.
        norms = np.where(norms == 0.0, 1.0, norms)
        return vectors / norms

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode a single text string into a 384-dimensional embedding.

        Parameters
        ----------
        text : str
            Input text.  Empty strings are allowed and produce a valid
            (but semantically meaningless) embedding vector.

        Returns
        -------
        np.ndarray
            Shape ``(384,)``, dtype ``float32``, L2-normalised.
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be a str, got {type(text).__name__!r}")

        self._load_model()

        # sentence-transformers returns shape (384,) for a single string.
        embedding: np.ndarray = self._model.encode(  # type: ignore[union-attr]
            text,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        embedding = np.array(embedding, dtype=np.float32)
        return self._l2_normalize(embedding)

    def encode_documents(self, documents: list[str]) -> np.ndarray:
        """
        Encode a list of documents in a single batched forward pass.

        Parameters
        ----------
        documents : list[str]
            Texts to encode.  An empty list returns a ``(0, 384)`` array.

        Returns
        -------
        np.ndarray
            Shape ``(len(documents), 384)``, dtype ``float32``, L2-normalised.
        """
        if not isinstance(documents, list):
            raise TypeError(
                f"documents must be a list, got {type(documents).__name__!r}"
            )
        if len(documents) == 0:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        self._load_model()

        embeddings: np.ndarray = self._model.encode(  # type: ignore[union-attr]
            documents,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        embeddings = np.array(embeddings, dtype=np.float32)
        return self._l2_normalize(embeddings)

    def similarity(
        self,
        query_embedding: np.ndarray,
        document_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Compute cosine similarity between one query and N documents.

        Because both inputs are expected to be L2-normalised (as produced by
        :meth:`encode_text` / :meth:`encode_documents`), cosine similarity
        reduces to a dot product.

        Parameters
        ----------
        query_embedding : np.ndarray
            Shape ``(384,)`` — single query vector.
        document_embeddings : np.ndarray
            Shape ``(n, 384)`` — n document vectors.

        Returns
        -------
        np.ndarray
            Shape ``(n,)`` of cosine-similarity scores in ``[-1, 1]``.
            Returns an empty array ``(0,)`` when *document_embeddings* is
            empty.
        """
        query_embedding = np.asarray(query_embedding, dtype=np.float32)
        document_embeddings = np.asarray(document_embeddings, dtype=np.float32)

        if document_embeddings.ndim == 1:
            document_embeddings = document_embeddings[np.newaxis, :]

        if document_embeddings.shape[0] == 0:
            return np.empty(0, dtype=np.float32)

        # Re-normalise defensively (callers may pass raw embeddings).
        q = self._l2_normalize(query_embedding)   # (384,)
        D = self._l2_normalize(document_embeddings)  # (n, 384)

        scores: np.ndarray = D @ q  # (n,)
        return scores.astype(np.float32)

    def top_k(
        self,
        query_embedding: np.ndarray,
        document_embeddings: np.ndarray,
        k: int,
    ) -> list[tuple[int, float]]:
        """
        Return the top-*k* documents most similar to the query.

        Parameters
        ----------
        query_embedding : np.ndarray
            Shape ``(384,)`` — single query vector.
        document_embeddings : np.ndarray
            Shape ``(n, 384)`` — n document vectors.
        k : int
            Number of top results to return.  Clamped to
            ``len(document_embeddings)`` when *k* exceeds it.

        Returns
        -------
        list[tuple[int, float]]
            List of ``(index, score)`` pairs sorted by descending similarity.
            Empty list when there are no documents or *k* ≤ 0.

        Raises
        ------
        TypeError
            If *k* is not an integer.
        """
        if not isinstance(k, int):
            raise TypeError(f"k must be an int, got {type(k).__name__!r}")

        document_embeddings = np.asarray(document_embeddings, dtype=np.float32)

        if document_embeddings.ndim == 1:
            document_embeddings = document_embeddings[np.newaxis, :]

        n_docs = document_embeddings.shape[0]

        if k <= 0 or n_docs == 0:
            return []

        # Clamp k to the number of available documents.
        effective_k = min(k, n_docs)

        scores = self.similarity(query_embedding, document_embeddings)  # (n,)

        # np.argpartition gives us the top-k indices without a full sort,
        # then we sort only those k elements.
        if effective_k < n_docs:
            top_indices = np.argpartition(scores, -effective_k)[-effective_k:]
        else:
            top_indices = np.arange(n_docs)

        # Sort the selected indices by score descending.
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [(int(idx), float(scores[idx])) for idx in top_indices]
