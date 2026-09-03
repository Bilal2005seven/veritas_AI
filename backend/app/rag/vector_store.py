"""
VeritasAI V1 — Vector Store

Responsibilities:
  - Persist and index document embeddings on disk.
  - Support adding new documents with their embeddings.
  - Support similarity search to retrieve top-k nearest neighbours.

TODO: Implement this module.
      FAISS is the intended library for the initial implementation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Document:
    """A text document stored in the vector store."""

    doc_id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class VectorStore:
    """
    Manages a FAISS index for dense vector similarity search.

    Usage (once implemented):
        store = VectorStore(path="./data/vector_store", dim=384)
        store.load()
        store.add(documents)
        results = store.search(query_embedding, top_k=5)
    """

    def __init__(self, path: str, dim: int = 384):
        self.path = path
        self.dim = dim
        self._index = None       # TODO: FAISS IndexFlatIP or IndexHNSWFlat
        self._metadata: List[Dict[str, Any]] = []

    def load(self) -> None:
        """
        Load a persisted FAISS index from disk.

        TODO:
          - Use faiss.read_index(path) to restore the index.
          - Load the corresponding metadata list from a JSON sidecar file.
          - Create an empty index if no index file exists yet.
        """
        raise NotImplementedError("VectorStore.load is not yet implemented.")

    def save(self) -> None:
        """
        Persist the FAISS index and metadata to disk.

        TODO:
          - Use faiss.write_index(self._index, path).
          - Serialise metadata to a JSON sidecar file alongside the index.
        """
        raise NotImplementedError("VectorStore.save is not yet implemented.")

    def add(self, documents: List[Document]) -> None:
        """
        Add documents and their embeddings to the index.

        Args:
            documents: List of Document objects with pre-computed embeddings.

        TODO:
          - Stack embeddings into a numpy float32 array.
          - Call self._index.add(vectors).
          - Append document metadata to self._metadata.
          - Auto-save after each batch.
        """
        raise NotImplementedError("VectorStore.add is not yet implemented.")

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Document]:
        """
        Find the top-k most similar documents to the query embedding.

        Args:
            query_embedding: Dense vector representation of the query.
            top_k: Number of results to return.

        Returns:
            List of Documents ordered by similarity (descending).

        TODO:
          - Convert query_embedding to a (1, dim) float32 numpy array.
          - Call self._index.search(vector, top_k) to get distances and indices.
          - Map result indices back to Document metadata.
          - Filter out results below a minimum similarity threshold.
        """
        raise NotImplementedError("VectorStore.search is not yet implemented.")
