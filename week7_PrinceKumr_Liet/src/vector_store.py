"""
vector_store.py
----------------
Stage 4 of the RAG pipeline: Vector Database.

Stores chunk embeddings and supports similarity search.

Two backends:
  - NumpyVectorStore: brute-force cosine similarity with numpy/sklearn.
                       No extra dependency, fine up to tens of thousands
                       of chunks -- perfect for a single-document RAG demo.
  - FaissVectorStore: wraps FAISS for larger corpora / production use.
                       Requires: pip install faiss-cpu
"""

import pickle
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class NumpyVectorStore:
    def __init__(self):
        self.embeddings = None   # (n, dim)
        self.chunks = []         # list[Chunk]

    def build(self, embeddings: np.ndarray, chunks: list):
        assert len(embeddings) == len(chunks), "embeddings/chunks length mismatch"
        self.embeddings = embeddings
        self.chunks = chunks

    def search(self, query_embedding: np.ndarray, top_k: int = 3):
        """Return the top_k (chunk, score) pairs most similar to the query."""
        if self.embeddings is None or len(self.chunks) == 0:
            return []
        sims = cosine_similarity(query_embedding.reshape(1, -1), self.embeddings)[0]
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [(self.chunks[i], float(sims[i])) for i in top_idx]

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"embeddings": self.embeddings, "chunks": self.chunks}, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.embeddings = data["embeddings"]
        self.chunks = data["chunks"]
        return self


class FaissVectorStore:
    """Optional FAISS-backed store for larger document collections."""

    def __init__(self, dim: int):
        import faiss
        self.index = faiss.IndexFlatIP(dim)  # inner product on normalized vecs = cosine sim
        self.chunks = []

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        return x / norms

    def build(self, embeddings: np.ndarray, chunks: list):
        vecs = self._normalize(embeddings.astype(np.float32))
        self.index.add(vecs)
        self.chunks = chunks

    def search(self, query_embedding: np.ndarray, top_k: int = 3):
        q = self._normalize(query_embedding.reshape(1, -1).astype(np.float32))
        scores, idx = self.index.search(q, top_k)
        results = []
        for score, i in zip(scores[0], idx[0]):
            if i == -1:
                continue
            results.append((self.chunks[i], float(score)))
        return results


def get_vector_store(name: str = "numpy", dim: int = None):
    if name == "numpy":
        return NumpyVectorStore()
    elif name == "faiss":
        if dim is None:
            raise ValueError("FaissVectorStore requires `dim`")
        return FaissVectorStore(dim)
    else:
        raise ValueError(f"Unknown vector store: {name}")
