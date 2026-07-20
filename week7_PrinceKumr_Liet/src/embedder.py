"""
embedder.py
-----------
Stage 3 of the RAG pipeline: Embedding Creation.

Two interchangeable backends are provided behind one interface:

  1. TfidfEmbedder            - pure scikit-learn, no downloads, no GPU.
                                 Great default: always works, easy to explain.
  2. SentenceTransformerEmbedder - true dense semantic embeddings
                                 (e.g. all-MiniLM-L6-v2). Better retrieval
                                 quality, but requires internet access the
                                 first time to download the model, plus the
                                 `sentence-transformers` package.

Swap backends with one line -> this is the "try different embedding
models" experiment the assignment brief asks for.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseEmbedder(ABC):
    @abstractmethod
    def fit(self, texts):
        """Learn the embedding space from a corpus of texts (if needed)."""

    @abstractmethod
    def transform(self, texts):
        """Return an (n_texts, dim) numpy array of embeddings."""

    def fit_transform(self, texts):
        self.fit(texts)
        return self.transform(texts)


class TfidfEmbedder(BaseEmbedder):
    """Lightweight, fully offline embedding backend using TF-IDF."""

    def __init__(self, max_features: int = 5000):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=(1, 2),
        )
        self._fitted = False

    def fit(self, texts):
        self.vectorizer.fit(texts)
        self._fitted = True

    def transform(self, texts):
        if not self._fitted:
            raise RuntimeError("Call fit() or fit_transform() before transform().")
        matrix = self.vectorizer.transform(texts)
        return matrix.toarray().astype(np.float32)


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Dense semantic embeddings via a pretrained sentence-transformers model.
    Requires: pip install sentence-transformers
    Requires internet access on first run to download the model weights.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def fit(self, texts):
        # No fitting needed: the model is pretrained.
        pass

    def transform(self, texts):
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.astype(np.float32)


def get_embedder(name: str = "tfidf", **kwargs) -> BaseEmbedder:
    """Factory so the CLI can select a backend by string name."""
    if name == "tfidf":
        return TfidfEmbedder(**kwargs)
    elif name == "sentence-transformer":
        return SentenceTransformerEmbedder(**kwargs)
    else:
        raise ValueError(f"Unknown embedder: {name}")
