"""
rag_pipeline.py
----------------
Wires stages 1-7 together into one RagPipeline class:

  1. Document Ingestion   -> document_loader.load_document
  2. Text Chunking        -> chunker.chunk_text
  3. Embedding Creation    -> embedder.BaseEmbedder
  4. Vector Database       -> vector_store.NumpyVectorStore / FaissVectorStore
  5. Query Processing      -> embedder.transform([question])
  6. Context Retrieval     -> vector_store.search
  7. Answer Generation     -> generator.BaseGenerator
"""

from typing import List
from .document_loader import load_document
from .chunker import chunk_text, Chunk
from .embedder import get_embedder
from .vector_store import get_vector_store
from .generator import get_generator


class RagPipeline:
    def __init__(
        self,
        embedder_name: str = "tfidf",
        vector_store_name: str = "numpy",
        generator_name: str = "extractive",
        chunk_size: int = 200,
        overlap: int = 40,
        top_k: int = 3,
        embedder_kwargs: dict = None,
        generator_kwargs: dict = None,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k

        self.embedder = get_embedder(embedder_name, **(embedder_kwargs or {}))
        self.vector_store_name = vector_store_name
        self.vector_store = None  # built once we know embedding dim
        self.generator = get_generator(generator_name, **(generator_kwargs or {}))

        self.chunks: List[Chunk] = []

    # ---------- Indexing (Stages 1-4) ----------

    def index_document(self, path: str, source_label: str = None):
        """Load a document from disk, chunk it, embed it, and build the index."""
        text = load_document(path)
        return self.index_text(text, source_label=source_label or path)

    def index_text(self, text: str, source_label: str = "document"):
        """Chunk raw text, embed it, and build the vector index."""
        self.chunks = chunk_text(
            text, chunk_size=self.chunk_size, overlap=self.overlap, source=source_label
        )
        if not self.chunks:
            raise ValueError("No text extracted from the document -- is it empty or scanned/image-only?")

        texts = [c.text for c in self.chunks]
        embeddings = self.embedder.fit_transform(texts)

        dim = embeddings.shape[1]
        self.vector_store = get_vector_store(self.vector_store_name, dim=dim)
        self.vector_store.build(embeddings, self.chunks)

        return {"num_chunks": len(self.chunks), "embedding_dim": dim}

    # ---------- Querying (Stages 5-7) ----------

    def retrieve(self, question: str, top_k: int = None):
        """Stages 5 & 6: embed the query and fetch the most similar chunks."""
        if self.vector_store is None:
            raise RuntimeError("Call index_document()/index_text() before querying.")
        top_k = top_k or self.top_k
        query_embedding = self.embedder.transform([question])[0]
        return self.vector_store.search(query_embedding, top_k=top_k)

    def answer(self, question: str, top_k: int = None):
        """Full pipeline: retrieve context, then generate a grounded answer."""
        results = self.retrieve(question, top_k=top_k)
        context_chunks = [chunk for chunk, score in results]
        answer_text = self.generator.generate(question, context_chunks)
        return {
            "question": question,
            "answer": answer_text,
            "sources": [
                {"chunk_id": c.id, "score": round(score, 4), "text": c.text}
                for c, score in results
            ],
        }
