"""
chunker.py
----------
Stage 2 of the RAG pipeline: Text Chunking.

Splits raw document text into overlapping, word-bounded chunks.
Chunking matters a lot for retrieval quality:
  - Chunks too large  -> irrelevant text dilutes the embedding, retrieval gets fuzzy.
  - Chunks too small   -> loses context, answers become fragmented.
  - Overlap            -> prevents an answer-bearing sentence from being
                           split exactly at a chunk boundary and lost.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    id: int
    text: str
    source: str = "document"


def chunk_text(
    text: str,
    chunk_size: int = 200,
    overlap: int = 40,
    source: str = "document",
) -> List[Chunk]:
    """
    Split text into overlapping chunks measured in words.

    Args:
        text: raw document text
        chunk_size: number of words per chunk
        overlap: number of words shared between consecutive chunks
        source: label to attach to each chunk (e.g. filename)

    Returns:
        List of Chunk objects
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    chunk_id = 0
    step = chunk_size - overlap

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_str = " ".join(chunk_words).strip()
        if chunk_str:
            chunks.append(Chunk(id=chunk_id, text=chunk_str, source=source))
            chunk_id += 1
        if end == len(words):
            break
        start += step

    return chunks


if __name__ == "__main__":
    sample = "This is a test. " * 100
    result = chunk_text(sample, chunk_size=20, overlap=5)
    print(f"Produced {len(result)} chunks")
    print(result[0])
