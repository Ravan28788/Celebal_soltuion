# RAG Document Question Answering

A Retrieval Augmented Generation (RAG) system that give the appropriate answer with given data like format from pdf,docx not the model's memorized training data.

Runs fully offline out of the box (no API key, no model download required),
with pluggable, swappable components at every stage so you can experiment
with better embeddings and LLMs once the basic pipeline is working.

## Why RAG?

A language model only knows what it saw during training. RAG fixes that by:
1. Retrieving the most relevant chunks of *your* document for a question
2. Augmenting the model's prompt with that retrieved context
3. Generating an answer grounded in the retrieved text, not guesswork

This is how most real-world "chat with your docs" products work.

## Architecture

```
 PDF / TXT file
      │
      ▼
┌─────────────────┐   1. Document Ingestion   (document_loader.py)
│  Raw text        │
└─────────────────┘
      │
      ▼
┌─────────────────┐   2. Text Chunking        (chunker.py)
│  Overlapping      │
│  text chunks      │
└─────────────────┘
      │
      ▼
┌─────────────────┐   3. Embedding Creation   (embedder.py)
│  Vector per chunk  │
└─────────────────┘
      │
      ▼
┌─────────────────┐   4. Vector Database      (vector_store.py)
│  Similarity index │
└─────────────────┘
      │
   user question
      │
      ▼
┌─────────────────┐   5. Query Processing     (embedder.py)
│  Question vector  │
└─────────────────┘
      │
      ▼
┌─────────────────┐   6. Context Retrieval    (vector_store.py)
│  Top-K chunks      │
└─────────────────┘
      │
      ▼
┌─────────────────┐   7. Answer Generation    (generator.py)
│  Final answer      │
└─────────────────┘
```

`rag_pipeline.py`'s `RagPipeline` class wires all seven stages into one object.

## Project structure

```
rag_project/
├── main.py                 # CLI entry point
├── requirements.txt
├── src/
│   ├── document_loader.py  # Stage 1: PDF/TXT ingestion
│   ├── chunker.py          # Stage 2: text chunking
│   ├── embedder.py         # Stage 3: TF-IDF / sentence-transformer embeddings
│   ├── vector_store.py     # Stage 4: numpy / FAISS similarity search
│   ├── generator.py        # Stage 7: extractive / HF / Claude answer generation
│   ├── rag_pipeline.py     # Wires stages 1-7 together
│   └── evaluate.py         # Simple Hit Rate @ K retrieval evaluation
├── notebooks/
│   └── demo.ipynb          # Step-by-step walkthrough, cell by cell
├── sample_docs/
│   └── sample.txt          # Example document to test with
└── outputs/                # Saved vector indexes land here
```

## Installation

```bash
pip install -r requirements.txt
```

That installs everything needed for the **default, fully offline** pipeline
(`numpy`, `scikit-learn`, `pypdf`). The optional backends below need extra
packages, listed (commented out) in `requirements.txt`.

## Quick start (CLI)

```bash
# Ask a single question about a document
python main.py --doc sample_docs/sample.txt --question "What are the four main stages of a RAG pipeline?"

# Ask multiple questions interactively
python main.py --doc sample_docs/sample.txt --interactive

# Use your own PDF
python main.py --doc /path/to/your/resume.pdf --question "What projects has this person worked on?"

# Save the index so you don't have to re-embed every run
python main.py --doc sample_docs/sample.txt --save-index outputs/index.pkl
python main.py --load-index outputs/index.pkl --question "..."
```

### CLI options

| Flag | Default | Description |
|---|---|---|
| `--doc` | — | Path to a `.pdf`, `.txt`, or `.md` file to index |
| `--question` | — | Single question to ask |
| `--interactive` | off | Ask multiple questions in a loop |
| `--embedder` | `tfidf` | `tfidf` or `sentence-transformer` |
| `--generator` | `extractive` | `extractive`, `huggingface`, or `anthropic` |
| `--chunk-size` | `200` | Words per chunk |
| `--overlap` | `40` | Overlapping words between consecutive chunks |
| `--top-k` | `3` | Number of chunks retrieved per query |
| `--save-index` | — | Path to persist the built vector index |
| `--load-index` | — | Path to reuse a previously saved index |

## Quick start (Python)

```python
from src.rag_pipeline import RagPipeline

pipeline = RagPipeline()
pipeline.index_document("sample_docs/sample.txt")

result = pipeline.answer("What is hybrid search?")
print(result["answer"])
print(result["sources"])  # which chunks the answer was grounded in
```

## Notebook walkthrough

`notebooks/demo.ipynb` runs every stage in its own cell — ingestion, chunking,
embedding, indexing, retrieval, generation, and a small evaluation — so you
can see exactly what's happening at each step. It runs top-to-bottom with no
setup beyond `pip install -r requirements.txt`.

## Swapping in stronger components

The default backends (`tfidf` embeddings + `extractive` generation) are
intentionally simple: no downloads, no API keys, runs anywhere. Once the
pipeline works end-to-end, upgrade piece by piece:

**Better embeddings** (semantic, not just keyword overlap):
```bash
pip install sentence-transformers
python main.py --doc notes.pdf --question "..." --embedder sentence-transformer
```

**Local open-source LLM** for fluent, synthesized answers instead of an
extracted passage:
```bash
pip install transformers torch
python main.py --doc notes.pdf --question "..." --generator huggingface
```

**Claude API** for the highest-quality grounded answers:
```bash
pip install anthropic
export ANTHROPIC_API_KEY=your_key_here
python main.py --doc notes.pdf --question "..." --generator anthropic
```

## Evaluation

```bash
python -m src.evaluate
```

Runs a small Hit Rate @ K check: for a list of (question, expected keyword)
pairs, did the retrieved chunks actually contain that keyword? This is a
lightweight stand-in for more rigorous retrieval evaluation (Recall@K, MRR,
LLM-as-judge) against a labeled dataset such as
[vectara/open_ragbench](https://huggingface.co/datasets/vectara/open_ragbench).

## Possible next steps

- **Hybrid search**: combine this project's vector search with keyword search
  (e.g. BM25 via `rank_bm25`) and merge the two ranked lists.
- **Re-ranking**: after retrieving the top ~10 chunks by embedding similarity,
  re-score them with a cross-encoder before picking the final top-K.
- **Chunking strategies**: try sentence-aware or semantic chunking instead of
  fixed word windows.
- **Bigger vector store**: swap `NumpyVectorStore` for `FaissVectorStore`
  (already implemented in `vector_store.py`) once a document collection grows
  past what brute-force cosine similarity handles comfortably.
- **Multi-document support**: index a folder of PDFs instead of one file, and
  track per-document source labels (already threaded through `Chunk.source`).

## Key learnings this project demonstrates

- How retrieval and generation combine to ground LLM answers in real data
- Why chunk size and overlap affect retrieval quality
- How embeddings and vector similarity search work under the hood
- How to design a pipeline with swappable, testable components
- How to evaluate retrieval quality with a simple, interpretable metric
