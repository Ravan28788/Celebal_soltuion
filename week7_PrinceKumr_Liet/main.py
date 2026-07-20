#!/usr/bin/env python3
"""
main.py
-------
Command-line interface for the RAG Document Question Answering system.

Usage:
    # One-shot: index a document and ask a single question
    python main.py --doc sample_docs/sample.txt --question "What is RAG?"

    # Interactive mode: index once, ask multiple questions
    python main.py --doc sample_docs/sample.txt --interactive

    # Choose different backends
    python main.py --doc notes.pdf --question "..." \
        --embedder tfidf --generator extractive --top-k 5

    # Persist / reuse an index instead of rebuilding every run
    python main.py --doc notes.pdf --save-index outputs/index.pkl
    python main.py --load-index outputs/index.pkl --question "..."
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from src.rag_pipeline import RagPipeline
from src.vector_store import NumpyVectorStore


def build_arg_parser():
    p = argparse.ArgumentParser(description="RAG Document Question Answering CLI")
    p.add_argument("--doc", type=str, help="Path to a .pdf, .txt, or .md file to index")
    p.add_argument("--question", type=str, help="Question to ask about the document")
    p.add_argument("--interactive", action="store_true", help="Ask multiple questions in a loop")
    p.add_argument("--embedder", type=str, default="tfidf",
                    choices=["tfidf", "sentence-transformer"], help="Embedding backend")
    p.add_argument("--generator", type=str, default="extractive",
                    choices=["extractive", "huggingface", "anthropic"], help="Answer generation backend")
    p.add_argument("--chunk-size", type=int, default=200, help="Words per chunk")
    p.add_argument("--overlap", type=int, default=40, help="Overlapping words between chunks")
    p.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve per query")
    p.add_argument("--save-index", type=str, help="Path to save the built vector index (pickle)")
    p.add_argument("--load-index", type=str, help="Path to load a previously saved vector index")
    return p


def print_result(result):
    print("\n" + "=" * 70)
    print(f"Q: {result['question']}")
    print("-" * 70)
    print(f"A: {result['answer']}")
    print("-" * 70)
    print("Top retrieved chunks:")
    for s in result["sources"]:
        preview = s["text"][:120].replace("\n", " ")
        print(f"  [chunk {s['chunk_id']}, score={s['score']}] {preview}...")
    print("=" * 70 + "\n")


def main():
    args = build_arg_parser().parse_args()

    pipeline = RagPipeline(
        embedder_name=args.embedder,
        generator_name=args.generator,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        top_k=args.top_k,
    )

    if args.load_index:
        print(f"Loading index from {args.load_index} ...")
        store = NumpyVectorStore().load(args.load_index)
        pipeline.vector_store = store
        pipeline.chunks = store.chunks
        # NOTE: TF-IDF vectorizer must be refit on the same corpus to transform
        # new queries consistently. Re-fit on the loaded chunk texts.
        pipeline.embedder.fit([c.text for c in store.chunks])
    elif args.doc:
        print(f"Indexing {args.doc} ...")
        stats = pipeline.index_document(args.doc)
        print(f"Indexed {stats['num_chunks']} chunks (embedding dim={stats['embedding_dim']})")
        if args.save_index:
            pipeline.vector_store.save(args.save_index)
            print(f"Saved index to {args.save_index}")
    else:
        print("Error: provide --doc to index a new document or --load-index to reuse one.")
        sys.exit(1)

    if args.interactive:
        print("\nEnter questions (type 'exit' or 'quit' to stop):")
        while True:
            try:
                q = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in ("exit", "quit", ""):
                break
            result = pipeline.answer(q)
            print_result(result)
    elif args.question:
        result = pipeline.answer(args.question)
        print_result(result)
    else:
        print("Provide --question for a single query or --interactive for a Q&A loop.")


if __name__ == "__main__":
    main()
