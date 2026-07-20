"""
evaluate.py
-----------
Lightweight retrieval evaluation.

Since RAG's core promise is "retrieve the right context", the most useful
beginner-friendly metric is Hit Rate @ K: given a set of (question, expected
keyword) pairs, did the retrieved chunks actually contain that keyword?

This is a stand-in for the more rigorous evaluation used in production RAG
systems (e.g. using an LLM-as-judge, or Recall/MRR against a labeled
question-answer dataset such as vectara/open_ragbench).
"""

from typing import List, Tuple
from .rag_pipeline import RagPipeline


def hit_rate_at_k(pipeline: RagPipeline, qa_pairs: List[Tuple[str, str]], k: int = 3) -> dict:
    """
    Args:
        pipeline: an already-indexed RagPipeline
        qa_pairs: list of (question, expected_keyword_or_phrase) tuples
        k: how many chunks to retrieve per question

    Returns:
        dict with per-question hits and the overall hit rate
    """
    results = []
    hits = 0

    for question, expected_keyword in qa_pairs:
        retrieved = pipeline.retrieve(question, top_k=k)
        combined_text = " ".join(chunk.text.lower() for chunk, _ in retrieved)
        hit = expected_keyword.lower() in combined_text
        hits += int(hit)
        results.append({
            "question": question,
            "expected_keyword": expected_keyword,
            "hit": hit,
            "top_score": round(retrieved[0][1], 4) if retrieved else 0.0,
        })

    return {
        "hit_rate": round(hits / len(qa_pairs), 3) if qa_pairs else 0.0,
        "num_questions": len(qa_pairs),
        "details": results,
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.rag_pipeline import RagPipeline

    pipeline = RagPipeline(top_k=3)
    pipeline.index_document("sample_docs/sample.txt")

    qa_pairs = [
        ("What are the four stages of RAG?", "chunk"),
        ("What is hybrid search?", "BM25"),
        ("Why is RAG better than fine-tuning?", "hallucination"),
        ("What is re-ranking?", "re-scores"),
    ]

    report = hit_rate_at_k(pipeline, qa_pairs, k=3)
    print(f"Hit Rate @3: {report['hit_rate']}  ({report['num_questions']} questions)")
    for d in report["details"]:
        status = "PASS" if d["hit"] else "FAIL"
        print(f"  [{status}] {d['question']}  (top score={d['top_score']})")
