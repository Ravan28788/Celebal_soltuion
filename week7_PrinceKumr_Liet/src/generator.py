"""
generator.py
------------
Stage 7 of the RAG pipeline: Answer Generation.

Three interchangeable backends, from simplest to most powerful:

  1. ExtractiveGenerator   - No model download, no API key, works instantly
                              offline. Stitches the retrieved chunks into a
                              readable answer. This is the DEFAULT so the
                              whole pipeline runs out of the box.

  2. HFGenerator           - Local open-source LLM (google/flan-t5-base)
                              via HuggingFace `transformers`. Needs:
                              pip install transformers torch
                              and internet access the first time to download
                              the model.

  3. AnthropicGenerator    - Calls the Claude API for high quality, grounded
                              answers. Needs: pip install anthropic
                              and an ANTHROPIC_API_KEY environment variable.

Swap backends with one line -> this is the "experiment with different
language models" experiment the assignment brief asks for.
"""

from abc import ABC, abstractmethod
from typing import List
from .chunker import Chunk


class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, question: str, context_chunks: List[Chunk]) -> str:
        ...


class ExtractiveGenerator(BaseGenerator):
    """
    Zero-dependency fallback: no LLM at all.
    Returns the most relevant retrieved sentences as a direct answer,
    which is a legitimate (if unpolished) RAG baseline.
    """

    def generate(self, question: str, context_chunks: List[Chunk]) -> str:
        if not context_chunks:
            return "I couldn't find relevant information in the document to answer that."

        best = context_chunks[0]
        answer = (
            f"Based on the document, here is the most relevant passage:\n\n"
            f"\"{best.text.strip()}\"\n\n"
            f"(source: {best.source}, chunk #{best.id})"
        )
        return answer


class HFGenerator(BaseGenerator):
    """Local, open-source text2text LLM."""

    def __init__(self, model_name: str = "google/flan-t5-base"):
        from transformers import pipeline
        self.pipe = pipeline("text2text-generation", model=model_name)

    def generate(self, question: str, context_chunks: List[Chunk]) -> str:
        context = "\n\n".join(c.text for c in context_chunks)
        prompt = (
            f"Answer the question using only the context below. "
            f"If the answer isn't in the context, say you don't know.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        result = self.pipe(prompt, max_new_tokens=200)
        return result[0]["generated_text"].strip()


class AnthropicGenerator(BaseGenerator):
    """Calls the Claude API for a high-quality, grounded answer."""

    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str = None):
        import os
        from anthropic import Anthropic
        self.client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model

    def generate(self, question: str, context_chunks: List[Chunk]) -> str:
        context = "\n\n".join(f"[Chunk {c.id}] {c.text}" for c in context_chunks)
        prompt = (
            "Answer the question using ONLY the context provided below. "
            "If the context doesn't contain the answer, say so explicitly.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}"
        )
        response = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()


def get_generator(name: str = "extractive", **kwargs) -> BaseGenerator:
    if name == "extractive":
        return ExtractiveGenerator()
    elif name == "huggingface":
        return HFGenerator(**kwargs)
    elif name == "anthropic":
        return AnthropicGenerator(**kwargs)
    else:
        raise ValueError(f"Unknown generator: {name}")
