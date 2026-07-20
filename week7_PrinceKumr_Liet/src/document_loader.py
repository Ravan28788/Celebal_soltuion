"""
document_loader.py
-------------------
Stage 1 of the RAG pipeline: Document Ingestion.

Loads raw text out of PDFs or .txt files so the rest of the
pipeline never has to care what format the source document was in.
"""

from pathlib import Path
from pypdf import PdfReader


def load_txt(path: str) -> str:
    """Read a plain text file and return its contents as a string."""
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def load_pdf(path: str) -> str:
    """Extract and concatenate text from every page of a PDF."""
    reader = PdfReader(path)
    pages_text = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages_text.append(text)
    return "\n".join(pages_text)


def load_document(path: str) -> str:
    """
    Dispatch on file extension and return the document's raw text.

    Supported: .pdf, .txt, .md
    """
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return load_pdf(path)
    elif ext in (".txt", ".md"):
        return load_txt(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .pdf, .txt, or .md")


if __name__ == "__main__":
    import sys
    text = load_document(sys.argv[1])
    print(f"Loaded {len(text)} characters.")
    print(text[:500])
