"""
Embeddings module — wraps sentence-transformers in a LangChain HuggingFaceEmbeddings instance.

Uses the all-MiniLM-L6-v2 model (open-weight, free, no API cost). This mirrors the
Nemotron-inspired approach from Adam Freygang's ARIA platform: lightweight, local
embeddings that run without external API calls.
"""
import threading
from typing import Any

from langchain_community.embeddings import HuggingFaceEmbeddings

from config import settings

_model: HuggingFaceEmbeddings | None = None
_lock = threading.Lock()


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Return a singleton HuggingFaceEmbeddings instance.

    The model is loaded once and cached. Thread-safe via a lock so concurrent
    requests don't trigger duplicate model downloads.
    """
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                _model = HuggingFaceEmbeddings(
                    model_name=settings.embedding_model,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                )
    return _model


def embed_query(text: str) -> Any:
    """Embed a single query string and return the embedding vector."""
    return get_embeddings().embed_query(text)


def embed_documents(documents: list[str]) -> Any:
    """Embed a list of document texts and return the embedding vectors."""
    return get_embeddings().embed_documents(documents)
