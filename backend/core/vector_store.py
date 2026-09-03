"""
Vector store module — ChromaDB-backed vector storage and retrieval.

Provides a thin wrapper around langchain_community.vectorstores.Chroma for:
  - Persisting document chunks with metadata
  - MMR (Maximum Marginal Relevance) retrieval to reduce redundant results
  - Dual-collection support (public / private) for future closed-system expansion
"""
from typing import Any

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from config import settings
from core.embeddings import get_embeddings


def get_vector_store(collection: str | None = None) -> Chroma:
    """
    Return a Chroma vector store for the given collection name.

    Defaults to the public collection. Private collections require an
    X-PBD-Token header (enforced at the API layer, not here).
    """
    collection_name = collection or settings.chroma_collection_public
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_persist_path,
    )


def add_documents(docs: list[Document], collection: str | None = None) -> None:
    """Add a list of LangChain Documents to the vector store with persistence."""
    store = get_vector_store(collection)
    store.add_documents(docs)
    store.persist()


def retrieve_mmr(
    query: str,
    k: int | None = None,
    fetch_k: int | None = None,
    collection: str | None = None,
) -> list[Document]:
    """
    Retrieve documents using Maximum Marginal Relevance.

    MMR balances relevance and diversity: it fetches a larger pool (fetch_k)
    then selects k results that are both relevant and non-redundant.

    Args:
        query: The search query text.
        k: Number of results to return (default: settings.top_k_retrieval).
        fetch_k: Pool size for MMR selection (default: settings.mmr_fetch_k).
        collection: Collection name (default: public).
    """
    store = get_vector_store(collection)
    return store.max_marginal_relevance_search(
        query=query,
        k=k or settings.top_k_retrieval,
        fetch_k=fetch_k or settings.mmr_fetch_k,
    )


def retrieve_similarity(
    query: str,
    k: int | None = None,
    collection: str | None = None,
) -> list[Document]:
    """Standard similarity search fallback."""
    store = get_vector_store(collection)
    return store.similarity_search(
        query=query,
        k=k or settings.top_k_retrieval,
    )


def get_collection_stats(collection: str | None = None) -> dict[str, Any]:
    """Return basic statistics about a collection (count, metadata)."""
    store = get_vector_store(collection)
    collection_obj = store._collection
    count = collection_obj.count() if hasattr(collection_obj, "count") else 0
    return {
        "collection": collection or settings.chroma_collection_public,
        "document_count": count,
        "persist_path": settings.chroma_persist_path,
    }
