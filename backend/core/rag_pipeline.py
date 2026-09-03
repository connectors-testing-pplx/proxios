"""
RAG pipeline — orchestrates retrieval, augmentation, and generation.

The pipeline:
  1. Embeds the query using sentence-transformers
  2. Retrieves top-k chunks from ChromaDB via MMR
  3. Formats retrieved chunks as context with source metadata
  4. Passes to the Claude client with the mode flag
  5. Streams the response back
  6. Returns the sources list alongside the answer
"""
import logging
from dataclasses import dataclass
from typing import AsyncIterator

from langchain_core.documents import Document

from config import settings
from core.claude_client import ClaudeClient
from core.vector_store import retrieve_mmr, retrieve_similarity

logger = logging.getLogger("proxios.rag")


@dataclass
class RetrievalResult:
    """Container for retrieval results."""

    context: str
    sources: list[dict]
    documents: list[Document]


class RAGPipeline:
    """End-to-end RAG pipeline connecting retrieval and generation."""

    def __init__(self) -> None:
        self.claude = ClaudeClient()
        logger.info("RAG pipeline initialized (model=%s)", settings.claude_model)

    async def retrieve(self, question: str) -> RetrievalResult:
        """
        Retrieve relevant document chunks and format them as context.

        Uses MMR (Maximum Marginal Relevance) by default to reduce redundant
        results. Falls back to similarity search if MMR fails.
        """
        try:
            docs = retrieve_mmr(question)
            logger.info("MMR retrieval returned %d documents for query: %s", len(docs), question[:80])
        except Exception as exc:
            logger.warning("MMR retrieval failed (%s), falling back to similarity search", exc)
            docs = retrieve_similarity(question)

        if not docs:
            logger.warning("No documents retrieved — vector store may be empty")
            return RetrievalResult(
                context="No relevant papers found in the corpus.",
                sources=[],
                documents=[],
            )

        context_text = self._format_context(docs)
        sources = self._extract_sources(docs)

        return RetrievalResult(
            context=context_text,
            sources=sources,
            documents=docs,
        )

    def _format_context(self, docs: list[Document]) -> str:
        """Format retrieved documents into a context string for Claude."""
        formatted = []
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            title = meta.get("paper_title", "Unknown title")
            pmc_id = meta.get("pmc_id", "Unknown PMC ID")
            authors = meta.get("authors", "")
            year = meta.get("year", "")
            section = meta.get("section", "")

            header = f"[{i}] {title}"
            if authors:
                header += f" — {authors}"
            if year:
                header += f" ({year})"
            if pmc_id:
                header += f" | PMC{pmc_id}"
            if section:
                header += f" | Section: {section}"

            formatted.append(f"{header}\n{doc.page_content}")

        return "\n\n---\n\n".join(formatted)

    def _extract_sources(self, docs: list[Document]) -> list[dict]:
        """Extract deduplicated source metadata from retrieved documents."""
        seen = set()
        sources = []
        for doc in docs:
            meta = doc.metadata
            pmc_id = meta.get("pmc_id", "")
            if pmc_id in seen:
                continue
            seen.add(pmc_id)
            sources.append({
                "title": meta.get("paper_title", "Unknown"),
                "authors": meta.get("authors", ""),
                "year": meta.get("year", ""),
                "pmc_id": pmc_id,
                "section": meta.get("section", ""),
            })
        return sources

    async def generate(
        self,
        context: str,
        question: str,
        mode: str,
    ) -> AsyncIterator[dict]:
        """
        Generate a response by streaming from Claude.

        Delegates to query_explore or query_deep_dive based on mode.
        """
        if mode == "explore":
            async for chunk in self.claude.query_explore(context, question):
                yield chunk
        elif mode == "deep_dive":
            async for chunk in self.claude.query_deep_dive(context, question):
                yield chunk
        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'explore' or 'deep_dive'.")

    async def generate_follow_ups(self, answer: str, question: str) -> list[str]:
        """Generate follow-up research questions (deep_dive mode only)."""
        return await self.claude.generate_follow_ups(answer, question)
