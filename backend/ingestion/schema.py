"""
Data schemas for ingested papers and chunks.

Defines the structured representation of a PMC paper and its text chunks
before they are embedded and stored in ChromaDB.
"""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PMCPaper:
    """Structured representation of a PMC open-access paper."""

    pmc_id: str
    pmid: str | None = None
    title: str = ""
    authors: list[str] = field(default_factory=list)
    journal: str = ""
    year: str = ""
    abstract: str = ""
    full_text: str = ""
    sections: list[dict[str, str]] = field(default_factory=list)
    figure_captions: list[str] = field(default_factory=list)
    open_access: bool = True
    license: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary for JSON storage."""
        return {
            "pmc_id": self.pmc_id,
            "pmid": self.pmid,
            "title": self.title,
            "authors": self.authors,
            "journal": self.journal,
            "year": self.year,
            "abstract": self.abstract,
            "full_text": self.full_text,
            "sections": self.sections,
            "figure_captions": self.figure_captions,
            "open_access": self.open_access,
            "license": self.license,
        }


@dataclass
class TextChunk:
    """A chunk of text extracted from a paper, ready for embedding."""

    text: str
    paper_title: str = ""
    pmc_id: str = ""
    authors: list[str] = field(default_factory=list)
    year: str = ""
    section: str = ""
    chunk_index: int = 0

    def to_metadata(self) -> dict[str, Any]:
        """Return metadata dict for ChromaDB storage."""
        return {
            "paper_title": self.paper_title,
            "pmc_id": self.pmc_id,
            "authors": ", ".join(self.authors) if self.authors else "",
            "year": self.year,
            "section": self.section,
            "chunk_index": self.chunk_index,
        }
