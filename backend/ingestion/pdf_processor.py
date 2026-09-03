"""
PDF processor — extracts and chunks text from scientific papers.

Uses PyMuPDF (fitz) for PDF extraction and LangChain's RecursiveCharacterTextSplitter
for chunking. Extracts title, abstract, sections, and figure captions while
skipping the references section and figure image data.

For XML full text from PMC, uses a simple XML parser to extract structured sections.
"""
import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings
from ingestion.schema import PMCPaper, TextChunk

logger = logging.getLogger("proxios.ingestion")


class PDFProcessor:
    """Processes PDF and XML papers into embeddable text chunks."""

    def __init__(self) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def process_pdf(self, pdf_path: str | Path) -> PMCPaper:
        """
        Extract structured text from a PDF using PyMuPDF.

        Returns a PMCPaper with full text and sections.
        """
        import fitz  # PyMuPDF

        doc = fitz.open(str(pdf_path))
        full_text_parts = []
        sections = []

        for page in doc:
            text = page.get_text()
            full_text_parts.append(text)

            # Detect section headings (heuristic: short lines that are all caps or bold)
            for line in text.split("\n"):
                stripped = line.strip()
                if stripped and len(stripped) < 80 and stripped.isupper():
                    sections.append({"heading": stripped, "text": ""})

        full_text = "\n\n".join(full_text_parts)

        # Extract title (usually first non-empty line of page 1)
        title = full_text.split("\n")[0].strip() if full_text else ""
        abstract = self._extract_section(full_text, "ABSTRACT", "INTRODUCTION")
        figure_captions = self._extract_figure_captions(full_text)

        paper = PMCPaper(
            pmc_id=Path(pdf_path).stem,
            title=title,
            abstract=abstract,
            full_text=full_text,
            sections=sections,
            figure_captions=figure_captions,
        )

        doc.close()
        return paper

    def process_xml(self, xml_text: str, metadata: dict[str, Any]) -> PMCPaper:
        """
        Extract structured text from PMC XML full text.

        Parses the JATS XML format used by PubMed Central.
        """
        paper = PMCPaper(
            pmc_id=metadata.get("pmc_id", ""),
            title=metadata.get("title", ""),
            authors=metadata.get("authors", []),
            journal=metadata.get("journal", ""),
            year=metadata.get("year", ""),
            open_access=True,
            license=metadata.get("license", ""),
        )

        try:
            root = ET.fromstring(xml_text)

            # Extract abstract
            abstract_elem = root.find(".//abstract")
            if abstract_elem is not None:
                paper.abstract = self._extract_text_from_element(abstract_elem)

            # Extract body sections (skip references)
            body = root.find(".//body")
            if body is not None:
                full_text_parts = [paper.abstract]
                for sec in body.iter("sec"):
                    title_elem = sec.find("title")
                    sec_title = title_elem.text if title_elem is not None else ""
                    if sec_title and "reference" in sec_title.lower():
                        continue  # Skip references section
                    sec_text = self._extract_text_from_element(sec)
                    if sec_text:
                        full_text_parts.append(sec_text)
                        sections = paper.sections
                        sections.append({"heading": sec_title, "text": sec_text})

                paper.full_text = "\n\n".join(full_text_parts)

            # Extract figure captions
            for fig in root.iter("fig"):
                caption_elem = fig.find("caption")
                if caption_elem is not None:
                    caption = self._extract_text_from_element(caption_elem)
                    if caption:
                        paper.figure_captions.append(caption)

        except ET.ParseError as exc:
            logger.warning("XML parse error for PMC%s: %s", paper.pmc_id, exc)
            paper.full_text = paper.abstract or ""

        return paper

    def chunk_paper(self, paper: PMCPaper) -> list[TextChunk]:
        """
        Split a paper into chunks using RecursiveCharacterTextSplitter.

        Each chunk carries metadata: paper_title, pmc_id, authors, year, section.
        """
        chunks = []

        # Chunk the abstract
        if paper.abstract:
            abstract_chunks = self.splitter.split_text(paper.abstract)
            for i, chunk_text in enumerate(abstract_chunks):
                chunks.append(TextChunk(
                    text=chunk_text,
                    paper_title=paper.title,
                    pmc_id=paper.pmc_id,
                    authors=paper.authors,
                    year=paper.year,
                    section="abstract",
                    chunk_index=i,
                ))

        # Chunk each section
        for section in paper.sections:
            heading = section.get("heading", "unknown")
            section_text = section.get("text", "")
            if not section_text:
                continue
            # Skip references section
            if heading and "reference" in heading.lower():
                continue

            section_chunks = self.splitter.split_text(section_text)
            for i, chunk_text in enumerate(section_chunks):
                chunks.append(TextChunk(
                    text=chunk_text,
                    paper_title=paper.title,
                    pmc_id=paper.pmc_id,
                    authors=paper.authors,
                    year=paper.year,
                    section=heading,
                    chunk_index=i,
                ))

        # If no sections were parsed, chunk the full text
        if not chunks and paper.full_text:
            full_chunks = self.splitter.split_text(paper.full_text)
            for i, chunk_text in enumerate(full_chunks):
                chunks.append(TextChunk(
                    text=chunk_text,
                    paper_title=paper.title,
                    pmc_id=paper.pmc_id,
                    authors=paper.authors,
                    year=paper.year,
                    section="full_text",
                    chunk_index=i,
                ))

        logger.info("Chunked PMC%s into %d chunks", paper.pmc_id, len(chunks))
        return chunks

    def _extract_text_from_element(self, elem: ET.Element) -> str:
        """Recursively extract all text from an XML element."""
        texts = []
        for node in elem.iter():
            if node.tag in ("ext-link", "xref", "disp-formula"):
                continue  # Skip links, cross-refs, and formulas
            if node.text:
                texts.append(node.text.strip())
            if node.tail:
                texts.append(node.tail.strip())
        return " ".join(t for t in texts if t)

    def _extract_section(self, full_text: str, start_marker: str, end_marker: str) -> str:
        """Extract a section of text between two markers (case-insensitive)."""
        pattern = re.compile(
            rf"{start_marker}.*?{end_marker}",
            re.DOTALL | re.IGNORECASE,
        )
        match = pattern.search(full_text)
        if match:
            return match.group()[len(start_marker):].strip()
        return ""

    def _extract_figure_captions(self, full_text: str) -> list[str]:
        """Extract figure captions from full text."""
        captions = []
        pattern = re.compile(r"(?:Figure|Fig\.?)\s*\d+[:\.]?\s*(.+?)(?=\n\n|\Z)", re.DOTALL)
        for match in pattern.finditer(full_text):
            captions.append(match.group(1).strip().rstrip("."))
        return captions
