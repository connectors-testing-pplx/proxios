"""
PMC fetcher — retrieves open-access papers from PubMed Central.

Uses NCBI E-utilities API (free, no API key required for <3 requests/sec).
Searches for peroxisome-related papers, filters to open-access with full text,
and downloads XML full text. Saves to data/papers/{pmc_id}.json.
"""
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import httpx

from config import settings

logger = logging.getLogger("proxios.ingestion")

# NCBI E-utilities base URLs
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

# Rate limit: 3 requests/second without an API key
RATE_LIMIT_INTERVAL = 1.0 / settings.pmc_rate_limit_rps


class PMCFetcher:
    """Fetches PMC open-access papers on peroxisome biology."""

    def __init__(self, output_dir: str = "./data/papers") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.client = httpx.Client(timeout=60.0)
        self._last_request_time = 0.0

    def _rate_limit(self) -> None:
        """Enforce NCBI's rate limit (3 req/sec without API key)."""
        elapsed = time.time() - self._last_request_time
        if elapsed < RATE_LIMIT_INTERVAL:
            time.sleep(RATE_LIMIT_INTERVAL - elapsed)
        self._last_request_time = time.time()

    def search_papers(self, max_results: int | None = None) -> list[str]:
        """
        Search PMC for peroxisome-related open-access papers.

        Returns a list of PMC IDs.
        """
        limit = max_results or settings.pmc_max_papers
        logger.info("Searching PMC for '%s' (max %d papers)", settings.pmc_search_term, limit)

        self._rate_limit()
        params = {
            "db": "pmc",
            "term": settings.pmc_search_term,
            "retmax": str(limit),
            "retmode": "json",
            "filter": "open_access[filter]",
        }
        resp = self.client.get(ESEARCH_URL, params=params)
        resp.raise_for_status()
        data = resp.json()

        pmc_ids = data.get("esearchresult", {}).get("idlist", [])
        logger.info("Found %d PMC papers", len(pmc_ids))
        return pmc_ids

    def fetch_summary(self, pmc_id: str) -> dict[str, Any]:
        """Fetch metadata summary for a single PMC ID."""
        self._rate_limit()
        params = {
            "db": "pmc",
            "id": pmc_id,
            "retmode": "json",
        }
        resp = self.client.get(ESUMMARY_URL, params=params)
        resp.raise_for_status()
        data = resp.json()
        return data.get("result", {}).get(pmc_id, {})

    def fetch_full_text(self, pmc_id: str) -> str:
        """Fetch the full-text XML for a PMC paper."""
        self._rate_limit()
        params = {
            "db": "pmc",
            "id": pmc_id,
            "rettype": "full",
            "retmode": "xml",
        }
        resp = self.client.get(EFETCH_URL, params=params)
        resp.raise_for_status()
        return resp.text

    def fetch_and_save(self, pmc_id: str) -> dict[str, Any] | None:
        """
        Fetch a single paper's metadata + full text and save to JSON.

        Returns the paper dict, or None on failure.
        """
        try:
            summary = self.fetch_summary(pmc_id)
            full_text_xml = self.fetch_full_text(pmc_id)

            paper_data = {
                "pmc_id": pmc_id,
                "title": summary.get("title", ""),
                "authors": [
                    a.get("name", "") for a in summary.get("authors", []) if a.get("name")
                ],
                "journal": summary.get("fulljournalname", ""),
                "year": summary.get("pubdate", "")[:4],
                "abstract": "",  # Extracted from XML in pdf_processor
                "full_text_xml": full_text_xml,
                "open_access": True,
                "license": summary.get("license", ""),
            }

            output_path = self.output_dir / f"{pmc_id}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(paper_data, f, ensure_ascii=False, indent=2)

            logger.info("Saved paper PMC%s to %s", pmc_id, output_path)
            return paper_data

        except Exception as exc:
            logger.error("Failed to fetch PMC%s: %s", pmc_id, exc)
            return None

    def fetch_corpus(self, max_results: int | None = None) -> list[dict[str, Any]]:
        """
        Fetch the full initial corpus of papers.

        Searches PMC, then fetches each paper with rate limiting.
        Saves each paper as a separate JSON file.
        """
        pmc_ids = self.search_papers(max_results)
        papers = []

        for i, pmc_id in enumerate(pmc_ids, 1):
            logger.info("Fetching paper %d/%d: PMC%s", i, len(pmc_ids), pmc_id)
            paper = self.fetch_and_save(pmc_id)
            if paper:
                papers.append(paper)

        logger.info("Fetched and saved %d/%d papers", len(papers), len(pmc_ids))
        return papers

    def close(self) -> None:
        """Clean up the HTTP client."""
        self.client.close()


def fetch_corpus_cli() -> None:
    """CLI entry point for fetching the initial PMC corpus."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    fetcher = PMCFetcher()
    try:
        fetcher.fetch_corpus()
    finally:
        fetcher.close()
