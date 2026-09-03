# PeroxiOS Changelog

## [0.1.0] - 2026-06-02 — Initial MVP

### Added
- Frontend: Single-page research interface with Explore/Deep Dive modes
- Frontend: PeroxiOS branding, PBD Project story section
- Backend: FastAPI skeleton with /api/query and /api/health endpoints
- Backend: LangChain RAG pipeline with ChromaDB vector store
- Backend: Claude API integration with prompt caching
- Backend: PMC open-access paper fetcher (NCBI E-utilities)
- Backend: PDF processor with PyMuPDF chunking
- Docs: Architecture diagram (Mermaid), data flow, token optimization notes
- Infra: GitHub repo structure, CI workflow scaffold

### Architecture Decisions
- Chose sentence-transformers all-MiniLM-L6-v2 for embeddings (open-weight, no API cost)
- Chose ChromaDB for vector store (lightweight, local-first, no cloud dependency at MVP)
- Chose FastAPI for backend (async, OpenAPI docs auto-generated at /docs)
- PMC-OA as initial corpus for clean copyright baseline
- Dual-corpus architecture planned for Phase 5 (public + PBD private)
