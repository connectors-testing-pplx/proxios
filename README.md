# PeroxiOS

> The operating system for peroxisome science.

PeroxiOS is an open-access, AI-powered research intelligence platform built on peroxisome biology literature. Ask questions. Get cited, synthesized answers. Discover connections across thousands of papers.

**A [PBD Project](https://pbdproject.org) initiative.**

## What it does

- Natural language queries over a curated corpus of peroxisome research papers
- Two modes: **Explore** (plain language) and **Deep Dive** (expert reasoning)
- Every answer cites its source papers (PMC IDs)
- Built on PMC open-access literature — legally redistributable, continuously growing

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Static HTML/CSS/JS |
| Backend | Python, FastAPI |
| Orchestration | LangChain |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector DB | ChromaDB |
| AI | Anthropic Claude (claude-opus-4-5) with prompt caching |
| Paper Source | PMC Open-Access Subset |

## Quick Start

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env  # add ANTHROPIC_API_KEY
python -m uvicorn app:app --reload
```

Open `frontend/index.html` in a browser, or serve with any static file server.

## Architecture

See [docs/architecture.md](docs/architecture.md) for system diagrams, data flow, and token optimization strategy.

## Roadmap

- [x] Phase 0: Repo structure
- [x] Phase 1: Frontend MVP
- [x] Phase 2: Backend RAG pipeline
- [ ] Phase 3: PMC corpus ingestion (~500 papers)
- [ ] Phase 4: Deploy to Vercel (frontend) + Fly.io or Railway (backend)
- [ ] Phase 5: Closed corpus for PBD Project internal use

## About PBD Project

PBD Project is a nonprofit advancing peroxisome science. Founded by Andrew Longenecker after his son Diego's diagnosis with PEX10 Zellweger Spectrum Disorder. [pbdproject.org](https://pbdproject.org)

## License

MIT License — open source. Paper corpus subject to PMC-OA license terms.
