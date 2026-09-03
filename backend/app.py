"""
PeroxiOS FastAPI application entrypoint.

Exposes:
  - GET  /api/health   — health check
  - POST /api/query    — RAG-powered query with SSE streaming
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import health, query


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup, clean up on shutdown."""
    # The RAG pipeline initializes lazily on first query to keep startup fast.
    yield


app = FastAPI(
    title="PeroxiOS",
    description="The operating system for peroxisome science. AI-powered research intelligence.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow the static frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(query.router, prefix="/api", tags=["query"])


@app.get("/")
async def root():
    """Root endpoint redirecting to docs."""
    return {
        "name": "PeroxiOS",
        "version": "0.1.0",
        "docs": "/docs",
        "endpoints": ["/api/health", "/api/query"],
    }
