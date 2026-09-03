"""
PeroxiOS configuration settings.

Loads from environment variables or a .env file in the backend directory.
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings for PeroxiOS backend."""

    # Claude API
    anthropic_api_key: str = ""
    claude_model: str = "claude-opus-4-5"

    # Vector store
    chroma_persist_path: str = "./data/chroma"
    chroma_collection_public: str = "peroxios_public"
    chroma_collection_private: str = "peroxios_private"

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"

    # Token limits
    max_tokens_explore: int = 1024
    max_tokens_deep_dive: int = 4096

    # Retrieval
    top_k_retrieval: int = 8
    mmr_fetch_k: int = 20

    # Cache
    cache_ttl_seconds: int = 3600

    # PMC ingestion
    pmc_search_term: str = (
        "peroxisome OR peroxisomal biogenesis disorder OR PEX gene OR Zellweger"
    )
    pmc_max_papers: int = 500
    pmc_rate_limit_rps: float = 3.0

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
