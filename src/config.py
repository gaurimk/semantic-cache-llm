"""
config.py
---------
Central place for every setting the app needs.

WHY THIS FILE EXISTS (non-technical explanation):
Instead of hunting through the code to change a number (like "how similar
do two questions need to be to count as the same question?"), every
adjustable setting lives here, and each one can also be overridden by
creating a `.env` file (copy `.env.example` to `.env` and edit it).

Nothing in this file costs money. Every default points at a local,
free tool:
  - Ollama running on your own machine (free LLM)
  - A free, local sentence-embedding model (downloaded once, then offline)
  - A local ChromaDB folder on disk (free, no server, no account)
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load variables from a .env file if one exists (does nothing if it doesn't)
load_dotenv()


def _get_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _get_float(name: str, default: float) -> float:
    val = os.getenv(name)
    return float(val) if val else default


def _get_int(name: str, default: int) -> int:
    val = os.getenv(name)
    return int(val) if val else default


@dataclass
class Settings:
    # --- Ollama (free, local LLM server) -----------------------------
    # Ollama exposes an OpenAI-compatible endpoint, so our proxy can
    # forward requests to it using the same request/response shape that
    # apps normally send to OpenAI. Install from https://ollama.com (free).
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    )
    default_model: str = field(
        default_factory=lambda: os.getenv("DEFAULT_MODEL", "llama3.2:1b")
    )

    # --- Embeddings (free, local, runs on CPU) ------------------------
    # sentence-transformers downloads a small open-source model the first
    # time you run the app, then reuses it from a local cache — no API
    # key, no per-call cost, works fully offline afterwards.
    embedding_model_name: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2"
        )
    )

    # --- Vector store (free, local, file-based) -----------------------
    # ChromaDB stores its data as files on disk in this folder — no
    # separate database server to install or pay for.
    chroma_persist_dir: str = field(
        default_factory=lambda: os.getenv("CHROMA_PERSIST_DIR", "./cache_data")
    )
    collection_name: str = field(
        default_factory=lambda: os.getenv("COLLECTION_NAME", "llm_semantic_cache")
    )

    # --- Cache policy ---------------------------------------------------
    similarity_threshold: float = field(
        default_factory=lambda: _get_float("SIMILARITY_THRESHOLD", 0.95)
    )
    near_miss_margin: float = field(
        default_factory=lambda: _get_float("NEAR_MISS_MARGIN", 0.10)
    )
    default_ttl_seconds: int = field(
        default_factory=lambda: _get_int("DEFAULT_TTL_SECONDS", 24 * 60 * 60)  # 24h
    )
    short_ttl_seconds: int = field(
        default_factory=lambda: _get_int("SHORT_TTL_SECONDS", 60 * 60)  # 1h
    )

    # --- Misc -------------------------------------------------------
    log_dir: str = field(default_factory=lambda: os.getenv("LOG_DIR", "./logs"))
    enable_metrics: bool = field(
        default_factory=lambda: _get_bool("ENABLE_METRICS", True)
    )
    # Rough per-1K-token cost estimate used ONLY to show illustrative
    # "money saved" numbers in /stats. Set to 0 if you don't want this.
    estimated_cost_per_1k_tokens_usd: float = field(
        default_factory=lambda: _get_float("ESTIMATED_COST_PER_1K_TOKENS_USD", 0.002)
    )


settings = Settings()
