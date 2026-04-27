"""
──────────────────────────────────────────────────────────────────────────────
VectorizerEngine  ·  config.py
──────────────────────────────────────────────────────────────────────────────
Single source of truth for every setting the application needs.

All values are read from environment variables (loaded from a .env file).
Import this module anywhere you need a setting – never read os.environ
directly in other modules.
──────────────────────────────────────────────────────────────────────────────
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env file from project root ─────────────────────────────────────────
load_dotenv(Path(__file__).parent / ".env")


# ── Helper ────────────────────────────────────────────────────────────────────
def _require(name: str) -> str:
    """Return an env-var value or crash early with a clear message."""
    value = os.getenv(name)
    if not value:
        raise EnvironmentError(f"Missing required environment variable: {name}")
    return value


# ── Watched Folder ────────────────────────────────────────────────────────────
# Optional: only used for local monitoring (main.py)
WATCH_FOLDER_PATH: str | None = os.getenv("WATCH_FOLDER_PATH")

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

# ── Embedding Provider ────────────────────────────────────────────────────────
EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "openai").lower()

OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
AZURE_OPENAI_API_KEY: str | None = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT: str | None = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str | None = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

# ── Chat Model (Agent) ────────────────────────────────────────────────────────
CHAT_PROVIDER: str = os.getenv("CHAT_PROVIDER", "azure_openai").lower()
AZURE_OPENAI_CHAT_DEPLOYMENT: str = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4")
AZURE_OPENAI_API_KEY_CHAT: str | None = os.getenv("AZURE_OPENAI_API_KEY_CHAT")
AZURE_OPENAI_ENDPOINT_CHAT: str | None = os.getenv("AZURE_OPENAI_ENDPOINT_CHAT")
OPENAI_CHAT_MODEL: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4-turbo")

# ── Qdrant ────────────────────────────────────────────────────────────────────
QDRANT_MODE: str = os.getenv("QDRANT_MODE", "local").lower()
QDRANT_LOCAL_PATH: str = os.getenv("QDRANT_LOCAL_PATH", "./qdrant_data")
QDRANT_URL: str | None = os.getenv("QDRANT_URL")
QDRANT_API_KEY: str | None = os.getenv("QDRANT_API_KEY")
# Connection timeout in seconds for remote Qdrant
QDRANT_TIMEOUT: float = float(os.getenv("QDRANT_TIMEOUT", "30.0"))
# Max retries for Qdrant operations
QDRANT_MAX_RETRIES: int = int(os.getenv("QDRANT_MAX_RETRIES", "3"))
# Retry delay in seconds (with exponential backoff)
QDRANT_RETRY_DELAY: float = float(os.getenv("QDRANT_RETRY_DELAY", "2.0"))

# ── Azure AI Search ──────────────────────────────────────────────────────────
AZURE_SEARCH_ENDPOINT: str | None = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY: str | None = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX_NAME: str = os.getenv("AZURE_SEARCH_INDEX_NAME", "genesis-projects")

# ── Vectorstore Targets ──────────────────────────────────────────────────────
# Which vectorstores to push to.  Comma-separated: "qdrant", "azure", or both.
VECTORSTORE_TARGETS: list[str] = [
    t.strip().lower()
    for t in os.getenv("VECTORSTORE_TARGETS", "qdrant").split(",")
    if t.strip()
]

# ── Processing ────────────────────────────────────────────────────────────────
MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "4"))
STABILITY_WAIT_SECONDS: int = int(os.getenv("STABILITY_WAIT_SECONDS", "10"))
# Max retries to check if files are present locally before giving up.
FILE_CHECK_MAX_RETRIES: int = int(os.getenv("FILE_CHECK_MAX_RETRIES", "30"))
# Seconds to wait between file presence checks.
FILE_CHECK_RETRY_DELAY: float = float(os.getenv("FILE_CHECK_RETRY_DELAY", "1.0"))
# Enable polling-based watching (recommended for network paths like \\server\share)
USE_POLLING_WATCHER: bool = os.getenv("USE_POLLING_WATCHER", "False").lower() in ("true", "1", "yes")
# Polling interval in seconds (how often to check for new folders on network path)
POLLING_INTERVAL_SECONDS: float = float(os.getenv("POLLING_INTERVAL_SECONDS", "10.0"))

# ── Embedding Batching ───────────────────────────────────────────────────────
# Number of texts to send per embedding API call.
EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "50"))
# Seconds to wait between embedding batches to respect rate limits.
EMBEDDING_BATCH_DELAY: float = float(os.getenv("EMBEDDING_BATCH_DELAY", "1.0"))

# ── Azure Blob Storage Watcher ───────────────────────────────────────────────
# Enable Azure Blob Storage watching (polls the container for new blobs)
USE_BLOB_WATCHER: bool = os.getenv("USE_BLOB_WATCHER", "False").lower() in ("true", "1", "yes")
# Azure Storage Connection String
AZURE_STORAGE_CONNECTION_STRING: str | None = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
# Azure Storage Container Name
AZURE_STORAGE_CONTAINER_NAME: str | None = os.getenv("AZURE_STORAGE_CONTAINER_NAME")

# ── N8N Webhook ───────────────────────────────────────────────────────────────
# URL of the N8N Webhook node that will be called after ingestion completes.
# Leave empty/unset to disable the callback.
N8N_WEBHOOK_URL: str | None = os.getenv("N8N_WEBHOOK_URL")

# ── Supported File Extensions ────────────────────────────────────────────────
# Map of file extension → loader type key used in loaders.py
SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".xlsx": "xlsx",
    ".txt": "txt",
}

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s  │  %(name)-22s  │  %(levelname)-7s  │  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
