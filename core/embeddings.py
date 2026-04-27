"""
──────────────────────────────────────────────────────────────────────────────
VectorizerEngine  ·  core/embeddings.py
──────────────────────────────────────────────────────────────────────────────
Builds and caches the embedding model used across the application.

Supports two providers:
  • "openai"       → Uses OpenAI embeddings (text-embedding-ada-002, etc.)
  • "azure_openai" → Uses Azure-hosted OpenAI embeddings.

The embedding instance is created once and reused (module-level singleton).
──────────────────────────────────────────────────────────────────────────────
"""

import logging

from langchain_core.embeddings import Embeddings

import config

logger = logging.getLogger("embeddings")

# ── Module-level cache ────────────────────────────────────────────────────────
_cached_embeddings: Embeddings | None = None


def get_embeddings() -> Embeddings:
    """
    Return the configured embedding model, creating it on first call.

    Raises
    ------
    ValueError
        If the configured EMBEDDING_PROVIDER is not recognised.
    EnvironmentError
        If required API keys / endpoints are missing.
    """
    global _cached_embeddings

    if _cached_embeddings is not None:
        return _cached_embeddings

    provider = config.EMBEDDING_PROVIDER

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        if not config.OPENAI_API_KEY:
            raise EnvironmentError("OPENAI_API_KEY is required for provider 'openai'.")

        _cached_embeddings = OpenAIEmbeddings(
            openai_api_key=config.OPENAI_API_KEY,
        )
        logger.info("Embedding model ready  →  OpenAI (default model).")

    elif provider == "azure_openai":
        from langchain_openai import AzureOpenAIEmbeddings

        if not config.AZURE_OPENAI_API_KEY or not config.AZURE_OPENAI_ENDPOINT:
            raise EnvironmentError(
                "AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT are "
                "required for provider 'azure_openai'."
            )

        _cached_embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_key=config.AZURE_OPENAI_API_KEY,
            azure_deployment=config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            api_version=config.AZURE_OPENAI_API_VERSION,
        )
        logger.info("Embedding model ready  →  Azure OpenAI (%s).", config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT)

    else:
        raise ValueError(
            f"Unknown EMBEDDING_PROVIDER '{provider}'.  "
            f"Supported: 'openai', 'azure_openai'."
        )

    return _cached_embeddings
