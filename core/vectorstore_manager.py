"""
──────────────────────────────────────────────────────────────────────────────
VectorizerEngine  ·  core/vectorstore_manager.py
──────────────────────────────────────────────────────────────────────────────
Unified interface for pushing document chunks into one or more vector
stores (Qdrant, Azure AI Search).

Design choices
──────────────
• Each target is handled by a small, focused function.
• `push_to_all_targets()` fans out to every enabled target in sequence so
  errors in one target don't prevent the others from succeeding.
• Qdrant  → one collection per project (isolated).
• Azure   → one shared index; isolation via `project_name` metadata field.
──────────────────────────────────────────────────────────────────────────────
"""

import logging
import re
import uuid
import time 
import sys

from langchain_core.documents import Document
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

import config
from core.embeddings import get_embeddings
from qdrant import (
    QdrantCloudModule, 
    QdrantLocalModule, 
    sanitise_collection_name,
    verify_collection_has_vectors
)
from core.utils import format_bytes

logger = logging.getLogger("vectorstore_manager")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sanitise_collection_name(name: str) -> str:
    """
    Turn a human-readable project name into a safe Qdrant collection name.

    Rules: lowercase, spaces/dashes → underscores, strip non-alphanumerics.
    """
    name = name.lower()
    name = re.sub(r"[\s\-]+", "_", name)
    name = re.sub(r"[^a-z0-9_]", "", name)
    name = name.strip("_")
    return name


# ── Per-target push functions ─────────────────────────────────────────────────

def _push_to_qdrant(chunks: list[Document], project_name: str) -> None:
    """
    Push chunks to Qdrant. Uses QdrantCloudModule for cloud or 
    QdrantLocalModule for local storage based on config.QDRANT_MODE.
    """
    collection = sanitise_collection_name(project_name)
    
    # Calculate content size for logging
    content_size = sum(len(chunk.page_content.encode('utf-8')) for chunk in chunks)

    logger.info(
        "Qdrant  │  collection='%s'  │  mode='%s'  │  chunks=%d  │  size=%s",
        collection,
        config.QDRANT_MODE,
        len(chunks),
        format_bytes(content_size) if chunks else "0 B"
    )

    if config.QDRANT_MODE == "cloud":
        # Cloud Module uses Qdrant Cloud Inference (no local embeddings needed)
        qdrant_cloud = QdrantCloudModule()
        qdrant_cloud.create_collection(collection_name=collection)
        qdrant_cloud.upsert_documents(collection_name=collection, chunks=chunks)
    else:
        # Local Module uses local storage and local embeddings
        embeddings = get_embeddings()
        qdrant_local = QdrantLocalModule()
        qdrant_local.create_collection(collection_name=collection)
        qdrant_local.upsert_documents(collection_name=collection, chunks=chunks, embeddings=embeddings)

    # Note: Verification is now handled inside the upsert_documents methods of the modules


def _embed_in_batches(
    texts: list[str],
    embeddings_model,
    batch_size: int,
    batch_delay: float,
    max_retries: int = 5,
) -> list[list[float]]:
    """
    Embed *texts* in small batches to respect API rate limits.

    Parameters
    ----------
    texts : list[str]
        All text strings to embed.
    embeddings_model
        A LangChain Embeddings instance.
    batch_size : int
        How many texts to send per API call.
    batch_delay : float
        Seconds to sleep between successful batches.
    max_retries : int
        How many times to retry a single batch on a 429 error.

    Returns
    -------
    list[list[float]]
        The embedding vectors, in the same order as *texts*.
    """
    all_vectors: list[list[float]] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for batch_idx in range(0, len(texts), batch_size):
        batch_texts = texts[batch_idx : batch_idx + batch_size]
        batch_num = (batch_idx // batch_size) + 1

        # ── Retry loop with exponential backoff for rate limits ───────────
        for attempt in range(1, max_retries + 1):
            try:
                vectors = embeddings_model.embed_documents(batch_texts)
                all_vectors.extend(vectors)
                logger.info(
                    "Azure   │  Embedded batch %d/%d  (%d texts)",
                    batch_num,
                    total_batches,
                    len(batch_texts),
                )
                break  # Success – move to the next batch.

            except Exception as exc:
                error_str = str(exc)
                is_rate_limit = "429" in error_str or "RateLimit" in error_str

                if is_rate_limit and attempt < max_retries:
                    wait = min(2 ** attempt, 60)  # 2s, 4s, 8s, 16s, 32s (capped at 60)
                    logger.warning(
                        "Azure   │  Rate limited on batch %d/%d  │  "
                        "retry %d/%d in %ds",
                        batch_num,
                        total_batches,
                        attempt,
                        max_retries,
                        wait,
                    )
                    time.sleep(wait)
                else:
                    # Non-rate-limit error or exhausted retries – propagate.
                    raise

        # Pause between batches to stay under the rate limit.
        if batch_idx + batch_size < len(texts):
            time.sleep(batch_delay)

    return all_vectors


def _push_to_azure(chunks: list[Document], project_name: str) -> None:
    """
    Push chunks into the shared Azure AI Search index.

    Uses the Azure SDK directly (not LangChain's AzureSearch wrapper)
    because the index defines `metadata` as an Edm.ComplexType:

        metadata: {
            source: string,
            attributes: [ { key: string, value: string }, … ]
        }

    Embeddings are generated in small batches with delays to avoid
    hitting Azure OpenAI rate limits (429).
    """
    embeddings_model = get_embeddings()
    index = config.AZURE_SEARCH_INDEX_NAME
    
    # Calculate content size
    content_size = sum(len(chunk.page_content.encode('utf-8')) for chunk in chunks)

    logger.info(
        "Azure   │  index='%s'  │  project='%s'  │  chunks=%d  │  content_size=%s",
        index,
        project_name,
        len(chunks),
        format_bytes(content_size) if chunks else "0 B"
    )

    # ── Step 1: Embed in batches ──────────────────────────────────────────
    texts = [chunk.page_content for chunk in chunks]
    vectors = _embed_in_batches(
        texts=texts,
        embeddings_model=embeddings_model,
        batch_size=config.EMBEDDING_BATCH_SIZE,
        batch_delay=config.EMBEDDING_BATCH_DELAY,
    )
    
    # Calculate vector size dynamically based on actual vector dimensions
    dim = len(vectors[0]) if vectors else 0
    # Each float is typically 4 bytes, so: num_vectors * dim * 4 bytes
    vector_size = len(vectors) * dim * 4
    
    logger.info(
        "Azure   │  All embeddings done  │  %d vectors  │  vector_size=%s",
        len(vectors),
        format_bytes(vector_size) if vectors else "0 B"
    )

    # ── Step 2: Build documents matching the index schema ─────────────────
    documents = []
    for chunk, text, vector in zip(chunks, texts, vectors):
        # The index schema defines 'metadata' as an Edm.ComplexType expected by N8N.
        # It contains a 'source' string and an 'attributes' collection of key-value pairs.
        attributes = [{"key": "project_name", "value": project_name}]
        
        # Add any other metadata from the document chunk
        if hasattr(chunk, 'metadata') and chunk.metadata:
            for k, v in chunk.metadata.items():
                if k != "source":
                    attributes.append({"key": str(k), "value": str(v)})
                    
        source = chunk.metadata.get("source", project_name) if hasattr(chunk, 'metadata') else project_name

        doc = {
            "@search.action": "upload",
            "id": uuid.uuid4().hex,          # unique 32-char hex string
            "content": text,
            "content_vector": list(map(float, vector)),
            "metadata": {
                "source": str(source),
                "attributes": attributes
            }
        }
        documents.append(doc)

    # ── Step 3: Upload in batches ─────────────────────────────────────────
    client = SearchClient(
        endpoint=config.AZURE_SEARCH_ENDPOINT,
        index_name=index,
        credential=AzureKeyCredential(config.AZURE_SEARCH_KEY),
    )

    UPLOAD_BATCH_SIZE = 100  # Azure recommends ≤1000; 100 keeps payloads manageable.
    total_uploaded = 0

    for i in range(0, len(documents), UPLOAD_BATCH_SIZE):
        batch = documents[i : i + UPLOAD_BATCH_SIZE]
        result = client.upload_documents(documents=batch)
        succeeded = sum(1 for r in result if r.succeeded)
        total_uploaded += succeeded

        if succeeded < len(batch):
            failed = len(batch) - succeeded
            logger.warning(
                "Azure   │  Upload batch %d–%d: %d succeeded, %d failed.",
                i, i + len(batch), succeeded, failed,
            )

    logger.info(
        "Azure   │  ✓  %d / %d chunks stored in index '%s'  │  total_vector_size=%s",
        total_uploaded,
        len(chunks),
        index,
        format_bytes(total_uploaded * dim * 4) if total_uploaded > 0 and dim else "0 B"
    )


# ── Target registry ──────────────────────────────────────────────────────────
# Add new targets here – just map the name to its push function.
_TARGET_REGISTRY: dict[str, callable] = {
    "qdrant": _push_to_qdrant,
    "azure":  _push_to_azure,
}


# ── Public API ────────────────────────────────────────────────────────────────

def push_to_all_targets(chunks: list[Document], project_name: str) -> None:
    """
    Fan-out: push *chunks* to every vector store listed in
    config.VECTORSTORE_TARGETS.

    Errors on one target are logged and do NOT prevent the remaining
    targets from being attempted.
    """
    if not chunks:
        logger.warning("No chunks to push – skipping vectorstore writes.")
        return

    for target_name in config.VECTORSTORE_TARGETS:
        push_fn = _TARGET_REGISTRY.get(target_name)
        if push_fn is None:
            logger.error("Unknown vectorstore target '%s' – skipping.", target_name)
            continue

        try:
            push_fn(chunks, project_name)
        except Exception as exc:
            logger.error(
                "Failed to push to '%s' for project '%s': %s",
                target_name,
                project_name,
                exc,
                exc_info=True,
            )
