"""
──────────────────────────────────────────────────────────────────────────────
VectorizerEngine  ·  core/pipeline.py
──────────────────────────────────────────────────────────────────────────────
The orchestrator.  Given a project folder path, this module:

  1. Discovers every supported file inside it.
  2. Loads & chunks each file in parallel (ThreadPoolExecutor).
  3. Enriches every chunk with project + file metadata.
  4. Pushes the combined chunk list to all configured vector stores.
  5. Notifies N8N (via webhook) that the project is ready.

This is the ONLY module that ties loaders, enrichment, vectorstore_manager,
and notifier together.  Every other module is independently testable.
──────────────────────────────────────────────────────────────────────────────
"""

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from langchain_core.documents import Document

import config
from core.loaders import discover_files, load_and_chunk_file
from core.enrichment import enrich_chunks
from core.vectorstore_manager import push_to_all_targets
from core.notifier import build_payload, notify_n8n
from core.utils import format_bytes

logger = logging.getLogger("pipeline")


def _load_single_file(file_path: Path) -> tuple[Path, list[Document]]:
    """
    Worker function executed inside a thread.
    Returns a (file_path, chunks) tuple so the caller can enrich them.
    """
    chunks = load_and_chunk_file(file_path)
    return file_path, chunks


def process_project_folder(folder_path: str | Path) -> int:
    """
    End-to-end ingestion of a single project folder.

    Parameters
    ----------
    folder_path : str | Path
        Absolute path to the newly created project folder.

    Returns
    -------
    int
        Total number of chunks successfully pushed to vector stores.
    """
    folder = Path(folder_path)
    project_name = folder.name  # The folder name IS the project name.

    logger.info("=" * 70)
    logger.info("PIPELINE START  │  project='%s'", project_name)
    logger.info("=" * 70)

    start_time = time.perf_counter()

    # ── Step 1: Discover files ────────────────────────────────────────────
    files = discover_files(folder)
    if not files:
        logger.warning("No supported files found in '%s'. Nothing to do.", project_name)
        return 0

    # ── Step 2: Load & chunk in parallel ──────────────────────────────────
    all_chunks: list[Document] = []

    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as pool:
        futures = {
            pool.submit(_load_single_file, f): f
            for f in files
        }

        for future in as_completed(futures):
            file_path = futures[future]
            try:
                _, chunks = future.result()
            except Exception as exc:
                logger.error("Worker failed for '%s': %s", file_path.name, exc)
                continue

            if not chunks:
                continue

            # ── Step 3: Enrich metadata ───────────────────────────────────
            enrich_chunks(chunks, project_name=project_name, source_file=file_path)
            all_chunks.extend(chunks)

    # Calculate total content size
    total_content_size = sum(len(chunk.page_content.encode('utf-8')) for chunk in all_chunks)

    logger.info(
        "Loading complete  │  files=%d  │  total_chunks=%d  │  total_content_size=%s  │  %.2fs",
        len(files),
        len(all_chunks),
        format_bytes(total_content_size) if all_chunks else "0 B",
        time.perf_counter() - start_time,
    )

    if not all_chunks:
        logger.warning("All files produced zero chunks. Nothing to push.")
        return 0

    # ── Step 4: Push to vector stores ─────────────────────────────────────
    push_to_all_targets(all_chunks, project_name)

    # ── Step 5: Notify N8N ────────────────────────────────────────────────
    collection_name = re.sub(r"[\s\-]+", "_", project_name.lower())
    collection_name = re.sub(r"[^a-z0-9_]", "", collection_name).strip("_")

    payload = build_payload(
        project_name=project_name,
        collection_name=collection_name,
        total_chunks=len(all_chunks),
        vectorstore_targets=config.VECTORSTORE_TARGETS,
    )
    notify_n8n(payload)

    elapsed = time.perf_counter() - start_time
    logger.info(
        "PIPELINE DONE  │  project='%s'  │  chunks=%d  │  %.2fs",
        project_name,
        len(all_chunks),
        elapsed,
    )
    return len(all_chunks)
