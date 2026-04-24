"""
──────────────────────────────────────────────────────────────────────────────
VectorizerEngine  ·  enrichment.py
──────────────────────────────────────────────────────────────────────────────
Enriches every document chunk with metadata that ties it back to its
source project and file.

Why this matters
────────────────
• Qdrant stores each project in its own collection, so metadata is nice
  to have but not critical for isolation.
• Azure AI Search puts ALL projects into ONE index.  The `project_name`
  field is what lets an LLM (or a search filter) scope its retrieval to
  a single project without confusion.
──────────────────────────────────────────────────────────────────────────────
"""

import logging
from pathlib import Path

from langchain_core.documents import Document

logger = logging.getLogger("enrichment")


def enrich_chunks(
    chunks: list[Document],
    project_name: str,
    source_file: Path | str,
) -> list[Document]:
    """
    Stamp every chunk with project-level and file-level metadata.

    Added metadata keys
    ───────────────────
    project_name : str   – The project (folder) this chunk belongs to.
    source_file  : str   – The original filename (e.g. "report.pdf").
    file_type    : str   – The file extension without the dot (e.g. "pdf").

    Parameters
    ----------
    chunks : list[Document]
        Chunks produced by loaders.load_and_chunk_file().
    project_name : str
        Human-readable project name (the folder name).
    source_file : Path | str
        Full path to the original file.

    Returns
    -------
    list[Document]
        The same list, mutated in-place for zero-copy efficiency.
    """
    source_path = Path(source_file)
    file_name = source_path.name
    file_type = source_path.suffix.lstrip(".").lower()

    for chunk in chunks:
        chunk.metadata["project_name"] = project_name
        chunk.metadata["source_file"] = file_name
        chunk.metadata["file_type"] = file_type

    logger.debug(
        "Enriched %d chunks with project='%s', file='%s'.",
        len(chunks),
        project_name,
        file_name,
    )
    return chunks
