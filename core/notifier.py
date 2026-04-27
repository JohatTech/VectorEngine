"""
──────────────────────────────────────────────────────────────────────────────
VectorizerEngine  ·  core/notifier.py
──────────────────────────────────────────────────────────────────────────────
Sends HTTP callbacks to external systems (e.g. N8N) after a project
has been fully ingested and vectorised.

Design
──────
• build_payload()  → returns a dict with all the data we want to send.
  Edit THIS function to add/remove fields in the future.
• notify_n8n()     → POSTs the payload to the configured N8N webhook URL.
• The module is intentionally decoupled from the pipeline so it can be
  reused or called independently.

N8N Webhook Setup
─────────────────
1. In N8N, create a Webhook node with method POST.
2. Copy the webhook URL into your .env as N8N_WEBHOOK_URL.
3. The payload arrives as the request body – use {{ $json.project_name }},
   {{ $json.collection_name }}, etc. in downstream N8N nodes.
──────────────────────────────────────────────────────────────────────────────
"""

import logging
from datetime import datetime, timezone
from typing import Any

import requests

import config

logger = logging.getLogger("notifier")


# ── Payload Builder ──────────────────────────────────────────────────────────
# Edit this function to control exactly what data N8N receives.
# Every key you add here becomes available as {{ $json.<key> }} in N8N.

def build_payload(
    project_name: str,
    collection_name: str,
    total_chunks: int,
    vectorstore_targets: list[str],
    azure_index_name: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Assemble the JSON payload that will be POSTed to N8N.

    Parameters
    ----------
    project_name : str
        Human-readable project name (the folder name).
    collection_name : str
        Sanitised collection name used in Qdrant.
    total_chunks : int
        Number of chunks that were vectorised.
    vectorstore_targets : list[str]
        Which stores received the data (e.g. ["qdrant", "azure"]).
    azure_index_name : str, optional
        The Azure AI Search index name (so N8N knows where to query).
    extra : dict, optional
        Any additional key-value pairs you want to include.

    Returns
    -------
    dict
        The complete payload ready for json serialisation.
    """
    payload: dict[str, Any] = {
        # ── Core identifiers ─────────────────────────────────────────────
        "project_name": project_name,
        "collection_name": collection_name,

        # ── Azure AI Search filter ───────────────────────────────────────
        # N8N can use this value directly in an Azure AI Search node filter:
        #   metadata/attributes/any(a: a/key eq 'project_name' and a/value eq '<value>')
        "azure_index_name": azure_index_name or config.AZURE_SEARCH_INDEX_NAME,


        # ── Stats ────────────────────────────────────────────────────────
        "total_chunks": total_chunks,
        "vectorstore_targets": vectorstore_targets,

        # ── Timestamp ────────────────────────────────────────────────────
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }

    # Merge any extra data the caller wants to include.
    if extra:
        payload.update(extra)

    return payload


# ── HTTP Sender ──────────────────────────────────────────────────────────────

def notify_n8n(payload: dict[str, Any]) -> bool:
    """
    POST *payload* as JSON to the configured N8N webhook URL.

    Returns True on success (2xx), False otherwise.
    Errors are logged but never crash the caller.
    """
    webhook_url = config.N8N_WEBHOOK_URL

    if not webhook_url:
        logger.warning("N8N_WEBHOOK_URL is not set – skipping notification.")
        return False

    logger.info(
        "Notifier  │  Sending callback to N8N  →  project='%s'",
        payload.get("project_name", "unknown"),
    )

    try:
        response = requests.post(
            webhook_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=15,
        )
        response.raise_for_status()

        logger.info(
            "Notifier  │  ✓  N8N responded %d  │  project='%s'",
            response.status_code,
            payload.get("project_name"),
        )
        return True

    except requests.RequestException as exc:
        logger.error(
            "Notifier[requestion exception error]  │  ✗  N8N callback failed: %s",
            exc,
        )
        return False
