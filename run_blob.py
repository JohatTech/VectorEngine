"""
──────────────────────────────────────────────────────────────────────────────
VectorizerEngine  ·  run_blob.py
──────────────────────────────────────────────────────────────────────────────
Entry point for Azure Blob Storage event processing.

Modes:
  1. Azure Function trigger (deployed via function_app.py in blob_module)
  2. Local polling watcher (polls Azure Blob container for new blobs)

Usage
─────
  # Start local blob watcher (polls Azure container):
  python run_blob.py
──────────────────────────────────────────────────────────────────────────────
"""

import logging
import time

import config
from blob_module.blob_watcher import AzureBlobWatcher

logger = logging.getLogger("run_blob")


def main() -> None:
    """Start the Azure Blob Storage polling watcher."""
    logger.info("VectorizerEngine — Blob Storage Watcher starting up …")
    logger.info("Container       →  %s", config.AZURE_STORAGE_CONTAINER_NAME)
    logger.info("Poll interval   →  %.1fs", config.POLLING_INTERVAL_SECONDS)
    logger.info("Targets         →  %s", ", ".join(config.VECTORSTORE_TARGETS))

    watcher = AzureBlobWatcher(poll_interval=config.POLLING_INTERVAL_SECONDS)
    watcher.start()

    logger.info("Azure Blob watcher is live. Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutdown requested – stopping watcher …")
        watcher.stop()

    logger.info("Blob watcher stopped.  Goodbye!")


if __name__ == "__main__":
    main()
