import logging
import threading
import time
from typing import Set, Optional

import config
from blob_module.blob_service import get_blob_service_client, process_blob

logger = logging.getLogger("blob_watcher")

class AzureBlobWatcher:
    """
    Polling-based watcher for Azure Blob Storage.
    Detects new blobs in a container and triggers processing.
    """
    
    def __init__(self, poll_interval: float = 30.0) -> None:
        self.poll_interval = poll_interval
        self._seen: Set[str] = set()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        
    def start(self) -> None:
        """Start the polling watcher thread."""
        self._stop_event.clear()
        
        # Initial scan to populate 'seen' blobs (optional: could process them too)
        try:
            self._populate_initial_blobs()
        except Exception as e:
            logger.error("Failed to perform initial blob scan: %s", e)

        self._thread = threading.Thread(
            target=self._poll_loop,
            daemon=True,
            name="blob-watcher",
        )
        self._thread.start()
        logger.info(
            "Azure Blob Watcher started  │  container='%s'  │  interval=%.1fs",
            config.AZURE_STORAGE_CONTAINER_NAME,
            self.poll_interval,
        )
    
    def stop(self) -> None:
        """Stop the polling watcher thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("Azure Blob Watcher stopped.")
    
    def _populate_initial_blobs(self) -> None:
        """Scan container to mark existing blobs as 'seen'."""
        client = get_blob_service_client()
        container_client = client.get_container_client(config.AZURE_STORAGE_CONTAINER_NAME)
        
        blobs = list(container_client.list_blobs())
        for blob in blobs:
            self._seen.add(blob.name)
        logger.info("Initial scan complete. Found %d existing blobs in container '%s'.", 
                    len(self._seen), config.AZURE_STORAGE_CONTAINER_NAME)

    def _poll_loop(self) -> None:
        """Main polling loop."""
        while not self._stop_event.is_set():
            try:
                self._scan_blobs()
            except Exception as exc:
                logger.error("Blob watcher error: %s", exc, exc_info=True)
            
            self._stop_event.wait(self.poll_interval)
    
    def _scan_blobs(self) -> None:
        """Scan the container for new blobs."""
        client = get_blob_service_client()
        container_client = client.get_container_client(config.AZURE_STORAGE_CONTAINER_NAME)
        
        blobs = list(container_client.list_blobs())
        
        for blob in blobs:
            if blob.name not in self._seen:
                logger.info("New blob detected: '%s'", blob.name)
                self._seen.add(blob.name)
                
                # Process in a separate thread to avoid blocking the scan
                thread = threading.Thread(
                    target=process_blob,
                    args=(blob.name,),
                    daemon=True,
                    name=f"blob-ingest-{blob.name}",
                )
                thread.start()
