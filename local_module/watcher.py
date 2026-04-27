"""
──────────────────────────────────────────────────────────────────────────────
VectorizerEngine  ·  local_module/watcher.py
──────────────────────────────────────────────────────────────────────────────
Filesystem event handler powered by the `watchdog` library, with support
for network paths via polling-based watching.

Two modes:
  1. Event-based (watchdog): Fast, for local filesystems
  2. Polling-based: Reliable, for network paths (\\server\share)

Both modes:
  1. Detect newly created sub-directories (projects).
  2. Verify files are present locally (with retries).
  3. Hand the folder to pipeline.process_project_folder().
  4. Log success or failure.

Only top-level folder creations trigger the pipeline – nested folders
within a project are handled by the recursive file discovery in
loaders.discover_files().
──────────────────────────────────────────────────────────────────────────────
"""

import logging
import threading
import time
from pathlib import Path
from typing import Optional

from watchdog.events import FileSystemEventHandler, DirCreatedEvent
from watchdog.observers import Observer

import config
from core.pipeline import process_project_folder
from core.loaders import check_files_present

logger = logging.getLogger("watcher")


class ProjectFolderHandler(FileSystemEventHandler):
    """
    Reacts to new directories created directly inside the watched folder.

    Each new project is processed in its own daemon thread so the observer
    is never blocked while a large project is being ingested.
    """

    def __init__(self) -> None:
        super().__init__()
        # Track folders we have already triggered on, to avoid double fires.
        self._seen: set[str] = set()
        self._lock = threading.Lock()

    # ── Event handler ─────────────────────────────────────────────────────

    def on_created(self, event: DirCreatedEvent) -> None:
        """Called by watchdog when a new filesystem entry is created."""

        # We only care about directories, not files.
        if not event.is_directory:
            return

        new_folder = Path(event.src_path)
        watch_root = Path(config.WATCH_FOLDER_PATH)

        # Only react to DIRECT children of the watch root (top-level projects).
        if new_folder.parent.resolve() != watch_root.resolve():
            return

        # Deduplicate – watchdog can fire multiple events for one creation.
        with self._lock:
            key = str(new_folder.resolve())
            if key in self._seen:
                return
            self._seen.add(key)

        logger.info(
            "New project folder detected  →  '%s'",
            new_folder.name,
        )

        # Process in a background thread so the observer stays responsive.
        thread = threading.Thread(
            target=self._process_with_delay,
            args=(new_folder,),
            daemon=True,
            name=f"ingest-{new_folder.name}",
        )
        thread.start()

    # ── Internal ──────────────────────────────────────────────────────────

    def _process_with_delay(self, folder: Path) -> None:
        """
        Check that files are present locally, then run the pipeline.

        Instead of waiting a fixed time, this actively verifies that
        supported files exist in the folder before starting the ingestion.
        Uses retry logic to handle sync delays (e.g., OneDrive).
        """
        logger.info(
            "Verifying files are present locally  →  '%s'",
            folder.name,
        )
        
        # Check if files are present with retries
        files_found = check_files_present(
            folder,
            max_retries=config.FILE_CHECK_MAX_RETRIES,
            retry_delay=config.FILE_CHECK_RETRY_DELAY,
        )
        
        if not files_found:
            logger.error(
                "Project folder has no supported files after checking  →  '%s'  │  skipping ingestion",
                folder.name,
            )
            return

        try:
            total_chunks = process_project_folder(folder)
            logger.info(
                "Project '%s' ingested successfully  │  %d chunks total.",
                folder.name,
                total_chunks,
            )
        except Exception as exc:
            logger.error(
                "Project '%s' ingestion FAILED: %s",
                folder.name,
                exc,
                exc_info=True,
            )


# ── Polling Watcher for Network Paths ──────────────────────────────────────────

class PollingFolderWatcher:
    """
    Polling-based folder watcher for unreliable network paths.
    
    Instead of relying on filesystem events (watchdog), this periodically
    scans the target folder for new directories and processes them.
    More reliable for network shares but uses more CPU.
    """
    
    def __init__(self, watch_path: Path, poll_interval: float = 10.0) -> None:
        """
        Parameters
        ----------
        watch_path : Path
            Folder to monitor.
        poll_interval : float
            Seconds between scans (default: 10.0).
        """
        self.watch_path = Path(watch_path)
        self.poll_interval = poll_interval
        self._seen: set[str] = set()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
    
    def start(self) -> None:
        """Start the polling watcher thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._poll_loop,
            daemon=True,
            name="polling-watcher",
        )
        self._thread.start()
        logger.info(
            "Polling watcher started  │  path='%s'  │  interval=%.1fs",
            self.watch_path,
            self.poll_interval,
        )
    
    def stop(self) -> None:
        """Stop the polling watcher thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("Polling watcher stopped.")
    
    def _poll_loop(self) -> None:
        """Main polling loop – runs in background thread."""
        while not self._stop_event.is_set():
            try:
                self._scan_folder()
            except Exception as exc:
                logger.error("Polling watcher error: %s", exc, exc_info=True)
            
            # Wait for the next poll, but allow stopping during wait
            self._stop_event.wait(self.poll_interval)
    
    def _scan_folder(self) -> None:
        """Scan the watched folder for new directories."""
        if not self.watch_path.exists():
            logger.warning(
                "Watched folder does not exist or is not accessible: %s",
                self.watch_path,
            )
            return
        
        try:
            folders = sorted(
                p for p in self.watch_path.iterdir()
                if p.is_dir() and not p.name.startswith(".")
            )
        except (OSError, PermissionError) as exc:
            logger.warning(
                "Failed to scan folder (network issue?): %s",
                exc,
            )
            return
        
        for folder in folders:
            with self._lock:
                key = str(folder.resolve())
                if key in self._seen:
                    continue
                self._seen.add(key)
            
            logger.info(
                "New project folder detected (polling)  →  '%s'",
                folder.name,
            )
            
            # Process in background thread
            thread = threading.Thread(
                target=self._process_folder,
                args=(folder,),
                daemon=True,
                name=f"ingest-{folder.name}",
            )
            thread.start()
    
    def _process_folder(self, folder: Path) -> None:
        """Process a project folder."""
        logger.info(
            "Verifying files are present locally  →  '%s'",
            folder.name,
        )
        
        # Check if files are present with retries
        files_found = check_files_present(
            folder,
            max_retries=config.FILE_CHECK_MAX_RETRIES,
            retry_delay=config.FILE_CHECK_RETRY_DELAY,
        )
        
        if not files_found:
            logger.error(
                "Project folder has no supported files after checking  →  '%s'  │  skipping ingestion",
                folder.name,
            )
            return
        
        try:
            total_chunks = process_project_folder(folder)
            logger.info(
                "Project '%s' ingested successfully  │  %d chunks total.",
                folder.name,
                total_chunks,
            )
        except Exception as exc:
            logger.error(
                "Project '%s' ingestion FAILED: %s",
                folder.name,
                exc,
                exc_info=True,
            )
