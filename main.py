"""
──────────────────────────────────────────────────────────────────────────────
VectorizerEngine  ·  main.py
──────────────────────────────────────────────────────────────────────────────
Entry point for the VectorizerEngine service.

What it does
────────────
  1. Validates that the watched folder exists.
  2. Optionally processes any project folders that already exist and
     have not been ingested yet (--backfill flag).
  3. Starts a watchdog Observer that runs forever, picking up new
     project folders as they appear.
  4. Handles Ctrl+C gracefully.

Usage
─────
  # Normal continuous monitoring:
  python main.py

  # Process existing folders first, then keep watching:
  python main.py --backfill

  # Process one specific folder and exit (no watching):
  python main.py --once "Project Alpha"
──────────────────────────────────────────────────────────────────────────────
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from watchdog.observers import Observer

import config
from watcher import ProjectFolderHandler, PollingFolderWatcher
from pipeline import process_project_folder

logger = logging.getLogger("main")


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_cli() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="VectorizerEngine",
        description=(
            "Monitor a folder for new project sub-folders, load their "
            "documents, and vectorise them into Qdrant and/or Azure AI Search."
        ),
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Process all existing project folders before starting the watcher.",
    )
    parser.add_argument(
        "--once",
        metavar="FOLDER_NAME",
        type=str,
        default=None,
        help=(
            "Process a single project folder by name and exit. "
            "The folder must be a direct child of the watch path."
        ),
    )
    return parser


# ── Backfill ──────────────────────────────────────────────────────────────────

def backfill_existing_folders(watch_path: Path) -> None:
    """
    Scan the watch path for sub-directories that already exist and
    process each one through the pipeline.

    Useful for first-time setup or recovery after downtime.
    """
    logger.info("Backfill  │  Scanning for existing project folders …")
    folders = sorted(
        p for p in watch_path.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )

    if not folders:
        logger.info("Backfill  │  No existing folders found.")
        return

    logger.info("Backfill  │  Found %d folder(s) to process.", len(folders))
    for folder in folders:
        try:
            process_project_folder(folder)
        except Exception as exc:
            logger.error("Backfill  │  Failed on '%s': %s", folder.name, exc, exc_info=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Application entry point."""
    args = build_cli().parse_args()
    watch_path = Path(config.WATCH_FOLDER_PATH)

    # ── Validate watched folder ───────────────────────────────────────────
    if not watch_path.exists():
        logger.error("Watch folder does not exist: %s", watch_path)
        sys.exit(1)
    if not watch_path.is_dir():
        logger.error("Watch path is not a directory: %s", watch_path)
        sys.exit(1)

    logger.info("VectorizerEngine starting up …")
    logger.info("Watch folder  →  %s", watch_path)
    logger.info("Targets       →  %s", ", ".join(config.VECTORSTORE_TARGETS))
    logger.info("Chunk size    →  %d  │  Overlap  →  %d", config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    logger.info("Max workers   →  %d", config.MAX_WORKERS)
    
    # Show which watcher mode will be used
    if config.USE_BLOB_WATCHER:
        logger.info("Watcher mode  →  AZURE BLOB STORAGE  │  Container  →  %s", config.AZURE_STORAGE_CONTAINER_NAME)
    elif config.USE_POLLING_WATCHER:
        logger.info("Watcher mode  →  POLLING (local)  │  Interval  →  %.1fs", config.POLLING_INTERVAL_SECONDS)
    else:
        logger.info("Watcher mode  →  EVENT-BASED (watchdog)  │  Stability wait  →  %ds", config.STABILITY_WAIT_SECONDS)

    # ── One-shot mode ─────────────────────────────────────────────────────
    if args.once:
        target = watch_path / args.once
        if not target.is_dir():
            logger.error("Folder '%s' not found in %s", args.once, watch_path)
            sys.exit(1)
        process_project_folder(target)
        return

    # ── Backfill existing folders (optional) ──────────────────────────────
    if args.backfill:
        backfill_existing_folders(watch_path)

    # ── Start the live watcher ────────────────────────────────────────────
    if config.USE_BLOB_WATCHER:
        from blob_watcher import AzureBlobWatcher
        watcher = AzureBlobWatcher(poll_interval=config.POLLING_INTERVAL_SECONDS)
        watcher.start()
        
        logger.info("Azure Blob watcher is live. Press Ctrl+C to stop.")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutdown requested – stopping watcher …")
            watcher.stop()
            
    elif config.USE_POLLING_WATCHER:
        # Network-safe polling watcher
        watcher = PollingFolderWatcher(
            watch_path=watch_path,
            poll_interval=config.POLLING_INTERVAL_SECONDS,
        )
        watcher.start()
        
        logger.info("Polling watcher is live.  Press Ctrl+C to stop.")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutdown requested – stopping watcher …")
            watcher.stop()
    else:
        # Event-based watchdog watcher
        handler = ProjectFolderHandler()
        observer = Observer()
        observer.schedule(handler, str(watch_path), recursive=False)
        observer.start()
        
        logger.info("Event-based watcher is live.  Press Ctrl+C to stop.")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutdown requested – stopping watcher …")
            observer.stop()
        
        observer.join()
    logger.info("VectorizerEngine stopped.  Goodbye!")


if __name__ == "__main__":
    main()
