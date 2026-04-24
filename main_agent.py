import logging
import json
import time
import os
from pathlib import Path
from watchdog.observers import Observer

# Force local Qdrant for this autonomous system
os.environ.setdefault("QDRANT_MODE", "local")
os.environ.setdefault("VECTORSTORE_TARGETS", "qdrant")

import config
from watcher import ProjectFolderHandler, PollingFolderWatcher
from pipeline import process_project_folder
from agent_system import AutonomousRAGAgent
from report_generator import generate_report

# Configure logging
logger = logging.getLogger("main_agent")

def run_rag_analysis(folder: Path):
    project_name = folder.name
    logger.info("=" * 70)
    logger.info("ANALYSIS START  │  project='%s'", project_name)
    logger.info("=" * 70)

    # 1. Vectorization
    try:
        logger.info("Step 1: Vectorizing documents ...")
        # Ensure config is set to local Qdrant
        config.QDRANT_MODE = "local"
        if "qdrant" not in config.VECTORSTORE_TARGETS:
            config.VECTORSTORE_TARGETS.append("qdrant")
            
        chunks = process_project_folder(folder)
        if chunks == 0:
            logger.warning("No documents found in '%s'. Skipping RAG.", project_name)
            return
    except Exception as e:
        logger.error("Step 1 Failed: Vectorization error for '%s': %s", project_name, e)
        return

    # 2. Agent Processing
    try:
        logger.info("Step 2: Running Autonomous RAG Agent ...")
        prompts_path = Path("pliego_form.json")
        if not prompts_path.exists():
            logger.error("Prompts file 'pliego_form.json' not found. Skipping Agent.")
            return
            
        with open(prompts_path, "r", encoding="utf-8") as f:
            prompts = json.load(f)
            
        agent = AutonomousRAGAgent(project_name)
        report_md = agent.process_prompts(prompts)
        
        # 3. Report Generation
        logger.info("Step 3: Generating PDF report ...")
        pdf_path = generate_report(project_name, report_md)
        logger.info("=" * 70)
        logger.info("SUCCESS: Analysis complete for '%s'", project_name)
        logger.info("Report Path: %s", pdf_path)
        logger.info("=" * 70)
    except Exception as e:
        logger.error("Step 2/3 Failed: RAG/Reporting error for '%s': %s", project_name, e)

# ── Custom Watcher Handlers ───────────────────────────────────────────────────

class AgentProjectHandler(ProjectFolderHandler):

    def _process_with_delay(self, folder: Path) -> None:
        from loaders import check_files_present
        if check_files_present(folder, config.FILE_CHECK_MAX_RETRIES, config.FILE_CHECK_RETRY_DELAY):
            run_rag_analysis(folder)

class PollingAgentWatcher(PollingFolderWatcher):

    def _process_folder(self, folder: Path) -> None:
        from loaders import check_files_present
        if check_files_present(folder, config.FILE_CHECK_MAX_RETRIES, config.FILE_CHECK_RETRY_DELAY):
            run_rag_analysis(folder)

# ── Main Entry Point ──────────────────────────────────────────────────────────

def main():
    # ── Validate LLM API Key ───────────────────────────────────────────
    if not config.AZURE_OPENAI_API_KEY and not config.OPENAI_API_KEY:
        logger.error("CRITICAL: No LLM API key found (AZURE_OPENAI_API_KEY or OPENAI_API_KEY).")
        logger.error("Please set the API key in your .env file.")
        return

    watch_path = Path(config.WATCH_FOLDER_PATH)
    if not watch_path.exists():
        logger.error("Watch folder does not exist: %s", watch_path)
        return

    logger.info("Autonomous RAG Agent System Starting ...")
    logger.info("Watching Folder  →  %s", watch_path)
    logger.info("LLM Provider     →  %s", config.CHAT_PROVIDER)
    
    if config.USE_POLLING_WATCHER:
        logger.info("Mode: POLLING")
        watcher = PollingAgentWatcher(watch_path, poll_interval=config.POLLING_INTERVAL_SECONDS)
        watcher.start()
        try:
            while True: time.sleep(1)
        except KeyboardInterrupt:
            watcher.stop()
    else:
        logger.info("Mode: EVENT-BASED (Watchdog)")
        handler = AgentProjectHandler()
        observer = Observer()
        observer.schedule(handler, str(watch_path), recursive=False)
        observer.start()
        try:
            while True: time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopping watcher ...")
            observer.stop()
            observer.join()

    logger.info("Agent System stopped. Goodbye!")

if __name__ == "__main__":
    main()
