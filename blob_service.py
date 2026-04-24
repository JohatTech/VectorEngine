import logging
import os
import zipfile
from pathlib import Path
from azure.storage.blob import BlobServiceClient
from typing import List

import config

logger = logging.getLogger("blob_service")

def get_blob_service_client():
    if not config.AZURE_STORAGE_CONNECTION_STRING:
        raise ValueError("AZURE_STORAGE_CONNECTION_STRING is not set")
    return BlobServiceClient.from_connection_string(config.AZURE_STORAGE_CONNECTION_STRING)

def download_blob_to_temp(blob_name: str, download_path: Path) -> Path:
    """
    Downloads a blob to a local path.
    """
    client = get_blob_service_client()
    container_client = client.get_container_client(config.AZURE_STORAGE_CONTAINER_NAME)
    blob_client = container_client.get_blob_client(blob_name)

    logger.info("Downloading blob: %s to %s", blob_name, download_path)
    download_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(download_path, "wb") as file:
        data = blob_client.download_blob()
        data.readinto(file)
    
    return download_path

def extract_if_zip(file_path: Path) -> Path:
    """
    If the file is a zip, extract it to a folder with the same name.
    Returns the path to the folder (either the extracted one or the parent of the file).
    """
    if file_path.suffix.lower() == ".zip":
        extract_to = file_path.parent / file_path.stem
        logger.info("Extracting %s to %s", file_path, extract_to)
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        return extract_to
    return file_path.parent

def process_blob(blob_name: str):
    """
    Downloads, extracts (if needed), and processes a blob.
    """
    # Clean up blob name for local path (strip leading slashes and literal quotes)
    clean_name = blob_name.lstrip("/").replace('"', '').replace("'", "")
    
    temp_root = Path("temp_blobs")
    blob_path = temp_root / clean_name
    
    try:
        # 1. Download (using original name for Azure, clean name for local path)
        downloaded_file = download_blob_to_temp(blob_name, blob_path)
        
        # 2. Extract if needed
        # The user said "extract the file", so we assume zip or similar if applicable.
        # Otherwise, the "project folder" is the directory containing the file.
        processing_dir = extract_if_zip(downloaded_file)
        
        # 3. Call pipeline
        from pipeline import process_project_folder
        process_project_folder(processing_dir)
        
        logger.info("Successfully processed blob: %s", blob_name)
    except Exception as e:
        logger.error("Error processing blob %s: %s", blob_name, e, exc_info=True)
    finally:
        # Optionally clean up temp files? 
        # For now, let's keep them or at least don't crash if we can't delete.
        pass
