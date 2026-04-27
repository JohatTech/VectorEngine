# blob_module — Azure Blob Storage event processing
from blob_module.blob_service import process_blob, download_blob_to_temp, extract_if_zip
from blob_module.blob_watcher import AzureBlobWatcher
