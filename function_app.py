import azure.functions as func
import logging
import os
import sys
from blob_module.blob_service import process_blob
# Ensure current directory is in path for imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

app = func.FunctionApp()

logging.info("VectorizerEngine Function App starting — registering blob trigger …")

@app.blob_trigger(arg_name="myblob", 
                  path="licitaciones",
                  connection="AZURE_STORAGE_CONNECTION_STRING") 
def blob_trigger_handler(myblob: func.InputStream):
    """
    Azure Function trigger that fires when a new blob is uploaded.
    """
    logging.info(">>> BLOB TRIGGER FIRED <<<")
    blob_name = myblob.name.split('/', 1)[1] if '/' in myblob.name else myblob.name
    logging.info(f"Python blob trigger function processed blob \n"
                 f"Name: {blob_name}\n"
                 f"Blob Size: {myblob.length} bytes")
    
    try:
        # Import config and blob_service lazily to avoid startup failures
        # if env vars are not yet available during module import
        from blob_module.blob_service import process_blob
        process_blob(blob_name)
    except Exception as e:
        logging.error(f"Error in blob trigger handler: {e}", exc_info=True)
