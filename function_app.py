import azure.functions as func
import logging
import os
import sys

# Ensure current directory is in path for imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from blob_service import process_blob

app = func.FunctionApp()

@app.blob_trigger(arg_name="myblob", 
                  path=f"{os.getenv('AZURE_STORAGE_CONTAINER_NAME')}/{{name}}",
                  connection="AZURE_STORAGE_CONNECTION_STRING") 
def blob_trigger_handler(myblob: func.InputStream):
    """
    Azure Function trigger that fires when a new blob is uploaded.
    """
    blob_name = myblob.name.split('/', 1)[1] # Remove container name from path
    logging.info(f"Python blob trigger function processed blob \n"
                 f"Name: {blob_name}\n"
                 f"Blob Size: {myblob.length} bytes")
    
    try:
        process_blob(blob_name)
    except Exception as e:
        logging.error(f"Error in blob trigger handler: {e}")
