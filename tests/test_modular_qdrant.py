import logging
from langchain_core.documents import Document
import config
from qdrant import QdrantLocalModule, QdrantCloudModule, sanitise_collection_name
from embeddings import get_embeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_modular")

def test_local():
    logger.info("Testing Local Module...")
    project_name = "Test Local Project"
    collection = sanitise_collection_name(project_name)
    
    chunks = [
        Document(page_content="Local test content 1", metadata={"source": "test"}),
        Document(page_content="Local test content 2", metadata={"source": "test"}),
    ]
    
    embeddings = get_embeddings()
    local_module = QdrantLocalModule()
    local_module.create_collection(collection)
    local_module.upsert_documents(collection, chunks, embeddings)
    logger.info("Local test complete.")

def test_cloud():
    if not config.QDRANT_URL or not config.QDRANT_API_KEY:
        logger.warning("Skipping Cloud test: QDRANT_URL or QDRANT_API_KEY not set.")
        return

    logger.info("Testing Cloud Module...")
    project_name = "Test Cloud Project"
    collection = sanitise_collection_name(project_name)
    
    chunks = [
        Document(page_content="Cloud test content 1", metadata={"source": "test"}),
        Document(page_content="Cloud test content 2", metadata={"source": "test"}),
    ]
    
    cloud_module = QdrantCloudModule()
    cloud_module.create_collection(collection)
    cloud_module.upsert_documents(collection, chunks)
    logger.info("Cloud test complete.")

if __name__ == "__main__":
    test_local()
    # test_cloud() # Uncomment if you have cloud credentials
