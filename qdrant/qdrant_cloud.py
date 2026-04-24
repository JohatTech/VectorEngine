import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Document as QdrantDocument
import config
from .base import retry_qdrant_operation, verify_collection_has_vectors

logger = logging.getLogger("qdrant.cloud")

class QdrantCloudModule:
    def __init__(self, url=None, api_key=None, timeout=None):
        self.url = url or config.QDRANT_URL
        self.api_key = api_key or config.QDRANT_API_KEY
        self.timeout = timeout or config.QDRANT_TIMEOUT
        
        logger.info("Connecting to Qdrant Cloud  │  url=%s", self.url)
        self.client = QdrantClient(
            url=self.url,
            api_key=self.api_key,
            timeout=self.timeout,
            cloud_inference=True
        )

    def create_collection(self, collection_name, vector_size=384, distance=Distance.COSINE):
        """Create a collection in Qdrant Cloud, deleting it first if it exists."""
        def operation():
            if self.client.collection_exists(collection_name):
                logger.info("Collection '%s' already exists on Qdrant Cloud. Deleting and recreating...", collection_name)
                self.client.delete_collection(collection_name)
            
            logger.info("Creating collection '%s' on Qdrant Cloud...", collection_name)
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=distance),
            )
            return True

        return retry_qdrant_operation(
            f"Create Cloud collection {collection_name}",
            operation,
            max_retries=config.QDRANT_MAX_RETRIES,
            retry_delay=config.QDRANT_RETRY_DELAY
        )

    def upsert_documents(self, collection_name, chunks, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """
        Upsert documents to Qdrant Cloud using Cloud Inference.
        """
        points = []
        for i, chunk in enumerate(chunks):
            # Use Qdrant's Document model for cloud inference
            point = PointStruct(
                id=i, # In a real app, you might want a more stable ID
                vector=QdrantDocument(
                    text=chunk.page_content,
                    model=model_name
                ),
                payload=chunk.metadata
            )
            points.append(point)

        def operation():
            logger.info("Upserting %d points to Cloud collection '%s'...", len(points), collection_name)
            return self.client.upsert(
                collection_name=collection_name,
                points=points
            )

        result = retry_qdrant_operation(
            f"Upsert to Cloud collection {collection_name}",
            operation,
            max_retries=config.QDRANT_MAX_RETRIES,
            retry_delay=config.QDRANT_RETRY_DELAY
        )
        
        # Verify
        verify_collection_has_vectors(self.client, collection_name)
        return result
