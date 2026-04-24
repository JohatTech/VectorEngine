import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_qdrant import QdrantVectorStore
import config
from .base import retry_qdrant_operation, verify_collection_has_vectors

logger = logging.getLogger("qdrant.local")

class QdrantLocalModule:
    def __init__(self, path=None):
        self.path = path or config.QDRANT_LOCAL_PATH
        logger.info("Connecting to local Qdrant  │  path=%s", self.path)
        self.client = QdrantClient(path=self.path)

    def create_collection(self, collection_name, vector_size=1536, distance=Distance.COSINE):
        """Create a local Qdrant collection, deleting it first if it exists."""
        def operation():
            if self.client.collection_exists(collection_name):
                logger.info("Local collection '%s' already exists. Deleting and recreating...", collection_name)
                self.client.delete_collection(collection_name)
            
            logger.info("Creating local collection '%s'...", collection_name)
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=distance),
            )
            return True

        return retry_qdrant_operation(
            f"Create local collection {collection_name}",
            operation,
            max_retries=config.QDRANT_MAX_RETRIES,
            retry_delay=config.QDRANT_RETRY_DELAY
        )

    def get_vectorstore(self, collection_name, embeddings):
        """
        Return a LangChain QdrantVectorStore instance for the local collection.
        """
        return QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=embeddings,
        )

    def upsert_documents(self, collection_name, chunks, embeddings):
        """
        Upsert documents using LangChain's vectorstore for local Qdrant.
        """
        store = self.get_vectorstore(collection_name, embeddings)
        
        def operation():
            logger.info("Adding %d documents to local Qdrant collection '%s'...", len(chunks), collection_name)
            return store.add_documents(chunks)

        result = retry_qdrant_operation(
            f"Add documents to local collection {collection_name}",
            operation,
            max_retries=config.QDRANT_MAX_RETRIES,
            retry_delay=config.QDRANT_RETRY_DELAY
        )
        
        # Verify
        verify_collection_has_vectors(self.client, collection_name)
        return result
