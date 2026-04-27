from langchain_community.vectorstores.azuresearch import AzureSearch
import logging
from qdrant import (
    QdrantCloudModule, 
    QdrantLocalModule, 
    verify_collection_has_vectors as base_verify,
    retry_qdrant_operation as base_retry
)
import config

logger = logging.getLogger("vectorizer")

def create_qdrant_vectorstore(collection_name, embeddings, qdrant_url=None, qdrant_api_key=None, path=None):
    """
    Legacy wrapper for creating a Qdrant vectorstore.
    Now uses the modular QdrantCloudModule or QdrantLocalModule.
    """
    if path or config.QDRANT_MODE == "local":
        module = QdrantLocalModule(path=path)
        module.create_collection(collection_name)
        return module.get_vectorstore(collection_name, embeddings), module.client
    else:
        module = QdrantCloudModule(url=qdrant_url, api_key=qdrant_api_key)
        module.create_collection(collection_name)
        # Note: QdrantCloudModule doesn't normally use LangChain's QdrantVectorStore 
        # in the new implementation because of Cloud Inference, but for this 
        # legacy wrapper we'll return a LangChain instance for compatibility.
        from langchain_qdrant import QdrantVectorStore
        store = QdrantVectorStore(
            client=module.client,
            collection_name=collection_name,
            embedding=embeddings,
        )
        return store, module.client

def verify_collection_has_vectors(client, collection_name):
    """Legacy wrapper for verifying vectors."""
    return base_verify(client, collection_name)

def _retry_qdrant_operation(operation_name, operation_func, max_retries=5, retry_delay=2.0):
    """Legacy wrapper for retry logic."""
    return base_retry(operation_name, operation_func, max_retries, retry_delay)

def create_azure_search_vectorstore(index_name, embeddings, azure_search_endpoint, azure_search_key):
    """
    Creates or loads an Azure AI Search VectorStore.
    """
    vectorstore = AzureSearch(
        azure_search_endpoint=azure_search_endpoint,
        azure_search_key=azure_search_key,
        index_name=index_name,
        embedding_function=embeddings.embed_query,
    )
    return vectorstore
