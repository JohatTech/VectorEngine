"""
Diagnostic script to test Qdrant vectorization end-to-end.

This script helps identify where vectors might be getting lost in the pipeline.
"""

import logging
from pathlib import Path
from langchain_core.documents import Document

import config
from embeddings import get_embeddings
from vectorizer import create_qdrant_vectorstore, verify_collection_has_vectors
from utils import format_bytes

logging.basicConfig(
    level="DEBUG",
    format="%(asctime)s  │  %(name)-22s  │  %(levelname)-7s  │  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("test_qdrant")


def test_basic_vectorization():
    """Test basic Qdrant vectorization with synthetic documents."""
    logger.info("=" * 80)
    logger.info("TEST 1: Basic Vectorization with Synthetic Documents")
    logger.info("=" * 80)
    
    # Create test documents
    test_docs = [
        Document(
            page_content="This is the first test document about machine learning.",
            metadata={"source": "test1.txt", "project": "test_project"}
        ),
        Document(
            page_content="This is the second test document about natural language processing.",
            metadata={"source": "test2.txt", "project": "test_project"}
        ),
        Document(
            page_content="This is the third test document about deep learning and neural networks.",
            metadata={"source": "test3.txt", "project": "test_project"}
        ),
    ]
    
    logger.info("Created %d test documents", len(test_docs))
    
    # Get embeddings
    logger.info("Loading embeddings model...")
    embeddings = get_embeddings()
    logger.info("Embeddings model loaded: %s", type(embeddings).__name__)
    
    # Create vectorstore
    logger.info("Creating Qdrant vectorstore (Mode: %s)...", config.QDRANT_MODE)
    test_collection = "test_vectorization"
    
    try:
        if config.QDRANT_MODE == "local":
            store, client = create_qdrant_vectorstore(
                collection_name=test_collection,
                embeddings=embeddings,
                path=config.QDRANT_LOCAL_PATH,
            )
        else:
            store, client = create_qdrant_vectorstore(
                collection_name=test_collection,
                embeddings=embeddings,
                qdrant_url=config.QDRANT_URL,
                qdrant_api_key=config.QDRANT_API_KEY,
            )
        logger.info("Vectorstore created successfully")
    except Exception as exc:
        logger.error("Failed to create vectorstore: %s", exc, exc_info=True)
        return False
    
    # Add documents
    logger.info("Adding %d documents to vectorstore...", len(test_docs))
    try:
        result = store.add_documents(test_docs)
        logger.info("add_documents() result: %s", result)
    except Exception as exc:
        logger.error("Failed to add documents: %s", exc, exc_info=True)
        return False
    
    # Verify vectors were stored
    logger.info("Verifying vectors in collection...")
    stored_count = verify_collection_has_vectors(client, test_collection)
    
    if stored_count == 0:
        logger.error("FAILED: No vectors stored in collection!")
        return False
    
    if stored_count < len(test_docs):
        logger.warning("PARTIAL: Only %d/%d vectors stored", stored_count, len(test_docs))
    else:
        logger.info("SUCCESS: All %d vectors stored correctly", stored_count)
    
    # Try a similarity search
    logger.info("Testing similarity search...")
    try:
        results = store.similarity_search("machine learning", k=2)
        logger.info("Similarity search returned %d results:", len(results))
        for i, doc in enumerate(results, 1):
            logger.info("  %d. %s... (score: %s)", i, doc.page_content[:50], "N/A")
        return True
    except Exception as exc:
        logger.error("Similarity search failed: %s", exc, exc_info=True)
        return False


def test_embedding_generation():
    """Test if embeddings are being generated correctly."""
    logger.info("=" * 80)
    logger.info("TEST 2: Embedding Generation")
    logger.info("=" * 80)
    
    embeddings = get_embeddings()
    test_texts = [
        "Hello world",
        "This is a test",
        "Testing embeddings",
    ]
    
    logger.info("Testing embedding generation for %d texts...", len(test_texts))
    
    try:
        vectors = embeddings.embed_documents(test_texts)
        logger.info("Generated %d vectors", len(vectors))
        
        for i, (text, vector) in enumerate(zip(test_texts, vectors), 1):
            logger.info(
                "  Vector %d: text='%s' → %d dimensions, size=%s",
                i,
                text,
                len(vector),
                format_bytes(len(vector) * 4),  # 4 bytes per float32
            )
        
        return True
    except Exception as exc:
        logger.error("Embedding generation failed: %s", exc, exc_info=True)
        return False


def test_collection_persistence():
    """Test if vectors persist in collection after creation."""
    logger.info("=" * 80)
    logger.info("TEST 3: Collection Persistence")
    logger.info("=" * 80)
    
    from qdrant_client import QdrantClient
    
    client = QdrantClient(path=config.QDRANT_LOCAL_PATH)
    
    # List all collections
    logger.info("Listing all Qdrant collections:")
    try:
        collections = client.get_collections()
        if not collections.collections:
            logger.warning("No collections found in Qdrant!")
        else:
            for collection in collections.collections:
                points_count = client.get_collection(collection.name).points_count
                logger.info("  - %s: %d vectors", collection.name, points_count)
        return True
    except Exception as exc:
        logger.error("Failed to list collections: %s", exc, exc_info=True)
        return False


def test_query_empty_collection():
    """Test querying a collection to see if it's actually empty."""
    logger.info("=" * 80)
    logger.info("TEST 4: Query Collection Content")
    logger.info("=" * 80)
    
    from qdrant_client import QdrantClient
    
    client = QdrantClient(path=config.QDRANT_LOCAL_PATH)
    
    # Try to query an existing collection
    test_collection = "test_vectorization"
    
    if not client.collection_exists(test_collection):
        logger.warning("Collection '%s' does not exist", test_collection)
        return False
    
    try:
        collection_info = client.get_collection(test_collection)
        logger.info("Collection '%s' info:", test_collection)
        logger.info("  Points count: %d", collection_info.points_count)
        logger.info("  Vectors count: %d", collection_info.vectors_count)
        logger.info("  Status: %s", collection_info.status)
        
        # Try to get a few points
        if collection_info.points_count > 0:
            points = client.scroll(collection_info.name, limit=5)
            logger.info("First %d points in collection:", min(5, len(points[0])))
            for i, point in enumerate(points[0], 1):
                logger.info(
                    "  Point %d: ID=%s, has_vector=%s, payload_keys=%s",
                    i,
                    point.id,
                    point.vector is not None,
                    list(point.payload.keys()) if point.payload else "none",
                )
        
        return True
    except Exception as exc:
        logger.error("Failed to query collection: %s", exc, exc_info=True)
        return False


if __name__ == "__main__":
    logger.info("Starting Qdrant Vectorization Diagnostics")
    logger.info("Qdrant Mode: %s", config.QDRANT_MODE)
    logger.info("Qdrant Path: %s", config.QDRANT_LOCAL_PATH)
    logger.info("")
    
    results = []
    
    # Run tests
    results.append(("Embedding Generation", test_embedding_generation()))
    results.append(("Collection Persistence", test_collection_persistence()))
    results.append(("Basic Vectorization", test_basic_vectorization()))
    results.append(("Query Collection", test_query_empty_collection()))
    
    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info("%s: %s", test_name, status)
    
    if all(result[1] for result in results):
        logger.info("")
        logger.info("All tests passed! Qdrant vectorization is working correctly.")
    else:
        logger.error("")
        logger.error("Some tests failed. Check the logs above for details.")
