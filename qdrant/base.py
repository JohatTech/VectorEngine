import logging
import re
import time
from qdrant_client.models import Distance, VectorParams

logger = logging.getLogger("qdrant.base")

def sanitise_collection_name(name: str) -> str:
    """
    Turn a human-readable project name into a safe Qdrant collection name.
    Rules: lowercase, spaces/dashes → underscores, strip non-alphanumerics.
    """
    name = name.lower()
    name = re.sub(r"[\s\-]+", "_", name)
    name = re.sub(r"[^a-z0-9_]", "", name)
    name = name.strip("_")
    return name

def retry_qdrant_operation(operation_name, operation_func, max_retries=5, retry_delay=2.0):
    """
    Execute a Qdrant operation with retry logic and exponential backoff.
    """
    last_exc = None
    
    for attempt in range(1, max_retries + 1):
        try:
            logger.debug("%s attempt %d/%d", operation_name, attempt, max_retries)
            return operation_func()
        except Exception as exc:
            last_exc = exc
            
            if attempt < max_retries:
                wait_time = retry_delay * (2 ** (attempt - 1))
                logger.warning(
                    "%s failed on attempt %d/%d  │  error=%s  │  retrying in %.1fs",
                    operation_name,
                    attempt,
                    max_retries,
                    type(exc).__name__,
                    wait_time,
                )
                time.sleep(wait_time)
            else:
                logger.error(
                    "%s FAILED after %d attempts  │  final_error=%s",
                    operation_name,
                    max_retries,
                    str(exc),
                )
    
    raise last_exc

def verify_collection_has_vectors(client, collection_name):
    """
    Verify that a collection has vectors stored.
    Returns the count of points (vectors) in the collection.
    """
    try:
        collection_info = client.get_collection(collection_name=collection_name)
        point_count = collection_info.points_count
        logger.info("Collection '%s' verification  │  points_count=%d", collection_name, point_count)
        return point_count
    except Exception as exc:
        logger.error("Failed to verify collection '%s': %s", collection_name, exc)
        return 0
