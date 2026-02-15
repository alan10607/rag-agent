"""
VectorVault - Qdrant Client Wrapper

Encapsulates all interactions with the Qdrant vector database:
- Connection management
- Collection creation and validation
- Vector upsert (batch)
- Similarity search
"""

from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from app import config
from app.logger import get_logger

logger = get_logger(__name__)


def get_client() -> QdrantClient:
    """Create and return a Qdrant client instance."""
    client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
    logger.info("Connected to Qdrant at %s:%s", config.QDRANT_HOST, config.QDRANT_PORT)
    return client


def ensure_collection(client: QdrantClient, collection_name: str | None = None) -> None:
    """Create the collection if it does not already exist."""
    name = collection_name or config.COLLECTION_NAME
    collections = [c.name for c in client.get_collections().collections]

    if name in collections:
        logger.info("Collection '%s' already exists.", name)
        return

    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(
            size=config.VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )
    logger.info("Created collection '%s' (size=%d, distance=Cosine).", name, config.VECTOR_SIZE)


def upsert_points(
    client: QdrantClient,
    points: list[PointStruct],
    collection_name: str | None = None,
) -> None:
    """Batch upsert points into the collection."""
    name = collection_name or config.COLLECTION_NAME
    batch_size = config.UPSERT_BATCH_SIZE

    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=name, points=batch)
        logger.info("Upserted batch %d-%d (%d points).", i, i + len(batch) - 1, len(batch))

    logger.info("Total upserted: %d points into '%s'.", len(points), name)


def search_vectors(
    client: QdrantClient,
    query_vector: list[float],
    top_k: int | None = None,
    collection_name: str | None = None,
) -> list[dict[str, Any]]:
    """Search for the most similar vectors and return results with payloads."""
    name = collection_name or config.COLLECTION_NAME
    k = top_k or config.DEFAULT_TOP_K

    results = client.query_points(
        collection_name=name,
        query=query_vector,
        limit=k,
        with_payload=True,
    )

    output = []
    for point in results.points:
        output.append(
            {
                "id": point.id,
                "score": point.score,
                "payload": point.payload,
            }
        )

    logger.info("Search returned %d results from '%s'.", len(output), name)
    return output
