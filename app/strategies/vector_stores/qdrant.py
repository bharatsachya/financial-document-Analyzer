"""Qdrant vector store implementation.

Provides async operations for storing and searching vectors
in Qdrant with tenant isolation support.
"""

import hashlib
import logging
from collections.abc import AsyncGenerator
from typing import Any, Literal

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.interfaces.vector_store import BaseVectorStore, SearchResult, UpsertResult

logger = logging.getLogger(__name__)


class QdrantStore(BaseVectorStore):
    """Vector store implementation using Qdrant.

    Supports:
    - Tenant isolation via required org_id in filters
    - Automatic collection creation
    - Async operations
    - Configurable distance metrics

    Attributes:
        client: The async Qdrant client instance.
        distance_metric: The distance metric for vector similarity.
    """

    def __init__(
        self,
        url: str,
        api_key: str | None = None,
        distance_metric: Literal["cosine", "euclidean", "dot"] = "cosine",
        timeout: int = 60,
    ) -> None:
        """Initialize the Qdrant store.

        Args:
            url: The Qdrant server URL (e.g., "http://localhost:6333").
            api_key: Optional API key for authentication.
            distance_metric: The distance metric to use.
            timeout: Request timeout in seconds.
        """
        self._client = AsyncQdrantClient(
            url=url,
            api_key=api_key,
            timeout=timeout,
        )
        self._distance_metric = distance_metric

    async def upsert(
        self,
        collection_name: str,
        chunks: list[str],
        embeddings: list[list[float]],
        metadata: list[dict[str, Any]],
        ids: list[str] | None = None,
    ) -> UpsertResult:
        """Insert or update vectors in the collection.

        Args:
            collection_name: Name of the target collection.
            chunks: Text content for each vector.
            embeddings: Embedding vectors.
            metadata: Payload metadata for each vector.
            ids: Optional vector IDs. Generated deterministically if None.

        Returns:
            UpsertResult with count of upserted vectors.

        Raises:
            ValueError: If chunk/embedding/metadata counts don't match.
            QdrantError: If the upsert operation fails.
        """
        n = len(chunks)
        if len(embeddings) != n or len(metadata) != n:
            raise ValueError(
                f"Length mismatch: chunks={n}, embeddings={len(embeddings)}, metadata={len(metadata)}"
            )

        # Ensure collection exists
        await self._ensure_collection(collection_name, len(embeddings[0]))

        # Generate IDs if not provided
        if ids is None:
            ids = [self._generate_id(collection_name, chunk) for chunk in chunks]

        # Prepare points
        points = [
            models.PointStruct(
                id=idx,
                vector=vector,
                payload={
                    "content": content,
                    **meta,
                },
            )
            for idx, (content, vector, meta) in enumerate(zip(ids, embeddings, metadata))
        ]

        try:
            operation_info = await self._client.upsert(
                collection_name=collection_name,
                points=points,
            )

            logger.info(
                f"Upserted {len(points)} vectors into collection '{collection_name}'"
            )

            return UpsertResult(
                upserted_count=len(points),
                operation_id=str(operation_info.status),
            )

        except UnexpectedResponse as e:
            logger.error(f"Qdrant upsert failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during upsert: {e}")
            raise RuntimeError(f"Upsert failed: {e}") from e

    async def search(
        self,
        collection_name: str,
        query_embedding: list[float],
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        """Search for similar vectors in the collection.

        Args:
            collection_name: Name of the collection to search.
            query_embedding: Query vector.
            limit: Maximum results to return.
            filters: Optional filter conditions (must include org_id for tenant isolation).
            score_threshold: Optional minimum similarity threshold.

        Returns:
            List of SearchResults ranked by similarity.

        Raises:
            QdrantError: If the search operation fails.
        """
        # Build filter from filters dict
        qdrant_filter = self._build_filter(filters)

        try:
            search_result = await self._client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                query_filter=qdrant_filter,
                limit=limit,
                score_threshold=score_threshold,
            )

            results = [
                SearchResult(
                    content=str(point.payload.get("content", "")),
                    score=point.score,
                    metadata=point.payload,
                )
                for point in search_result
            ]

            logger.info(f"Found {len(results)} results in collection '{collection_name}'")
            return results

        except UnexpectedResponse as e:
            logger.error(f"Qdrant search failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during search: {e}")
            raise RuntimeError(f"Search failed: {e}") from e

    async def create_collection(
        self,
        collection_name: str,
        vector_size: int,
    ) -> None:
        """Create a new collection.

        Args:
            collection_name: Name for the new collection.
            vector_size: Dimension of vectors to be stored.
        """
        if await self.collection_exists(collection_name):
            logger.info(f"Collection '{collection_name}' already exists")
            return

        try:
            await self._client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=self._map_distance_metric(self._distance_metric),
                ),
            )

            logger.info(f"Created collection '{collection_name}' with vector size {vector_size}")

        except UnexpectedResponse as e:
            logger.error(f"Failed to create collection: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error creating collection: {e}")
            raise RuntimeError(f"Collection creation failed: {e}") from e

    async def delete_collection(self, collection_name: str) -> None:
        """Delete a collection.

        Args:
            collection_name: Name of the collection to delete.
        """
        try:
            await self._client.delete_collection(collection_name=collection_name)
            logger.info(f"Deleted collection '{collection_name}'")

        except UnexpectedResponse as e:
            logger.error(f"Failed to delete collection: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error deleting collection: {e}")
            raise RuntimeError(f"Collection deletion failed: {e}") from e

    async def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists.

        Args:
            collection_name: Name of the collection to check.

        Returns:
            True if collection exists, False otherwise.
        """
        try:
            collections = await self._client.get_collections()
            return any(c.name == collection_name for c in collections.collections)

        except Exception as e:
            logger.error(f"Error checking collection existence: {e}")
            return False

    def _generate_id(self, collection_name: str, content: str) -> str:
        """Generate a deterministic ID for a vector.

        Args:
            collection_name: Name of the collection.
            content: Content string to hash.

        Returns:
            A hexadecimal string ID.
        """
        hash_input = f"{collection_name}:{content}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:32]

    async def _ensure_collection(self, collection_name: str, vector_size: int) -> None:
        """Ensure collection exists, creating if necessary.

        Args:
            collection_name: Name of the collection.
            vector_size: Dimension of vectors.
        """
        if not await self.collection_exists(collection_name):
            await self.create_collection(collection_name, vector_size)

    def _build_filter(self, filters: dict[str, Any] | None) -> models.Filter | None:
        """Build a Qdrant filter from a dictionary.

        Args:
            filters: Dictionary of filter conditions.

        Returns:
            A Qdrant Filter object or None.
        """
        if not filters:
            return None

        conditions = []
        for key, value in filters.items():
            if isinstance(value, (str, int, float, bool)):
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value),
                    )
                )

        if not conditions:
            return None

        return models.Filter(must=conditions)

    def _map_distance_metric(
        self, metric: Literal["cosine", "euclidean", "dot"]
    ) -> models.Distance:
        """Map string metric name to Qdrant Distance enum.

        Args:
            metric: The metric name.

        Returns:
            The corresponding Qdrant Distance enum.
        """
        mapping: dict[str, models.Distance] = {
            "cosine": models.Distance.COSINE,
            "euclidean": models.Distance.EUCLID,
            "dot": models.Distance.DOT,
        }
        return mapping.get(metric, models.Distance.COSINE)
