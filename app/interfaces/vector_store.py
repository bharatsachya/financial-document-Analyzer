"""Abstract base class for vector storage strategies.

The Strategy Pattern allows different vector databases
to be used interchangeably.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SearchResult:
    """Represents a search result from the vector store.

    Attributes:
        content: The text content of the matching chunk.
        score: The similarity score (lower is better for most implementations).
        metadata: The associated metadata from the vector store.
    """

    content: str
    score: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class UpsertResult:
    """Result of an upsert operation.

    Attributes:
        upserted_count: Number of vectors upserted.
        operation_id: Optional identifier for the operation.
    """

    upserted_count: int
    operation_id: str | None = None


class BaseVectorStore(ABC):
    """Abstract base class for vector storage strategies.

    All concrete vector store implementations must inherit from this class
    and implement the required methods.

    Example:
        ```python
        class QdrantStore(BaseVectorStore):
            async def upsert(
                self,
                collection_name: str,
                chunks: list[Chunk],
                embeddings: list[list[float]],
            ) -> UpsertResult:
                # Insert into Qdrant
                pass
        ```
    """

    @abstractmethod
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
            collection_name: The name of the collection to upsert into.
            chunks: The text chunks corresponding to the embeddings.
            embeddings: The embedding vectors for each chunk.
            metadata: Metadata payloads for each chunk.
            ids: Optional list of IDs for each vector. If None, generates IDs.

        Returns:
            An UpsertResult containing the count of upserted vectors.

        Raises:
            VectorStoreError: If the upsert operation fails.
        """
        ...

    @abstractmethod
    async def search(
        self,
        collection_name: str,
        query_embedding: list[float],
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar vectors in the collection.

        Args:
            collection_name: The name of the collection to search.
            query_embedding: The query vector to search with.
            limit: Maximum number of results to return.
            filters: Optional filters to apply to the search.

        Returns:
            A list of SearchResult objects ranked by similarity.

        Raises:
            VectorStoreError: If the search operation fails.
        """
        ...

    @abstractmethod
    async def create_collection(
        self,
        collection_name: str,
        vector_size: int,
    ) -> None:
        """Create a new collection if it doesn't exist.

        Args:
            collection_name: The name of the collection to create.
            vector_size: The dimension of vectors to be stored.

        Raises:
            VectorStoreError: If collection creation fails.
        """
        ...

    @abstractmethod
    async def delete_collection(self, collection_name: str) -> None:
        """Delete a collection.

        Args:
            collection_name: The name of the collection to delete.

        Raises:
            VectorStoreError: If deletion fails.
        """
        ...

    @abstractmethod
    async def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists.

        Args:
            collection_name: The name of the collection to check.

        Returns:
            True if the collection exists, False otherwise.
        """
        ...
