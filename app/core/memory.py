"""Semantic Memory Manager using Mem0 for tone/style preferences.

This module provides an async interface to Mem0 for storing and retrieving
adviser style preferences. Mem0 is configured to use the local Qdrant
instance as the vector store and OpenRouter (text-embedding-3-small) as
the embedder.

Architecture:
- Semantic Memory (Tone/Style): Mem0 library backed by local Qdrant
- Procedural Memory (Logic/Audits): PostgreSQL (see app.db.models)
- Episodic Memory (Events): PostgreSQL (see app.db.models)
- Factual Data (Graph): Neo4j (see app.db.graph)
"""

import logging
from typing import Any

from mem0 import Memory
from openai import AsyncOpenAI

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


class SemanticMemoryManager:
    """Manages semantic memory for adviser style preferences using Mem0.

    Wraps Mem0 operations in async methods with proper error handling
    for connection timeouts and service unavailability.
    """

    # Collection name for Mem0/Qdrant
    _COLLECTION_NAME = "adviser_style_preferences"

    def __init__(
        self,
        settings: Settings | None = None,
        openai_client: AsyncOpenAI | None = None,
    ) -> None:
        """Initialize the SemanticMemoryManager.

        Args:
            settings: Optional application settings. Defaults to get_settings().
            openai_client: Optional pre-configured OpenAI client for embeddings.
        """
        self._settings = settings or get_settings()
        self._openai_client = openai_client
        self._mem0_client: Memory | None = None
        self._initialized: bool = False

    async def _initialize(self) -> None:
        """Lazy initialization of Mem0 client.

        Creates the Mem0 client only when first needed to avoid
        unnecessary connection attempts.
        """
        if self._initialized:
            return

        try:
            # Configure Mem0 to use local Qdrant
            config = {
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "url": self._settings.qdrant_url,
                        "api_key": self._settings.qdrant_api_key,
                        "collection_name": self._COLLECTION_NAME,
                    },
                },
                "embedder": {
                    "provider": "openai",
                    "config": {
                        "model": self._settings.llm_embedding_model,
                        "api_key": self._settings.openrouter_api_key,
                        "openai_base_url": self._settings.openai_base_url,
                    },
                },
            }

            self._mem0_client = Memory.from_config(config)
            self._initialized = True
            logger.info(
                f"SemanticMemoryManager initialized: Qdrant at {self._settings.qdrant_url}, "
                f"embedder: {self._settings.llm_embedding_model}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize Mem0 client: {e}", exc_info=True)
            # Don't raise - allow graceful degradation

    async def add_preference(
        self,
        adviser_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Add a style preference for an adviser.

        Stores the preference text as a semantic memory entry. The text
        is embedded and stored in Qdrant for later retrieval via
        semantic search.

        Args:
            adviser_id: The adviser identifier (used as user_id in Mem0).
            text: The preference text to store (e.g., "Make reports more formal").
            metadata: Optional additional metadata for the preference.

        Returns:
            The memory ID of the created preference entry.

        Raises:
            RuntimeError: If Mem0 is unavailable after retries.

        Example:
            >>> manager = SemanticMemoryManager()
            >>> memory_id = await manager.add_preference(
            ...     adviser_id="adv_001",
            ...     text="Use formal language for client communications"
            ... )
        """
        await self._initialize()

        if not self._mem0_client:
            raise RuntimeError(
                "Mem0 client is not available. Cannot add preference."
            )

        try:
            # Prepare user metadata
            user_metadata = {"user_id": adviser_id}
            if metadata:
                user_metadata.update(metadata)

            # Add memory to Mem0
            result = self._mem0_client.add(
                text,
                user_id=adviser_id,
                metadata=user_metadata,
            )

            memory_id = str(result.get("results", [{}])[0].get("id", ""))
            logger.info(
                f"Added preference for adviser {adviser_id}: "
                f"'{text[:60]}...' (id: {memory_id})"
            )
            return memory_id

        except ConnectionError as e:
            logger.error(f"Connection error adding preference: {e}", exc_info=True)
            raise RuntimeError(f"Unable to connect to vector store: {e}") from e
        except TimeoutError as e:
            logger.error(f"Timeout adding preference: {e}", exc_info=True)
            raise RuntimeError(f"Vector store request timed out: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error adding preference: {e}", exc_info=True)
            raise RuntimeError(f"Failed to add preference: {e}") from e

    async def search_preferences(
        self,
        adviser_id: str,
        topic: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Search for relevant style preferences for an adviser.

        Performs semantic search on stored preferences to find entries
        relevant to the given topic.

        Args:
            adviser_id: The adviser identifier to search for.
            topic: The topic/query to search for (e.g., "formal tone").
            limit: Maximum number of results to return.

        Returns:
            List of preference dicts with keys:
            - id: Memory entry ID
            - memory: The preference text
            - score: Relevance score (lower is more relevant)
            - metadata: Associated metadata

        Returns empty list if Mem0 is unavailable.

        Example:
            >>> manager = SemanticMemoryManager()
            >>> preferences = await manager.search_preferences(
            ...     adviser_id="adv_001",
            ...     topic="report tone"
            ... )
        """
        await self._initialize()

        if not self._mem0_client:
            logger.warning(
                "Mem0 client unavailable, returning empty preference list"
            )
            return []

        try:
            # Search memories for this adviser
            results = self._mem0_client.search(
                query=topic,
                user_id=adviser_id,
                limit=limit,
            )

            # Normalize results to consistent format
            preferences = []
            for result in results:
                preferences.append(
                    {
                        "id": result.get("id", ""),
                        "memory": result.get("memory", ""),
                        "score": result.get("score", 0.0),
                        "metadata": result.get("metadata", {}),
                        "created_at": result.get("created_at", ""),
                    }
                )

            logger.info(
                f"Found {len(preferences)} preferences for adviser {adviser_id} "
                f"on topic '{topic}'"
            )
            return preferences

        except ConnectionError as e:
            logger.error(f"Connection error searching preferences: {e}", exc_info=True)
            return []
        except TimeoutError as e:
            logger.error(f"Timeout searching preferences: {e}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"Unexpected error searching preferences: {e}", exc_info=True)
            return []

    async def get_all_preferences(
        self,
        adviser_id: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get all stored preferences for an adviser.

        Args:
            adviser_id: The adviser identifier.
            limit: Maximum number of results to return.

        Returns:
            List of all preference entries for the adviser.
            Returns empty list if Mem0 is unavailable.
        """
        await self._initialize()

        if not self._mem0_client:
            logger.warning(
                "Mem0 client unavailable, returning empty preference list"
            )
            return []

        try:
            # Get all memories for this adviser
            results = self._mem0_client.get_all(user_id=adviser_id, limit=limit)

            # Normalize results
            preferences = [
                {
                    "id": r.get("id", ""),
                    "memory": r.get("memory", ""),
                    "metadata": r.get("metadata", {}),
                    "created_at": r.get("created_at", ""),
                }
                for r in results
            ]

            logger.info(f"Retrieved {len(preferences)} total preferences for adviser {adviser_id}")
            return preferences

        except ConnectionError as e:
            logger.error(f"Connection error getting all preferences: {e}", exc_info=True)
            return []
        except TimeoutError as e:
            logger.error(f"Timeout getting all preferences: {e}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"Unexpected error getting all preferences: {e}", exc_info=True)
            return []

    async def delete_preference(self, memory_id: str) -> bool:
        """Delete a specific preference by ID.

        Args:
            memory_id: The memory entry ID to delete.

        Returns:
            True if deleted successfully, False otherwise.
        """
        await self._initialize()

        if not self._mem0_client:
            logger.warning("Mem0 client unavailable, cannot delete preference")
            return False

        try:
            self._mem0_client.delete(memory_id)
            logger.info(f"Deleted preference: {memory_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting preference {memory_id}: {e}", exc_info=True)
            return False

    async def delete_all_preferences(self, adviser_id: str) -> bool:
        """Delete all preferences for an adviser.

        Args:
            adviser_id: The adviser identifier.

        Returns:
            True if deleted successfully, False otherwise.
        """
        await self._initialize()

        if not self._mem0_client:
            logger.warning("Mem0 client unavailable, cannot delete preferences")
            return False

        try:
            self._mem0_client.delete_all(user_id=adviser_id)
            logger.info(f"Deleted all preferences for adviser {adviser_id}")
            return True

        except Exception as e:
            logger.error(
                f"Error deleting all preferences for adviser {adviser_id}: {e}",
                exc_info=True,
            )
            return False

    def is_available(self) -> bool:
        """Check if the memory manager is available.

        Returns:
            True if initialized and Mem0 client is ready.
        """
        return self._initialized and self._mem0_client is not None


# =============================================================================
# Global Instance for Convenience
# =============================================================================

_global_manager: SemanticMemoryManager | None = None


async def get_memory_manager() -> SemanticMemoryManager:
    """Get or create the global SemanticMemoryManager instance.

    Returns:
        The singleton SemanticMemoryManager instance.
    """
    global _global_manager
    if _global_manager is None:
        _global_manager = SemanticMemoryManager()
        await _global_manager._initialize()
    return _global_manager
