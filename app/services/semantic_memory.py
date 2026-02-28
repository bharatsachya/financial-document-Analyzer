"""Semantic Memory Manager for tone and style preferences.

Wraps the QdrantStyleMemory client to provide a high-level interface
for semantic memory operations. This is the "what" layer - storing
stylistic preferences and tone patterns.
"""

import logging
from typing import Any

from app.core.config import Settings, get_settings
from app.db.qdrant_client import (
    ensure_collection,
    list_preferences,
    query_preferences,
    upsert_preference,
)

logger = logging.getLogger(__name__)


class SemanticMemoryManager:
    """Manager for semantic memory operations.

    Provides methods to store, retrieve, and search for stylistic
    preferences learned from adviser feedback.
    """

    def __init__(
        self,
        settings: Settings | None = None,
    ) -> None:
        """Initialize SemanticMemoryManager.

        Args:
            settings: Optional application settings.
        """
        self._settings = settings or get_settings()

    def ensure_collection_exists(self) -> None:
        """Ensure the Qdrant collection for preferences exists."""
        ensure_collection(self._settings)

    def search_preferences(
        self,
        adviser_id: str,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Search for semantic preferences based on query embedding.

        Retrieves the top-k most relevant stylistic preferences
        for the given adviser.

        Args:
            adviser_id: The adviser to search preferences for.
            query_embedding: Embedding vector for the query/topic.
            top_k: Number of results to return.

        Returns:
            List of preference dicts with keys: id, score, rule_text,
            adviser_id, created_at.
        """
        try:
            results = query_preferences(
                adviser_id=adviser_id,
                query_embedding=query_embedding,
                top_k=top_k,
                settings=self._settings,
            )
            logger.info(
                f"Retrieved {len(results)} semantic preferences "
                f"for adviser {adviser_id}"
            )
            return results

        except Exception as e:
            logger.error(
                f"Error searching semantic preferences for adviser {adviser_id}: {e}",
                exc_info=True,
            )
            return []

    def store_preference(
        self,
        *,
        adviser_id: str,
        rule_text: str,
        embedding: list[float],
        org_id: str = "",
        example_original: str = "",
        example_edited: str = "",
    ) -> str:
        """Store a new stylistic preference.

        Args:
            adviser_id: Adviser who provided the preference.
            rule_text: The stylistic rule/feedback.
            embedding: 1536-dim embedding vector.
            org_id: Organization ID for multi-tenancy.
            example_original: Optional original text snippet.
            example_edited: Optional edited text snippet.

        Returns:
            The UUID of the stored preference point.

        Raises:
            Exception: If storage fails.
        """
        try:
            self.ensure_collection_exists()

            point_id = upsert_preference(
                adviser_id=adviser_id,
                rule_text=rule_text,
                embedding=embedding,
                org_id=org_id,
                example_original=example_original,
                example_edited=example_edited,
                settings=self._settings,
            )

            logger.info(
                f"Stored semantic preference {point_id} "
                f"for adviser {adviser_id}: {rule_text[:80]}"
            )
            return point_id

        except Exception as e:
            logger.error(
                f"Error storing semantic preference for adviser {adviser_id}: {e}",
                exc_info=True,
            )
            raise

    def list_all_preferences(
        self,
        adviser_id: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """List all preferences for an adviser.

        Used for display in UI panels.

        Args:
            adviser_id: The adviser to list preferences for.
            limit: Maximum number of preferences to return.

        Returns:
            List of preference dicts.
        """
        try:
            results = list_preferences(
                adviser_id=adviser_id,
                limit=limit,
                settings=self._settings,
            )
            logger.info(
                f"Listed {len(results)} semantic preferences "
                f"for adviser {adviser_id}"
            )
            return results

        except Exception as e:
            logger.error(
                f"Error listing semantic preferences for adviser {adviser_id}: {e}",
                exc_info=True,
            )
            return []


# Global instance
_global_manager: SemanticMemoryManager | None = None


def get_semantic_memory_manager() -> SemanticMemoryManager:
    """Get the global SemanticMemoryManager instance.

    Returns:
        The singleton SemanticMemoryManager.
    """
    global _global_manager
    if _global_manager is None:
        _global_manager = SemanticMemoryManager()
    return _global_manager
