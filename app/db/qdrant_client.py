"""Qdrant client for persistent style preference memory.

Uses the official `qdrant-client` Python SDK to manage the
`org_style_preferences` collection. This stores embedded stylistic
rules extracted from adviser feedback, enabling RAG-based
report personalization.
"""

import logging
import uuid
from datetime import datetime, timezone

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
EMBEDDING_DIM = 1536  # text-embedding-3-small output dimension
COLLECTION_NAME = "org_style_preferences"


def _build_client(settings: Settings | None = None) -> QdrantClient:
    """Create a QdrantClient from application settings.

    Handles both local (no API key) and cloud (with API key) deployments.
    """
    settings = settings or get_settings()
    kwargs: dict = {"url": settings.qdrant_url, "timeout": 15}
    if settings.qdrant_api_key:
        kwargs["api_key"] = settings.qdrant_api_key
    return QdrantClient(**kwargs)


# ── Collection bootstrap ─────────────────────────────────────────────────────


def ensure_collection(settings: Settings | None = None) -> None:
    """Create the style preferences collection if it doesn't exist.

    Uses cosine distance and dimension 1536 (text-embedding-3-small).
    Safe to call multiple times — no-ops if the collection is already present.
    """
    client = _build_client(settings)
    collection = settings.qdrant_style_collection if settings else COLLECTION_NAME

    if client.collection_exists(collection):
        logger.info(f"Qdrant collection '{collection}' already exists")
        return

    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )
    logger.info(f"Created Qdrant collection '{collection}' (dim={EMBEDDING_DIM}, cosine)")


# ── Upsert ────────────────────────────────────────────────────────────────────


def upsert_preference(
    *,
    adviser_id: str,
    rule_text: str,
    embedding: list[float],
    org_id: str = "",
    example_original: str = "",
    example_edited: str = "",
    settings: Settings | None = None,
) -> str:
    """Store a learned style preference as a vector in Qdrant.

    Args:
        adviser_id: Adviser who triggered the preference.
        rule_text: The stylistic rule / user feedback string.
        embedding: 1536-dim vector from text-embedding-3-small.
        org_id: Organization ID for multi-tenancy.
        example_original: Snippet of original text (optional).
        example_edited: Snippet of rewritten text (optional).
        settings: Optional app settings.

    Returns:
        The point UUID that was upserted.
    """
    settings = settings or get_settings()
    client = _build_client(settings)
    collection = settings.qdrant_style_collection

    if not client.collection_exists(collection):
        ensure_collection(settings)

    point_id = str(uuid.uuid4())

    client.upsert(
        collection_name=collection,
        points=[
            PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "adviser_id": adviser_id,
                    "org_id": org_id,
                    "rule_text": rule_text,
                    "example_original": example_original[:500],
                    "example_edited": example_edited[:500],
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        ],
    )

    logger.info(f"Upserted preference {point_id} for adviser {adviser_id}: {rule_text[:80]}")
    return point_id


# ── Query (RAG retrieval) ────────────────────────────────────────────────────


def query_preferences(
    *,
    adviser_id: str,
    query_embedding: list[float],
    top_k: int = 3,
    settings: Settings | None = None,
) -> list[dict]:
    """Retrieve the top-k style preferences for an adviser via semantic search.

    This is the core RAG retrieval step used before generating any report.
    Results are filtered strictly by adviser_id.

    Args:
        adviser_id: Adviser to filter preferences for.
        query_embedding: Embedding of the current topic / section.
        top_k: Number of results to return.
        settings: Optional app settings.

    Returns:
        List of dicts with `rule_text`, `score`, and metadata.
    """
    settings = settings or get_settings()
    client = _build_client(settings)
    collection = settings.qdrant_style_collection

    if not client.collection_exists(collection):
        logger.info(f"Collection {collection} does not exist yet. Returning empty preferences.")
        return []

    # Strict filter by adviser_id
    adviser_filter = Filter(
        must=[FieldCondition(key="adviser_id", match=MatchValue(value=adviser_id))]
    )

    results = client.query_points(
        collection_name=collection,
        query=query_embedding,
        query_filter=adviser_filter,
        limit=top_k,
        with_payload=True,
    )

    preferences = []
    for point in results.points:
        payload = point.payload or {}
        preferences.append(
            {
                "id": point.id,
                "score": point.score,
                "rule_text": payload.get("rule_text", ""),
                "adviser_id": payload.get("adviser_id", ""),
                "created_at": payload.get("created_at", ""),
            }
        )

    logger.info(f"Retrieved {len(preferences)} preferences for adviser {adviser_id}")
    return preferences


# ── List all (for Memory Insights UI) ────────────────────────────────────────


def list_preferences(
    *,
    adviser_id: str,
    limit: int = 20,
    settings: Settings | None = None,
) -> list[dict]:
    """List all stored preferences for an adviser (no embedding needed).

    Used by the Memory Insights panel in the frontend.

    Args:
        adviser_id: Adviser to filter by.
        limit: Max results.
        settings: Optional app settings.

    Returns:
        List of preference dicts.
    """
    settings = settings or get_settings()
    client = _build_client(settings)
    collection = settings.qdrant_style_collection

    if not client.collection_exists(collection):
        logger.info(f"Collection {collection} does not exist yet. Returning empty preferences list.")
        return []

    adviser_filter = Filter(
        must=[FieldCondition(key="adviser_id", match=MatchValue(value=adviser_id))]
    )

    results, _next_offset = client.scroll(
        collection_name=collection,
        scroll_filter=adviser_filter,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )

    preferences = []
    for point in results:
        payload = point.payload or {}
        preferences.append(
            {
                "id": point.id,
                "rule_text": payload.get("rule_text", ""),
                "adviser_id": payload.get("adviser_id", ""),
                "example_original": payload.get("example_original", ""),
                "example_edited": payload.get("example_edited", ""),
                "created_at": payload.get("created_at", ""),
            }
        )

    logger.info(f"Listed {len(preferences)} preferences for adviser {adviser_id}")
    return preferences
