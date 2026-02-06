"""FastAPI dependencies for dependency injection.

Provides reusable dependencies for routes including:
- Database sessions
- Organization context
"""

import logging
import uuid
from collections.abc import AsyncGenerator

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.db.session import get_async_session

logger = logging.getLogger(__name__)


async def get_db(
    settings: Settings = Depends(get_settings),
) -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting async database sessions.

    Args:
        settings: Application settings.

    Yields:
        An async database session.
    """
    try:
        async for session in get_async_session(settings):
            yield session
    except Exception as e:
        logger.error(f"Error getting database session: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database connection error",
        ) from e


async def get_org_id(
    x_org_id: str | None = Header(default=None, description="Organization ID for multi-tenancy"),
) -> uuid.UUID:
    """Dependency for extracting organization ID from headers.

    Args:
        x_org_id: The organization ID from X-Org-ID header.

    Returns:
        The organization UUID.

    Raises:
        HTTPException: If org_id is missing or invalid.
    """
    try:
        if not x_org_id:
            logger.warning("X-Org-ID header is missing")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="X-Org-ID header is required",
            )

        try:
            return uuid.UUID(x_org_id)
        except ValueError as e:
            logger.warning(f"Invalid organization ID format: {x_org_id}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid organization ID format",
            ) from e

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_org_id: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing organization ID",
        ) from e
