"""FastAPI dependencies for dependency injection.

Provides reusable dependencies for routes including:
- Database sessions
- Authentication
- Organization context
"""

import logging
import uuid
from collections.abc import AsyncGenerator

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.db.models import User, UserRead
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


async def get_current_user(
    token: str | None = Header(default=None, alias="Authorization", description="Bearer token"),
    session: AsyncSession = Depends(get_db),
) -> User:
    """Dependency for getting the authenticated user from token.

    Args:
        token: The bearer token from Authorization header.
        session: Database session.

    Returns:
        The authenticated user.

    Raises:
        HTTPException: If token is missing or user not found.
    """
    try:
        if not token:
            logger.warning("Authorization header is missing")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Not authenticated",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Strip "Bearer " prefix if present
        if token.startswith("Bearer "):
            token = token[7:]

        # TODO: Implement proper JWT validation
        # For now, this is a placeholder that demonstrates the pattern
        # In production, decode JWT and fetch user from database
        logger.warning("Authentication not yet implemented")

        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Authentication not yet implemented",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_current_user: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing authentication",
        ) from e


async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """Dependency for getting the authenticated active user.

    Args:
        current_user: The authenticated user.

    Returns:
        The authenticated user if active.

    Raises:
        HTTPException: If user is inactive.
    """
    try:
        if not current_user.is_active:
            logger.warning(f"Inactive user attempted access: {current_user.id}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Inactive user",
            )

        return current_user

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_current_active_user: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error verifying user status",
        ) from e
