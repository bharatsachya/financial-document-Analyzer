"""User management API routes.

Handles CRUD operations for users.
"""

import bcrypt
import hashlib
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.api.schemas import UserCreate, UserListResponse, UserResponse
from app.db.models import User, Organization

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/users", tags=["users"])


def hash_password(password: str) -> str:
    """Hash a password using bcrypt.

    Args:
        password: Plain text password.

    Returns:
        Hashed password.
    """
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


@router.post("", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    user_data: UserCreate,
    session: AsyncSession = Depends(get_db),
) -> UserResponse:
    """Create a new user.

    Args:
        user_data: User creation data.
        session: Database session.

    Returns:
        The created user.

    Raises:
        HTTPException: If user creation fails.
    """
    try:
        logger.info(f"Creating user: {user_data.email}")

        # Check if email already exists
        try:
            existing_query = select(User).where(User.email == user_data.email)
            existing_result = await session.execute(existing_query)
            existing_user = existing_result.scalar_one_or_none()

            if existing_user:
                logger.warning(f"User with email '{user_data.email}' already exists")
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"User with email '{user_data.email}' already exists",
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error checking existing user: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error checking existing user",
            ) from e

        # Verify organization exists
        try:
            org_query = select(Organization).where(Organization.id == user_data.organization_id)
            org_result = await session.execute(org_query)
            organization = org_result.scalar_one_or_none()

            if not organization:
                logger.warning(f"Organization not found: {user_data.organization_id}")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Organization with id '{user_data.organization_id}' not found",
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error checking organization: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error checking organization",
            ) from e

        # Create new user
        try:
            hashed_password = hash_password(user_data.password)

            user = User(
                email=user_data.email,
                full_name=user_data.full_name,
                hashed_password=hashed_password,
                organization_id=user_data.organization_id,
                is_active=True,
            )

            session.add(user)
            await session.commit()
            await session.refresh(user)

            logger.info(f"Created user: {user.id}")

            return UserResponse.model_validate(user)

        except SQLAlchemyError as e:
            logger.error(f"Database error creating user: {e}", exc_info=True)
            await session.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create user",
            ) from e
        except Exception as e:
            logger.error(f"Error creating user: {e}", exc_info=True)
            await session.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error creating user",
            ) from e

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in create_user: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred",
        ) from e


@router.get("", response_model=UserListResponse)
async def list_users(
    session: AsyncSession = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
    organization_id: uuid.UUID | None = None,
) -> UserListResponse:
    """List users with optional filtering.

    Args:
        session: Database session.
        skip: Number of records to skip.
        limit: Maximum number of records to return.
        organization_id: Optional filter by organization ID.

    Returns:
        List of users.
    """
    try:
        logger.info(f"Listing users: skip={skip}, limit={limit}, org_filter={organization_id}")

        try:
            # Build query
            from sqlalchemy import func

            query = select(User)

            if organization_id:
                query = query.where(User.organization_id == organization_id)

            # Get total count
            count_query = select(func.count()).select_from(query.subquery())
            count_result = await session.execute(count_query)
            total = count_result.scalar_one() or 0

            # Get users
            query = query.offset(skip).limit(limit)
            result = await session.execute(query)
            users = result.scalars().all()

            logger.info(f"Found {len(users)} users")

            return UserListResponse(
                users=[UserResponse.model_validate(user) for user in users],
                total=total,
                page=skip // limit + 1 if limit > 0 else 1,
                page_size=limit,
            )

        except Exception as e:
            logger.error(f"Error fetching users: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error fetching users",
            ) from e

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in list_users: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred",
        ) from e


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
) -> UserResponse:
    """Get a specific user by ID.

    Args:
        user_id: The user ID.
        session: Database session.

    Returns:
        The user details.

    Raises:
        HTTPException: If user not found.
    """
    try:
        logger.info(f"Fetching user: {user_id}")

        try:
            query = select(User).where(User.id == user_id)
            result = await session.execute(query)
            user = result.scalar_one_or_none()

            if not user:
                logger.warning(f"User not found: {user_id}")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found",
                )

            logger.info(f"Found user: {user_id}")
            return UserResponse.model_validate(user)

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error fetching user {user_id}: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error fetching user",
            ) from e

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_user: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred",
        ) from e


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
) -> None:
    """Delete a user.

    Args:
        user_id: The user ID to delete.
        session: Database session.

    Raises:
        HTTPException: If user not found.
    """
    try:
        logger.info(f"Deleting user: {user_id}")

        try:
            query = select(User).where(User.id == user_id)
            result = await session.execute(query)
            user = result.scalar_one_or_none()

            if not user:
                logger.warning(f"User not found for deletion: {user_id}")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found",
                )

            await session.delete(user)
            await session.commit()

            logger.info(f"Deleted user: {user_id}")

        except HTTPException:
            raise
        except SQLAlchemyError as e:
            logger.error(f"Database error deleting user: {e}", exc_info=True)
            await session.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete user",
            ) from e
        except Exception as e:
            logger.error(f"Error deleting user: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error deleting user",
            ) from e

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in delete_user: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred",
        ) from e
