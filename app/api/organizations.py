"""Organization management API routes.

Handles CRUD operations for organizations.
"""

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.api.schemas import OrganizationCreate, OrganizationListResponse, OrganizationResponse
from app.db.models import Organization

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/organizations", tags=["organizations"])


@router.post("", response_model=OrganizationResponse, status_code=status.HTTP_201_CREATED)
async def create_organization(
    org_data: OrganizationCreate,
    session: AsyncSession = Depends(get_db),
) -> OrganizationResponse:
    """Create a new organization.

    Args:
        org_data: Organization creation data.
        session: Database session.

    Returns:
        The created organization.

    Raises:
        HTTPException: If organization creation fails.
    """
    try:
        logger.info(f"Creating organization: {org_data.name}")

        # Check if slug already exists
        try:
            existing_query = select(Organization).where(Organization.slug == org_data.slug)
            existing_result = await session.execute(existing_query)
            existing_org = existing_result.scalar_one_or_none()

            if existing_org:
                logger.warning(f"Organization with slug '{org_data.slug}' already exists")
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Organization with slug '{org_data.slug}' already exists",
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error checking existing organization: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error checking existing organization",
            ) from e

        # Create new organization
        try:
            organization = Organization(
                name=org_data.name,
                slug=org_data.slug,
            )

            session.add(organization)
            await session.commit()
            await session.refresh(organization)

            logger.info(f"Created organization: {organization.id}")

            return OrganizationResponse.model_validate(organization)

        except SQLAlchemyError as e:
            logger.error(f"Database error creating organization: {e}", exc_info=True)
            await session.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create organization",
            ) from e
        except Exception as e:
            logger.error(f"Error creating organization: {e}", exc_info=True)
            await session.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error creating organization",
            ) from e

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in create_organization: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred",
        ) from e


@router.get("", response_model=OrganizationListResponse)
async def list_organizations(
    session: AsyncSession = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
) -> OrganizationListResponse:
    """List all organizations.

    Args:
        session: Database session.
        skip: Number of records to skip.
        limit: Maximum number of records to return.

    Returns:
        List of organizations.
    """
    try:
        logger.info(f"Listing organizations: skip={skip}, limit={limit}")

        try:
            # Get total count
            from sqlalchemy import func

            count_query = select(func.count()).select_from(Organization)
            count_result = await session.execute(count_query)
            total = count_result.scalar_one() or 0

            # Get organizations
            query = select(Organization).offset(skip).limit(limit)
            result = await session.execute(query)
            organizations = result.scalars().all()

            logger.info(f"Found {len(organizations)} organizations")

            return OrganizationListResponse(
                organizations=[
                    OrganizationResponse.model_validate(org) for org in organizations
                ],
                total=total,
            )

        except Exception as e:
            logger.error(f"Error fetching organizations: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error fetching organizations",
            ) from e

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in list_organizations: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred",
        ) from e


@router.get("/{org_id}", response_model=OrganizationResponse)
async def get_organization(
    org_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
) -> OrganizationResponse:
    """Get a specific organization by ID.

    Args:
        org_id: The organization ID.
        session: Database session.

    Returns:
        The organization details.

    Raises:
        HTTPException: If organization not found.
    """
    try:
        logger.info(f"Fetching organization: {org_id}")

        try:
            query = select(Organization).where(Organization.id == org_id)
            result = await session.execute(query)
            organization = result.scalar_one_or_none()

            if not organization:
                logger.warning(f"Organization not found: {org_id}")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Organization not found",
                )

            logger.info(f"Found organization: {org_id}")
            return OrganizationResponse.model_validate(organization)

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error fetching organization {org_id}: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error fetching organization",
            ) from e

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_organization: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred",
        ) from e


@router.delete("/{org_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_organization(
    org_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
) -> None:
    """Delete an organization.

    Args:
        org_id: The organization ID to delete.
        session: Database session.

    Raises:
        HTTPException: If organization not found.
    """
    try:
        logger.info(f"Deleting organization: {org_id}")

        try:
            query = select(Organization).where(Organization.id == org_id)
            result = await session.execute(query)
            organization = result.scalar_one_or_none()

            if not organization:
                logger.warning(f"Organization not found for deletion: {org_id}")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Organization not found",
                )

            await session.delete(organization)
            await session.commit()

            logger.info(f"Deleted organization: {org_id}")

        except HTTPException:
            raise
        except SQLAlchemyError as e:
            logger.error(f"Database error deleting organization: {e}", exc_info=True)
            await session.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete organization",
            ) from e
        except Exception as e:
            logger.error(f"Error deleting organization: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error deleting organization",
            ) from e

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in delete_organization: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred",
        ) from e
