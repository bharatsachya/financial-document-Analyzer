"""API request and response schemas.

Pydantic v2 models for API serialization/deserialization.
"""

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, EmailStr, Field
from pydantic.types import PositiveInt

from app.db.models import DocumentStatus


# =============================================================================
# Document Schemas
# =============================================================================


class DocumentUploadResponse(BaseModel):
    """Response for document upload endpoint."""

    document_id: uuid.UUID = Field(description="ID of the created document record")
    task_id: str | None = Field(default=None, description="Celery task ID for tracking")
    status: DocumentStatus = Field(default=DocumentStatus.QUEUED)
    message: str = Field(default="Document queued for processing")


class DocumentStatusResponse(BaseModel):
    """Response for document status queries."""

    id: uuid.UUID
    filename: str
    status: DocumentStatus
    error_message: str | None = None
    chunk_count: int | None = None
    vector_count: int | None = None
    created_at: datetime
    processing_started_at: datetime | None = None
    processing_completed_at: datetime | None = None

    model_config = {"from_attributes": True}


class DocumentListResponse(BaseModel):
    """Response for listing documents."""

    documents: list[DocumentStatusResponse]
    total: int
    page: int
    page_size: int


# =============================================================================
# Organization Schemas
# =============================================================================


class OrganizationCreate(BaseModel):
    """Request schema for creating an organization."""

    name: str = Field(min_length=1, max_length=255, description="Organization name")
    slug: str = Field(
        min_length=1,
        max_length=100,
        description="URL-friendly identifier for the organization",
    )


class OrganizationResponse(BaseModel):
    """Response schema for organization."""

    id: uuid.UUID
    name: str
    slug: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class OrganizationListResponse(BaseModel):
    """Response for listing organizations."""

    organizations: list[OrganizationResponse]
    total: int


# =============================================================================
# User Schemas
# =============================================================================


class UserCreate(BaseModel):
    """Request schema for creating a user."""

    email: EmailStr = Field(description="User email address")
    full_name: str = Field(
        min_length=1, max_length=255, description="User's full name"
    )
    password: str = Field(
        min_length=8, max_length=100, description="User password (will be hashed)"
    )
    organization_id: uuid.UUID = Field(description="ID of the organization")

    class Config:
        json_schema_extra = {
            "example": {
                "email": "user@example.com",
                "full_name": "John Doe",
                "password": "securepassword123",
                "organization_id": "00000000-0000-0000-0000-000000000000",
            }
        }


class UserResponse(BaseModel):
    """Response schema for user."""

    id: uuid.UUID
    email: str
    full_name: str | None
    is_active: bool
    organization_id: uuid.UUID
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class UserListResponse(BaseModel):
    """Response for listing users."""

    users: list[UserResponse]
    total: int
    page: int = 1
    page_size: int = 20


# =============================================================================
# Error Schemas
# =============================================================================


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str = Field(description="Error message")
    error_code: str | None = Field(default=None, description="Application-specific error code")
    extra: dict[str, Any] | None = Field(default=None, description="Additional error context")
