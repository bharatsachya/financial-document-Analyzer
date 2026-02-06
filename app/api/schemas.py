"""API request and response schemas.

Pydantic v2 models for API serialization/deserialization.
"""

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, EmailStr, Field
from pydantic.types import PositiveInt

from app.db.models import DocumentStatus

# Re-export template-related models for backwards compatibility
from app.strategies.template_engine import DetectedVariable


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


# =============================================================================
# Template Schemas
# =============================================================================


class TemplateAnalysisResponse(BaseModel):
    """Response from template analysis."""

    template_id: uuid.UUID
    filename: str
    detected_variables: list[DetectedVariable]
    total_paragraphs: int
    analyzed_at: datetime

    model_config = {"from_attributes": True}


class TemplateRenderRequest(BaseModel):
    """Request to render a template with context data."""

    template_id: uuid.UUID
    context_data: dict[str, Any]
    output_filename: str | None = None


# =============================================================================
# Batch Processing & Status Schemas
# =============================================================================


class TemplateStatusResponse(BaseModel):
    """Response for template status queries."""

    template_id: uuid.UUID
    filename: str
    status: str  # TemplateStatus as string
    progress: int | None = Field(default=None, description="0-100 progress percentage")
    detected_variables: dict[str, Any] | None = None
    download_ready: bool
    download_url: str | None = None
    error_message: str | None = None
    created_at: datetime
    processing_started_at: datetime | None = None
    processing_completed_at: datetime | None = None
    batch_id: str | None = None

    model_config = {"from_attributes": True}


class BatchStatusResponse(BaseModel):
    """Response for batch status queries."""

    batch_id: str
    batch_status: str | None = Field(description="processing, queued, or completed")
    total_templates: int
    completed: int = Field(description="Count of completed templates")
    failed: int = Field(description="Count of failed templates")
    in_progress: int = Field(description="Count of templates currently processing")
    queued: int = Field(default=0, description="Count of files waiting in next batch")
    next_batch_id: str | None = None
    previous_batch_id: str | None = None
    templates: list[TemplateStatusResponse]


class BatchUploadResponse(BaseModel):
    """Response for batch upload."""

    batch_id: str
    template_count: int
    message: str


class RandomInjectRequest(BaseModel):
    """Request for random variable injection."""

    template_id: uuid.UUID
    seed: int | None = Field(default=None, description="Optional seed for reproducible random values")


class InjectFinalizeRequest(BaseModel):
    """Request to finalize template with variable injection."""

    template_id: uuid.UUID
    variables: list[dict[str, Any]] = Field(description="Variables with values to inject")


class InjectFinalizeResponse(BaseModel):
    """Response for template finalization."""

    template_id: uuid.UUID
    status: str = "queued"
    task_id: str | None = None
    message: str = "Template finalization queued"
