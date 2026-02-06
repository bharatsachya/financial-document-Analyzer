"""API request and response schemas.

Pydantic v2 models for API serialization/deserialization.
"""

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

# Re-export template-related models for backwards compatibility
from app.strategies.template_engine import DetectedVariable


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
