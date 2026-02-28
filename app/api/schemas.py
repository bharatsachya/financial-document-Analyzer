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


class TemplatePreviewResponse(BaseModel):
    """Response containing extracted template text for preview."""
    template_id: uuid.UUID
    template_text: str



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


# =============================================================================
# Report Learning & Preference Schemas  (Director-Typist Model)
# =============================================================================


class RewriteSectionRequest(BaseModel):
    """Request to rewrite a section based on natural-language feedback."""

    original_text: str = Field(
        description="The current draft text to rewrite",
        min_length=1,
    )
    user_feedback: str = Field(
        description="Natural-language instruction (e.g., 'Make this more formal')",
        min_length=1,
        max_length=1000,
    )


class RewriteSectionResponse(BaseModel):
    """Response with the rewritten section."""

    new_text: str = Field(description="Rewritten text")
    model: str = Field(default="openai/gpt-4o-mini", description="Model used")


class FeedbackCaptureRequest(BaseModel):
    """Request to capture adviser approval of a rewrite (Director-Typist)."""

    adviser_id: str = Field(
        description="Adviser identifier (e.g., adv_001)",
        min_length=1,
        max_length=50,
    )
    original_text: str = Field(
        description="Original AI-generated text before editing",
        min_length=1,
    )
    chosen_text: str = Field(
        default="",
        description="Approved rewritten text",
    )
    user_feedback: str = Field(
        default="",
        description="The natural-language feedback that drove the rewrite",
    )
    # Legacy compat — if only edited_text is provided, map it to chosen_text
    edited_text: str = Field(
        default="",
        description="(Legacy) Final edited text — use chosen_text instead",
    )
    report_type: str | None = Field(
        default=None,
        description="Type of report (e.g., investment_summary, risk_assessment)",
    )


class FeedbackCaptureResponse(BaseModel):
    """Response for feedback capture."""

    adviser_id: str
    status: str = "queued"
    task_id: str | None = None
    message: str = "Feedback captured and preference learning queued"


class GeneratePersonalizedReportRequest(BaseModel):
    """Request to generate a personalized report."""

    adviser_id: str = Field(
        description="Adviser identifier for preference lookup",
    )
    prompt: str = Field(
        description="Original prompt for report generation",
        min_length=1,
    )
    report_type: str | None = Field(
        default=None,
        description="Type of report being generated",
    )


class GeneratePersonalizedReportResponse(BaseModel):
    """Response for personalized report generation."""

    task_id: str = Field(description="Celery task ID for tracking")
    status: str = "queued"
    message: str = "Report generation with personalization queued"


class GenerateDraftRequest(BaseModel):
    """Request to generate a draft report directly via LLM."""

    adviser_id: str = Field(
        description="Adviser identifier (e.g., adv_001)",
        min_length=1,
    )
    client_id: str = Field(
        description="Client identifier (e.g., client_123)",
        min_length=1,
    )
    topic: str = Field(
        description="The topic or context of the report (e.g., 'Portfolio Review')",
        min_length=1,
    )
    template_id: uuid.UUID | None = Field(
        default=None,
        description="The ID of the stored template to use",
    )
    template_text: str | None = Field(
        default=None,
        description="The raw template text to fill in (if not using template_id)",
    )


class GenerateDraftResponse(BaseModel):
    """Response containing the generated draft report."""

    report_id: uuid.UUID
    generated_text: str
    extracted_variables: dict[str, Any]
    sources: dict[str, list[str]]
    session_id: uuid.UUID
    timestamp: str
    version_id: uuid.UUID | None = None
    version_number: int | None = None


class DraftVersionListItem(BaseModel):
    """Summary of a draft version for list views."""
    id: uuid.UUID
    version_number: int
    adviser_id: str
    feedback_used: str | None
    generated_text: str
    created_at: datetime

    model_config = {"from_attributes": True}


class DraftVersionListResponse(BaseModel):
    """Response containing a list of draft versions."""
    template_id: uuid.UUID
    versions: list[DraftVersionListItem]
