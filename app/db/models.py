"""Database models using SQLModel.

Defines the core data models for the Template Intelligence Engine:
- Organization: Multi-tenancy root
- TemplateStorage: Template metadata and processing status
- AnalysisPrompt: Custom prompts for template analysis
"""

import datetime
import enum
import uuid
from typing import Any

from sqlalchemy import Column, DateTime, Enum as SQLEnum, ForeignKey, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlmodel import Field, SQLModel


class TemplateStatus(str, enum.Enum):
    """Status of template processing pipeline.

    The pipeline follows these states:
    QUEUED -> ANALYZING -> COMPLETED -> FINALIZING -> COMPLETED
              |_____________________________|
                                  v
                               FAILED
    """

    QUEUED = "queued"
    ANALYZING = "analyzing"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"


# =============================================================================
# Database Models
# =============================================================================


class Organization(SQLModel, table=True):
    """Organization model for multi-tenancy.

    All templates belong to an organization, ensuring data isolation between tenants.
    """

    __tablename__ = "organizations"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        sa_column=Column(UUID(as_uuid=True), primary_key=True),
    )
    name: str = Field(min_length=1, max_length=255)
    slug: str = Field(min_length=1, max_length=100, unique=True, index=True)
    created_at: datetime.datetime = Field(
        default_factory=datetime.datetime.utcnow,
        sa_column=Column(DateTime(timezone=True), server_default=text("NOW()")),
    )
    updated_at: datetime.datetime = Field(
        default_factory=datetime.datetime.utcnow,
        sa_column=Column(
            DateTime(timezone=True),
            server_default=text("NOW()"),
            onupdate=text("NOW()"),
        ),
    )


# =============================================================================
# Template Storage Models
# =============================================================================


class TemplateBase(SQLModel):
    """Base template fields."""

    name: str = Field(max_length=255)
    original_filename: str = Field(max_length=512)
    file_path: str = Field(max_length=1024)
    description: str | None = Field(default=None, max_length=1024)


class TemplateStorage(TemplateBase, table=True):
    """Template storage model for analyzed Word templates.

    Stores metadata about analyzed templates including detected variables,
    enabling quick retrieval for the inject phase.
    """

    __tablename__ = "templates"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        sa_column=Column(UUID(as_uuid=True), primary_key=True),
    )
    org_id: uuid.UUID = Field(
        sa_column=Column(
            UUID(as_uuid=True),
            ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        )
    )
    # Variables detected during analysis (JSONB for queryability)
    detected_variables: dict[str, Any] | None = Field(
        default=None,
        sa_column=Column(JSONB),
    )
    # Paragraph count for reference
    paragraph_count: int | None = Field(default=None, ge=0)
    # Template processing metadata
    analysis_method: str | None = Field(default="regex", max_length=50)  # "regex" or "llm"
    is_tagged: bool = Field(default=False)  # True if Jinja2 tags have been injected

    # Processing status tracking
    status: TemplateStatus = Field(
        default=TemplateStatus.QUEUED,
        sa_column=Column(SQLEnum(TemplateStatus), default="queued", index=True),
    )
    task_id: str | None = Field(default=None, max_length=255, index=True)
    processing_started_at: datetime.datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True)),
    )
    processing_completed_at: datetime.datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True)),
    )
    error_message: str | None = Field(default=None, max_length=2048)

    # Batch tracking
    batch_id: str | None = Field(default=None, max_length=255, index=True)
    previous_batch_id: str | None = Field(default=None, max_length=255, index=True)  # For queued batches
    batch_status: str | None = Field(default=None, max_length=50)  # "processing", "queued", "completed"

    # Injection tracking
    injection_status: str | None = Field(default=None, max_length=50)  # "queued", "processing", "completed", "failed"
    injection_task_id: str | None = Field(default=None, max_length=255, index=True)
    injection_started_at: datetime.datetime | None = Field(default=None, sa_column=Column(DateTime(timezone=True)))
    injection_completed_at: datetime.datetime | None = Field(default=None, sa_column=Column(DateTime(timezone=True)))
    injection_error_message: str | None = Field(default=None, max_length=2048)

    # Download tracking
    download_ready: bool = Field(default=False)
    download_url: str | None = Field(default=None, max_length=1024)

    created_at: datetime.datetime = Field(
        default_factory=datetime.datetime.utcnow,
        sa_column=Column(DateTime(timezone=True), server_default=text("NOW()")),
    )
    updated_at: datetime.datetime = Field(
        default_factory=datetime.datetime.utcnow,
        sa_column=Column(
            DateTime(timezone=True),
            server_default=text("NOW()"),
            onupdate=text("NOW()"),
        ),
    )


# =============================================================================
# Template Response Models
# =============================================================================


class TemplateRead(TemplateBase):
    """Template response model."""

    id: uuid.UUID
    org_id: uuid.UUID
    detected_variables: dict[str, Any] | None
    paragraph_count: int | None
    analysis_method: str | None
    is_tagged: bool
    status: TemplateStatus
    task_id: str | None
    processing_started_at: datetime.datetime | None
    processing_completed_at: datetime.datetime | None
    error_message: str | None
    batch_id: str | None
    previous_batch_id: str | None
    batch_status: str | None
    download_ready: bool
    download_url: str | None
    injection_status: str | None
    injection_task_id: str | None
    injection_started_at: datetime.datetime | None
    injection_completed_at: datetime.datetime | None
    injection_error_message: str | None
    created_at: datetime.datetime
    updated_at: datetime.datetime


class TemplateCreate(SQLModel):
    """Template creation model."""

    name: str
    original_filename: str
    description: str | None = None


class TemplateListResponse(SQLModel):
    """Response for listing templates."""

    templates: list[TemplateRead]
    total: int


# =============================================================================
# Analysis Prompt Models
# =============================================================================


class AnalysisPromptBase(SQLModel):
    """Base analysis prompt fields."""

    name: str = Field(max_length=255)
    prompt_text: str = Field(max_length=8192)
    is_default: bool = Field(default=False)


class AnalysisPrompt(AnalysisPromptBase, table=True):
    """Analysis prompt model for storing custom prompts.

    Stores custom prompts for template analysis per organization.
    If is_default=True, this prompt will be used when no prompt_id is specified.
    """

    __tablename__ = "analysis_prompts"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        sa_column=Column(UUID(as_uuid=True), primary_key=True),
    )
    org_id: uuid.UUID = Field(
        sa_column=Column(
            UUID(as_uuid=True),
            ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        )
    )
    created_at: datetime.datetime = Field(
        default_factory=datetime.datetime.utcnow,
        sa_column=Column(DateTime(timezone=True), server_default=text("NOW()")),
    )
    updated_at: datetime.datetime = Field(
        default_factory=datetime.datetime.utcnow,
        sa_column=Column(
            DateTime(timezone=True),
            server_default=text("NOW()"),
            onupdate=text("NOW()"),
        ),
    )


class AnalysisPromptRead(AnalysisPromptBase):
    """Analysis prompt response model."""

    id: uuid.UUID
    org_id: uuid.UUID
    created_at: datetime.datetime
    updated_at: datetime.datetime


class AnalysisPromptCreate(SQLModel):
    """Analysis prompt creation model."""

    name: str = Field(max_length=255)
    prompt_text: str = Field(max_length=8192)
    is_default: bool = False


class AnalysisPromptListResponse(SQLModel):
    """Response for listing analysis prompts."""

    prompts: list[AnalysisPromptRead]
    total: int
