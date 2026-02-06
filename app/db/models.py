"""Database models using SQLModel.

Defines the core data models for the Document Intelligence Platform:
- Organization: Multi-tenancy root
- User: Authenticated users belonging to organizations
- Document: File tracking with processing status
"""

import datetime
import enum
import uuid
from typing import Any

from pydantic import EmailStr
from sqlalchemy import Column, DateTime, Enum as SQLEnum, ForeignKey, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship
from sqlmodel import Field, Relationship, SQLModel


class DocumentStatus(str, enum.Enum):
    """Status of document processing pipeline.

    The pipeline follows these states:
    QUEUED -> PARSING -> CHUNKING -> EMBEDDING -> INDEXING -> COMPLETED
                    |___________________________|
                                      v
                                   FAILED
    """

    QUEUED = "queued"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    COMPLETED = "completed"
    FAILED = "failed"


class TemplateStatus(str, enum.Enum):
    """Status of template processing pipeline.

    The pipeline follows these states:
    QUEUED -> ANALYZING -> COMPLETED -> FINALIZING -> COMPLETED
              |_____________________________||
                                  v
                               FAILED
    """

    QUEUED = "queued"
    ANALYZING = "analyzing"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"


# =============================================================================
# Shared Models (for API responses, not database tables)
# =============================================================================


class OrganizationBase(SQLModel):
    """Base organization fields."""

    name: str = Field(min_length=1, max_length=255)
    slug: str = Field(min_length=1, max_length=100)


class UserBase(SQLModel):
    """Base user fields."""

    email: EmailStr = Field(unique=True, index=True, max_length=255)
    full_name: str | None = Field(default=None, max_length=255)
    is_active: bool = Field(default=True)


class DocumentBase(SQLModel):
    """Base document fields."""

    filename: str = Field(max_length=512)
    file_path: str = Field(max_length=1024)
    file_size: int | None = Field(default=None, ge=0)
    mime_type: str | None = Field(default=None, max_length=100)
    status: DocumentStatus = Field(default=DocumentStatus.QUEUED)
    error_message: str | None = Field(default=None, max_length=2048)


# =============================================================================
# Database Models
# =============================================================================


class Organization(OrganizationBase, table=True):
    """Organization model for multi-tenancy.

    All documents and users belong to an organization, ensuring
    data isolation between tenants.
    """

    __tablename__ = "organizations"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        sa_column=Column(UUID(as_uuid=True), primary_key=True),
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

    # Relationships
    users: list["User"] = Relationship(back_populates="organization")
    documents: list["Document"] = Relationship(back_populates="organization")


class User(UserBase, table=True):
    """User model representing authenticated users.

    Users belong to an organization and can upload documents.
    """

    __tablename__ = "users"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        sa_column=Column(UUID(as_uuid=True), primary_key=True),
    )
    hashed_password: str = Field(max_length=255, exclude=True)
    organization_id: uuid.UUID = Field(
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

    # Relationships
    organization: Organization = Relationship(back_populates="users")
    documents: list["Document"] = Relationship(back_populates="user")


class Document(DocumentBase, table=True):
    """Document model tracking file processing status.

    Stores metadata about uploaded files and tracks their
    progress through the ingestion pipeline.
    """

    __tablename__ = "documents"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        sa_column=Column(UUID(as_uuid=True), primary_key=True),
    )
    user_id: uuid.UUID = Field(
        sa_column=Column(
            UUID(as_uuid=True),
            ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        )
    )
    org_id: uuid.UUID = Field(
        sa_column=Column(
            UUID(as_uuid=True),
            ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        )
    )
    task_id: str | None = Field(default=None, max_length=255, index=True)
    chunk_count: int | None = Field(default=None, ge=0)
    vector_count: int | None = Field(default=None, ge=0)
    file_metadata: dict[str, Any] | None = Field(
        default=None,
        sa_column=Column(JSONB),
    )
    processing_started_at: datetime.datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True)),
    )
    processing_completed_at: datetime.datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True)),
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

    # Relationships
    user: User = Relationship(back_populates="documents")
    organization: Organization = Relationship(back_populates="documents")


# =============================================================================
# Response Models
# =============================================================================


class OrganizationRead(OrganizationBase):
    """Organization response model."""

    id: uuid.UUID
    created_at: datetime.datetime
    updated_at: datetime.datetime


class UserRead(UserBase):
    """User response model."""

    id: uuid.UUID
    organization_id: uuid.UUID
    created_at: datetime.datetime
    updated_at: datetime.datetime


class UserCreate(UserBase):
    """User creation model."""

    password: str = Field(min_length=8, max_length=100)
    organization_id: uuid.UUID


class DocumentRead(DocumentBase):
    """Document response model."""

    id: uuid.UUID
    user_id: uuid.UUID
    org_id: uuid.UUID
    task_id: str | None
    chunk_count: int | None
    vector_count: int | None
    file_metadata: dict[str, Any] | None = None
    processing_started_at: datetime.datetime | None
    processing_completed_at: datetime.datetime | None
    created_at: datetime.datetime
    updated_at: datetime.datetime


class DocumentCreate(SQLModel):
    """Document creation model."""

    filename: str
    file_size: int | None = None
    mime_type: str | None = None


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
    # Optional link to the source document from ingestion pipeline
    source_document_id: uuid.UUID | None = Field(
        default=None,
        sa_column=Column(
            UUID(as_uuid=True),
            ForeignKey("documents.id", ondelete="SET NULL"),
            nullable=True,
            index=True,
        ),
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
    source_document_id: uuid.UUID | None
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
    source_document_id: uuid.UUID | None = None


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
