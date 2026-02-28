"""Database models using SQLModel.

Defines the core data models for the Template Intelligence Engine
and the multi-tier memory system for the Intelligence Engine.
"""

import datetime
import enum
import uuid
from typing import Any

from sqlalchemy import Column, DateTime, Enum as SQLEnum, ForeignKey, Index, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlmodel import Field, SQLModel


class TemplateStatus(str, enum.Enum):
    """Status of template processing pipeline."""

    QUEUED = "queued"
    ANALYZING = "analyzing"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"


class MemoryType(str, enum.Enum):
    """Type of procedural memory."""

    PREFERENCE = "preference"
    HABIT = "habit"
    CORRECTION = "correction"


class EventType(str, enum.Enum):
    """Type of episodic event."""

    FEEDBACK_GIVEN = "feedback_given"
    REPORT_GENERATED = "report_generated"
    ERROR_OCCURRED = "error_occurred"
    TEMPLATE_ANALYZED = "template_analyzed"
    VARIABLE_CORRECTED = "variable_corrected"


# =============================================================================
# Core Models
# =============================================================================


class Organization(SQLModel, table=True):
    """Organization model for multi-tenancy."""

    __tablename__ = "organizations"
    __table_args__ = {"extend_existing": True}

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        sa_column=Column(UUID(as_uuid=True), primary_key=True),
    )
    name: str = Field(max_length=255)
    slug: str = Field(max_length=100, unique=True, index=True)
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


class TemplateStorage(SQLModel, table=True):
    """Template storage model for analyzed Word templates."""

    __tablename__ = "templates"
    __table_args__ = {"extend_existing": True}

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
    name: str = Field(max_length=255)
    original_filename: str = Field(max_length=512)
    file_path: str = Field(max_length=1024)
    description: str | None = Field(default=None, max_length=1024)
    detected_variables: dict[str, Any] | None = Field(
        default=None,
        sa_column=Column(JSONB),
    )
    paragraph_count: int | None = Field(default=None, ge=0)
    analysis_method: str | None = Field(default="regex", max_length=50)
    is_tagged: bool = Field(default=False)

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

    batch_id: str | None = Field(default=None, max_length=255, index=True)
    previous_batch_id: str | None = Field(default=None, max_length=255, index=True)
    batch_status: str | None = Field(default=None, max_length=50)

    injection_status: str | None = Field(default=None, max_length=50)
    injection_task_id: str | None = Field(default=None, max_length=255, index=True)
    injection_started_at: datetime.datetime | None = Field(
        default=None, sa_column=Column(DateTime(timezone=True))
    )
    injection_completed_at: datetime.datetime | None = Field(
        default=None, sa_column=Column(DateTime(timezone=True))
    )
    injection_error_message: str | None = Field(default=None, max_length=2048)

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
# Procedural Memory Models
# =============================================================================


class ProceduralMemory(SQLModel, table=True):
    """Procedural memory stores learned adviser patterns and habits.

    This is the "how" layer - storing rules and patterns that the adviser
    follows consistently, learned from their behavior over time.
    """

    __tablename__ = "procedural_memory"
    __table_args__ = (
        Index("idx_procedural_adviser_memory_type", "adviser_id", "memory_type"),
        Index("idx_procedural_adviser_org", "adviser_id", "org_id"),
        {"extend_existing": True},
    )

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        sa_column=Column(UUID(as_uuid=True), primary_key=True),
    )
    adviser_id: str = Field(max_length=100, index=True)
    org_id: uuid.UUID = Field(
        sa_column=Column(
            UUID(as_uuid=True),
            ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        )
    )
    memory_type: MemoryType = Field(
        sa_column=Column(SQLEnum(MemoryType), index=True)
    )
    key: str = Field(
        max_length=255,
        description="The memory key (e.g., 'client_name_format', 'risk_level_preference')",
    )
    value: dict[str, Any] = Field(
        sa_column=Column(JSONB),
        description="The memory value as JSONB for flexibility",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for this memory (0-1)",
    )
    access_count: int = Field(
        default=0,
        ge=0,
        description="Number of times this memory was accessed",
    )
    last_accessed_at: datetime.datetime | None = Field(
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


class VariableCorrection(SQLModel, table=True):
    """Tracks corrections made to variable values.

    Learns patterns in how the adviser corrects incorrect variable mappings,
    enabling better auto-injection in the future.
    """

    __tablename__ = "variable_corrections"
    __table_args__ = (
        Index("idx_correction_adviser_var", "adviser_id", "variable_name"),
    )

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        sa_column=Column(UUID(as_uuid=True), primary_key=True),
    )
    adviser_id: str = Field(max_length=100, index=True)
    org_id: uuid.UUID = Field(
        sa_column=Column(
            UUID(as_uuid=True),
            ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        )
    )
    template_id: uuid.UUID | None = Field(
        sa_column=Column(UUID(as_uuid=True), ForeignKey("templates.id"))
    )
    variable_name: str = Field(max_length=255, index=True)
    incorrect_value: str = Field(max_length=512)
    corrected_value: str = Field(max_length=512)
    correction_count: int = Field(
        default=1,
        ge=1,
        description="Number of times this correction was made",
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


# =============================================================================
# Episodic Memory Models
# =============================================================================


class EpisodicEvent(SQLModel, table=True):
    """Episodic memory stores discrete events for temporal context.

    This is the "when" layer - storing specific events that occurred
    at particular times, enabling temporal pattern recognition.
    """

    __tablename__ = "episodic_events"
    __table_args__ = (
        Index("idx_episodic_adviser_session", "adviser_id", "session_id"),
        Index("idx_episodic_adviser_event_type", "adviser_id", "event_type"),
        Index("idx_episodic_timestamp", "timestamp"),
    )

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        sa_column=Column(UUID(as_uuid=True), primary_key=True),
    )
    adviser_id: str = Field(max_length=100, index=True)
    org_id: uuid.UUID = Field(
        sa_column=Column(
            UUID(as_uuid=True),
            ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        )
    )
    session_id: uuid.UUID | None = Field(
        default=None,
        sa_column=Column(UUID(as_uuid=True), index=True),
        description="Group related events (e.g., a single report generation session)",
    )
    event_type: EventType = Field(
        sa_column=Column(SQLEnum(EventType), index=True)
    )
    context: dict[str, Any] = Field(
        sa_column=Column(JSONB),
        description="Full event context as JSONB",
    )
    embedding: list[float] | None = Field(
        default=None,
        sa_column=Column(JSONB),
        description="Optional embedding for semantic retrieval",
    )
    timestamp: datetime.datetime = Field(
        default_factory=datetime.datetime.utcnow,
        sa_column=Column(DateTime(timezone=True), server_default=text("NOW()"), index=True),
    )


class FeedbackEvent(SQLModel, table=True):
    """Stores detailed feedback events for learning.

    Captures the full feedback loop: original text, feedback provided,
    and the final chosen text.
    """

    __tablename__ = "feedback_events"
    __table_args__ = (
        Index("idx_feedback_adviser", "adviser_id"),
        Index("idx_feedback_template", "template_id"),
    )

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        sa_column=Column(UUID(as_uuid=True), primary_key=True),
    )
    adviser_id: str = Field(max_length=100, index=True)
    org_id: uuid.UUID = Field(
        sa_column=Column(
            UUID(as_uuid=True),
            ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        )
    )
    episodic_event_id: uuid.UUID | None = Field(
        sa_column=Column(UUID(as_uuid=True), ForeignKey("episodic_events.id"))
    )
    original_text: str = Field(
        max_length=10000,
        description="Original AI-generated text before editing",
    )
    feedback_text: str = Field(
        max_length=2000,
        description="User's natural-language feedback",
    )
    chosen_text: str = Field(
        max_length=10000,
        description="Approved/edited final text",
    )
    report_section: str | None = Field(
        default=None,
        max_length=255,
        description="Section of the report this feedback applies to",
    )
    report_type: str | None = Field(
        default=None,
        max_length=255,
        description="Type of report (e.g., investment_summary, risk_assessment)",
    )
    template_id: uuid.UUID | None = Field(
        default=None,
        sa_column=Column(UUID(as_uuid=True), ForeignKey("templates.id")),
    )
    created_at: datetime.datetime = Field(
        default_factory=datetime.datetime.utcnow,
        sa_column=Column(DateTime(timezone=True), server_default=text("NOW()")),
    )


# =============================================================================
# Analysis Prompt Models
# =============================================================================


class AnalysisPrompt(SQLModel, table=True):
    """Analysis prompt model for storing custom prompts."""

    __tablename__ = "analysis_prompts"
    __table_args__ = {"extend_existing": True}

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
    name: str = Field(max_length=255)
    prompt_text: str = Field(max_length=8192)
    is_default: bool = Field(default=False)
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
# Report Generation Request/Response Models
# =============================================================================


class GenerateReportRequest(SQLModel):
    """Request to generate a report using the Intelligence Engine."""

    adviser_id: str = Field(min_length=1, max_length=100)
    client_id: str = Field(min_length=1, max_length=100)
    topic: str = Field(min_length=1, max_length=1000)
    template_text: str = Field(min_length=1, description="The template to fill in")
    org_id: uuid.UUID | None = None


class GenerateReportResponse(SQLModel):
    """Response from report generation."""

    report_id: uuid.UUID
    generated_text: str
    extracted_variables: dict[str, Any]
    sources: dict[str, list[str]]
    session_id: uuid.UUID
    timestamp: datetime.datetime


class MemoryContext(SQLModel):
    """Context retrieved from multi-tier memory."""

    client_facts: dict[str, Any] = Field(default_factory=dict)
    procedural_rules: list[dict[str, Any]] = Field(default_factory=list)
    semantic_rules: list[dict[str, Any]] = Field(default_factory=list)
    recent_events: list[dict[str, Any]] = Field(default_factory=list)


# =============================================================================
# Draft Versioning Models
# =============================================================================


class DraftVersion(SQLModel, table=True):
    """Stores versions of drafted reports for a template."""

    __tablename__ = "draft_versions"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        sa_column=Column(UUID(as_uuid=True), primary_key=True),
    )
    template_id: uuid.UUID = Field(
        sa_column=Column(UUID(as_uuid=True), ForeignKey("templates.id"))
    )
    org_id: uuid.UUID = Field(
        sa_column=Column(UUID(as_uuid=True), ForeignKey("organizations.id"))
    )
    version_number: int = Field(
        default=1,
        description="Sequential version number for this specific template",
    )
    adviser_id: str = Field(max_length=100)
    generated_text: str = Field(
        description="The full drafted report text generated by the LLM"
    )
    extracted_variables: dict[str, Any] | None = Field(
        default=None,
        sa_column=Column(JSONB),
        description="Variables embedded in this draft",
    )
    feedback_used: str | None = Field(
        default=None,
        description="Any subjective stylistic feedback that prompted this version",
    )
    pdf_path: str | None = Field(
        default=None,
        max_length=1024,
        description="Path to the PDF generation snapshot",
    )
    created_at: datetime.datetime = Field(
        default_factory=datetime.datetime.utcnow,
        sa_column=Column(DateTime(timezone=True), server_default=text("NOW()")),
    )

