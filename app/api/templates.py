"""Template management API routes.

Handles template analysis, Jinja2 tag injection, template storage, and download.
"""

import logging
import os
import random
import string
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from celery import group
from fastapi import APIRouter, Body, Depends, Header, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db, get_org_id
from app.api.schemas import (
    BatchStatusResponse,
    BatchUploadResponse,
    InjectFinalizeRequest,
    InjectFinalizeResponse,
    RandomInjectRequest,
    TemplateAnalysisResponse,
    TemplateStatusResponse,
)
from app.core.config import Settings, get_settings
from app.db.models import TemplateStatus, TemplateListResponse, TemplateRead, TemplateStorage
from app.strategies.template_engine import DetectedVariable
from app.strategies.template_engine.analyzer import TemplateAnalyzer
from app.strategies.template_engine.injector import TemplateInjector

logger = logging.getLogger(__name__)


# =============================================================================
# Helper Functions
# =============================================================================


def _generate_batch_id() -> str:
    """Generate a unique batch ID."""
    return f"batch_{uuid.uuid4().hex[:12]}"


router = APIRouter(prefix="/templates", tags=["templates"])


# =============================================================================
# Request/Response Models
# =============================================================================


class FinalizeTemplateRequest(BaseModel):
    """Request model for template finalization."""

    template_id: str = Field(
        ...,
        description="The template ID from the analyze response (must be a valid UUID)",
    )
    variables: list[DetectedVariable] = Field(
        ..., description="Reviewed list of variables to inject"
    )
    original_filename: str = Field(..., description="Original filename to locate the temp file")

    @field_validator("template_id")
    @classmethod
    def validate_template_id(cls, v: str) -> str:
        """Validate that template_id is a valid UUID."""
        try:
            uuid.UUID(v)
        except ValueError as exc:
            raise ValueError(
                f"Invalid template_id '{v}'. Must be a valid UUID format "
                "(e.g., '550e8400-e29b-41d4-a716-446655440000')"
            ) from exc
        return v

    def get_template_id_uuid(self) -> uuid.UUID:
        """Return template_id as a UUID object."""
        return uuid.UUID(self.template_id)


# =============================================================================
# Endpoints
# =============================================================================


@router.post(
    "/analyze",
    response_model=TemplateAnalysisResponse,
    status_code=status.HTTP_200_OK,
)
async def analyze_template(
    file: UploadFile,
    org_id: uuid.UUID = Depends(get_org_id),
    settings: Settings = Depends(get_settings),
) -> TemplateAnalysisResponse:
    """Analyze a Word template to detect dynamic injection points.

    This endpoint processes the uploaded document and identifies text segments
    that should be replaced with Jinja2 variables (e.g., {{ client_name }}).

    Returns a list of detected variables for human review before finalizing.

    Args:
        file: The Word document (.docx) to analyze.
        org_id: Organization ID from X-Org-ID header.
        settings: Application settings.

    Returns:
        TemplateAnalysisResponse with detected variables.

    Raises:
        HTTPException: If file type is invalid or analysis fails.
    """
    try:
        logger.info(
            f"Starting template analysis for org: {org_id}, file: {file.filename}"
        )

        # Validate file type
        if not file.filename or not file.filename.endswith(".docx"):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Only .docx files are supported",
            )

        # Save uploaded file temporarily
        template_id = uuid.uuid4()
        temp_dir = Path(settings.upload_dir) / "templates" / str(org_id)
        temp_dir.mkdir(parents=True, exist_ok=True)

        temp_file_path = temp_dir / f"{template_id}_{file.filename}"
        content = await file.read()
        temp_file_path.write_bytes(content)

        logger.info(f"Saved temporary file: {temp_file_path}")

        # Run analyzer
        analyzer = TemplateAnalyzer(
            use_llm=settings.use_llm_for_templates,
            openai_api_key=settings.openrouter_api_key,
            base_url=settings.openai_base_url,
            model=settings.llm_chat_model,
        )
        detected_variables = await analyzer.analyze(str(temp_file_path))

        # Count paragraphs in document
        try:
            from docx import Document

            doc = Document(str(temp_file_path))
            total_paragraphs = len(doc.paragraphs)
        except ImportError:
            total_paragraphs = 0

        response = TemplateAnalysisResponse(
            template_id=template_id,
            filename=file.filename,
            detected_variables=detected_variables,
            total_paragraphs=total_paragraphs,
            analyzed_at=datetime.utcnow(),
        )

        logger.info(f"Analysis complete: {len(detected_variables)} variables detected")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Template analysis failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Template analysis failed: {str(e)}",
        ) from e


@router.post("/finalize", status_code=status.HTTP_201_CREATED)
async def finalize_template(
    request: FinalizeTemplateRequest,
    org_id: uuid.UUID = Depends(get_org_id),
    session: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> Any:
    """Finalize template by injecting Jinja2 tags.

    Accepts the reviewed/edited variable list from the analyze endpoint,
    performs the tag injection, and saves the "Gold Master" template.

    Args:
        request: Request containing template_id, variables, and original_filename.
        org_id: Organization ID from X-Org-ID header.
        session: Database session.
        settings: Application settings.

    Returns:
        Dict with template status and download URL.

    Raises:
        HTTPException: If template file not found or injection fails.
    """
    try:
        template_id = request.get_template_id_uuid()
        logger.info(f"Finalizing template: {template_id}")

        # Locate the temporary file
        temp_dir = Path(settings.upload_dir) / "templates" / str(org_id)
        temp_file_path = temp_dir / f"{template_id}_{request.original_filename}"

        if not temp_file_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Template file not found. Please re-run analysis.",
            )

        # Run injector
        injector = TemplateInjector()
        output_path = await injector.inject_tags(
            file_path=str(temp_file_path),
            variables=variables,
        )

        # Generate final filename
        final_path = (
            Path(settings.upload_dir) / "templates" / str(org_id) / f"{template_id}_tagged.docx"
        )
        Path(output_path).rename(final_path)

        logger.info(f"Template finalized: {final_path}")

        return {
            "template_id": str(template_id),
            "status": "finalized",
            "download_url": f"/templates/download/{template_id}",
            "variable_count": len(variables),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Template finalization failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Template finalization failed: {str(e)}",
        ) from e


@router.get("/download/{template_id}")
async def download_template(
    template_id: uuid.UUID,
    custom_filename: str | None = Query(default=None, description="Custom filename for download"),
    org_id: uuid.UUID | None = Query(default=None, description="Organization ID (for browser downloads)"),
    x_org_id: uuid.UUID | None = Header(default=None, alias="X-Org-ID", description="Organization ID from header (for API clients)"),
    settings: Settings = Depends(get_settings),
    session: AsyncSession = Depends(get_db),
) -> FileResponse:
    """Download the finalized template.

    Supports both header-based auth (API clients) and query param (browser downloads).
    For browser downloads: /templates/download/{id}?org_id={org_id}
    For API clients: Use X-Org-ID header

    Args:
        template_id: The template ID to download.
        custom_filename: Optional custom filename for download.
        org_id: Organization ID from query parameter (for browser downloads).
        x_org_id: Organization ID from X-Org-ID header (for API clients).
        settings: Application settings.
        session: Database session.

    Returns:
        FileResponse with the filled template.

    Raises:
        HTTPException: If template not found or org_id missing.
    """
    try:
        # Determine org_id from either query param or header (query param takes precedence)
        resolved_org_id = org_id or x_org_id
        if not resolved_org_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Organization ID required. Provide X-Org-ID header or org_id query parameter.",
            )

        # Query template with org_id filtering for tenant isolation
        from sqlalchemy import select

        query = select(TemplateStorage).where(
            TemplateStorage.id == template_id,
            TemplateStorage.org_id == resolved_org_id
        )
        result = await session.execute(query)
        template = result.scalar_one_or_none()

        if not template:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Template not found",
            )

        # Check if download is ready
        if not template.download_ready:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Template is not ready for download",
            )

        # Use the file_path from database record
        template_path = Path(template.file_path)

        if not template_path.exists():
            logger.error(
                f"Template file not found: {template_path} "
                f"(template_id: {template_id}, org_id: {resolved_org_id})"
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Template file not found. The file may have been removed or not yet generated.",
            )

        # Use custom filename if provided, otherwise generate default
        if custom_filename:
            # Ensure it has the right extension
            if not custom_filename.endswith('.docx'):
                download_filename = f"{custom_filename}.docx"
            else:
                download_filename = custom_filename
        else:
            # Generate user-friendly filename (original_filename is a string)
            orig_path = Path(template.original_filename)
            download_filename = f"{orig_path.stem}_filled{orig_path.suffix}"

        return FileResponse(
            path=str(template_path),
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            filename=download_filename,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Template download failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Download failed",
        ) from e


# =============================================================================
# Template Storage Endpoints
# =============================================================================


@router.post(
    "/save",
    response_model=TemplateRead,
    status_code=status.HTTP_201_CREATED,
)
async def save_template(
    template_id: uuid.UUID = Body(..., description="The template ID from analysis"),
    variables: list[DetectedVariable] = Body(..., description="Detected variables to store"),
    original_filename: str = Body(..., description="Original filename"),
    name: str | None = Body(default=None, description="Custom name for the template"),
    description: str | None = Body(default=None, description="Template description"),
    paragraph_count: int | None = Body(default=None, description="Total paragraph count"),
    source_document_id: uuid.UUID | None = Body(default=None, description="Source document ID"),
    org_id: uuid.UUID = Depends(get_org_id),
    session: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> TemplateStorage:
    """Save an analyzed template to storage for later use.

    This endpoint persists the analyzed template metadata and detected variables
    to the database, enabling quick retrieval for the inject phase.

    Args:
        template_id: The template ID from the analyze response.
        variables: List of detected variables.
        original_filename: Original filename.
        name: Custom name for the template (defaults to filename).
        description: Optional template description.
        paragraph_count: Total paragraphs in document.
        source_document_id: Optional source document from ingestion.
        org_id: Organization ID from X-Org-ID header.
        session: Database session.
        settings: Application settings.

    Returns:
        TemplateRead with saved template metadata.

    Raises:
        HTTPException: If template file not found or save fails.
    """
    try:
        logger.info(f"Saving template: {template_id}")

        # Locate the temporary file
        temp_dir = Path(settings.upload_dir) / "templates" / str(org_id)
        temp_file_path = temp_dir / f"{template_id}_{original_filename}"

        if not temp_file_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Template file not found. Please re-run analysis.",
            )

        # Create permanent storage location
        storage_dir = Path(settings.upload_dir) / "stored_templates" / str(org_id)
        storage_dir.mkdir(parents=True, exist_ok=True)

        stored_filename = f"{template_id}_{original_filename}"
        stored_path = storage_dir / stored_filename

        # Copy file to permanent storage
        import shutil
        shutil.copy(str(temp_file_path), str(stored_path))

        # Convert detected variables to dict format for JSONB storage
        variables_dict = {
            "variables": [
                {
                    "original_text": v.original_text,
                    "suggested_variable_name": v.suggested_variable_name,
                    "rationale": v.rationale,
                    "paragraph_index": v.paragraph_index,
                }
                for v in variables
            ],
            "count": len(variables),
        }

        # Create template record
        template = TemplateStorage(
            id=template_id,
            org_id=org_id,
            name=name or original_filename,
            original_filename=original_filename,
            file_path=str(stored_path),
            description=description,
            source_document_id=source_document_id,
            detected_variables=variables_dict,
            paragraph_count=paragraph_count,
            analysis_method="llm" if settings.use_llm_for_templates else "regex",
            is_tagged=False,  # Not tagged yet - user can finalize later
        )

        session.add(template)
        await session.commit()
        await session.refresh(template)

        logger.info(f"Template saved: {template_id}")
        return template

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Template save failed: {e}", exc_info=True)
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Template save failed: {str(e)}",
        ) from e


@router.get(
    "/stored",
    response_model=TemplateListResponse,
    status_code=status.HTTP_200_OK,
)
async def list_stored_templates(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    org_id: uuid.UUID = Depends(get_org_id),
    session: AsyncSession = Depends(get_db),
) -> TemplateListResponse:
    """List all stored templates for the organization.

    Args:
        page: Page number (1-indexed).
        page_size: Number of items per page.
        org_id: Organization ID from X-Org-ID header.
        session: Database session.

    Returns:
        TemplateListResponse with list of templates.

    Raises:
        HTTPException: If query fails.
    """
    try:
        # Query templates for this organization
        statement = (
            select(TemplateStorage)
            .where(TemplateStorage.org_id == org_id)
            .where(TemplateStorage.status == TemplateStatus.COMPLETED)
            .order_by(TemplateStorage.created_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )

        result = await session.execute(statement)
        templates = result.scalars().all()

        # Get total count
        count_statement = (
            select(TemplateStorage)
            .where(TemplateStorage.org_id == org_id)
            .where(TemplateStorage.status == TemplateStatus.COMPLETED)
        )
        count_result = await session.execute(count_statement)
        total = len(count_result.scalars().all())

        logger.info(f"Retrieved {len(templates)} templates for org: {org_id}")

        return TemplateListResponse(
            templates=templates,
            total=total,
        )

    except Exception as e:
        logger.error(f"Template list failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Template list failed: {str(e)}",
        ) from e


@router.get(
    "/stored/{template_id}",
    response_model=TemplateRead,
    status_code=status.HTTP_200_OK,
)
async def get_stored_template(
    template_id: uuid.UUID,
    org_id: uuid.UUID = Depends(get_org_id),
    session: AsyncSession = Depends(get_db),
) -> TemplateStorage:
    """Retrieve a specific stored template.

    Args:
        template_id: The template ID to retrieve.
        org_id: Organization ID from X-Org-ID header.
        session: Database session.

    Returns:
        TemplateRead with template details.

    Raises:
        HTTPException: If template not found.
    """
    try:
        statement = (
            select(TemplateStorage)
            .where(TemplateStorage.id == template_id)
            .where(TemplateStorage.org_id == org_id)
        )

        result = await session.execute(statement)
        template = result.scalar_one_or_none()

        if not template:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Template not found",
            )

        logger.info(f"Retrieved template: {template_id}")
        return template

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Template retrieval failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Template retrieval failed: {str(e)}",
        ) from e


# =============================================================================
# Batch Processing Endpoints
# =============================================================================


@router.post(
    "/analyze-batch",
    response_model=BatchUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def analyze_template_batch(
    files: list[UploadFile],
    org_id: uuid.UUID = Depends(get_org_id),
    session: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> BatchUploadResponse:
    """Upload multiple templates for batch analysis.

    This endpoint accepts multiple Word documents and queues them for
    asynchronous analysis. Returns a batch ID for tracking progress.

    Args:
        files: List of Word documents (.docx) to analyze.
        org_id: Organization ID from X-Org-ID header.
        session: Database session.
        settings: Application settings.

    Returns:
        BatchUploadResponse with batch_id and template count.
    """
    try:
        logger.info(f"Batch upload for org: {org_id}, files: {len(files)}")

        # Validate files
        for file in files:
            if not file.filename or not file.filename.endswith(".docx"):
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail="Only .docx files are supported",
                )

        # Generate batch ID
        batch_id = _generate_batch_id()
        template_ids = []

        # Save files and create database records
        temp_dir = Path(settings.upload_dir) / "templates" / str(org_id)
        temp_dir.mkdir(parents=True, exist_ok=True)

        for file in files:
            template_id = uuid.uuid4()
            temp_file_path = temp_dir / f"{template_id}_{file.filename}"
            content = await file.read()
            temp_file_path.write_bytes(content)

            # Create template record
            template = TemplateStorage(
                id=template_id,
                org_id=org_id,
                name=file.filename,
                original_filename=file.filename,
                file_path=str(temp_file_path),
                status=TemplateStatus.QUEUED,
                batch_id=batch_id,
                batch_status="queued",
            )
            session.add(template)
            template_ids.append(str(template_id))

        await session.commit()

        # Queue Celery tasks for each template
        from app.worker import celery_app, analyze_template_task

        # Use task signatures instead of send_task to avoid type errors
        job = group(
            analyze_template_task.s(str(tid))
            for tid in template_ids
        )
        job.apply_async()

        logger.info(f"Batch {batch_id} created with {len(template_ids)} templates")

        return BatchUploadResponse(
            batch_id=batch_id,
            template_count=len(template_ids),
            message=f"Uploaded {len(template_ids)} templates for processing",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch upload failed: {e}", exc_info=True)
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch upload failed: {str(e)}",
        ) from e


@router.post(
    "/queue-next-batch",
    response_model=BatchUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def queue_next_batch(
    files: list[UploadFile],
    current_batch_id: str = Query(..., description="Current batch ID"),
    org_id: uuid.UUID = Depends(get_org_id),
    session: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> BatchUploadResponse:
    """Queue files for next batch (auto-starts when current completes).

    This endpoint creates a new batch with status=QUEUED and links it to the
    current batch. When the current batch completes, the next batch will
    automatically start processing.

    Args:
        files: List of Word documents (.docx) to analyze.
        current_batch_id: The currently processing batch ID.
        org_id: Organization ID from X-Org-ID header.
        session: Database session.
        settings: Application settings.

    Returns:
        BatchUploadResponse with next_batch_id.
    """
    try:
        logger.info(f"Queue next batch after: {current_batch_id}, files: {len(files)}")

        # Validate current batch exists
        batch_query = select(TemplateStorage).where(
            TemplateStorage.batch_id == current_batch_id,
            TemplateStorage.org_id == org_id,
        )
        batch_result = await session.execute(batch_query)
        current_templates = batch_result.scalars().all()

        if not current_templates:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Current batch {current_batch_id} not found",
            )

        # Validate files
        for file in files:
            if not file.filename or not file.filename.endswith(".docx"):
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail="Only .docx files are supported",
                )

        # Generate next batch ID
        next_batch_id = _generate_batch_id()
        template_ids = []

        # Save files and create database records
        temp_dir = Path(settings.upload_dir) / "templates" / str(org_id)
        temp_dir.mkdir(parents=True, exist_ok=True)

        for file in files:
            template_id = uuid.uuid4()
            temp_file_path = temp_dir / f"{template_id}_{file.filename}"
            content = await file.read()
            temp_file_path.write_bytes(content)

            # Create template record with previous_batch_id
            template = TemplateStorage(
                id=template_id,
                org_id=org_id,
                name=file.filename,
                original_filename=file.filename,
                file_path=str(temp_file_path),
                status=TemplateStatus.QUEUED,
                batch_id=next_batch_id,
                previous_batch_id=current_batch_id,
                batch_status="queued",
            )
            session.add(template)
            template_ids.append(str(template_id))

        await session.commit()

        logger.info(f"Next batch {next_batch_id} created with {len(template_ids)} templates")

        return BatchUploadResponse(
            batch_id=next_batch_id,
            template_count=len(template_ids),
            message=f"Queued {len(template_ids)} templates for next batch",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Queue next batch failed: {e}", exc_info=True)
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Queue next batch failed: {str(e)}",
        ) from e


@router.get(
    "/status/{template_id}",
    response_model=TemplateStatusResponse,
    status_code=status.HTTP_200_OK,
)
async def get_template_status(
    template_id: uuid.UUID,
    org_id: uuid.UUID = Depends(get_org_id),
    session: AsyncSession = Depends(get_db),
) -> TemplateStatusResponse:
    """Get processing status for a specific template.

    Args:
        template_id: The template ID to check.
        org_id: Organization ID from X-Org-ID header.
        session: Database session.

    Returns:
        TemplateStatusResponse with current status and progress.
    """
    try:
        query = select(TemplateStorage).where(
            TemplateStorage.id == template_id,
            TemplateStorage.org_id == org_id,
        )

        result = await session.execute(query)
        template = result.scalar_one_or_none()

        if not template:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Template not found",
            )

        # Calculate progress percentage
        progress = None
        if template.status == TemplateStatus.COMPLETED:
            progress = 100
        elif template.status in [TemplateStatus.ANALYZING, TemplateStatus.FINALIZING]:
            progress = 50  # Simplified progress
        elif template.status == TemplateStatus.QUEUED:
            progress = 0

        return TemplateStatusResponse(
            template_id=template.id,
            filename=template.original_filename,
            status=template.status.value,
            progress=progress,
            detected_variables=template.detected_variables,
            download_ready=template.download_ready,
            download_url=template.download_url,
            error_message=template.error_message,
            created_at=template.created_at,
            processing_started_at=template.processing_started_at,
            processing_completed_at=template.processing_completed_at,
            batch_id=template.batch_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Template status check failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Status check failed: {str(e)}",
        ) from e


@router.get(
    "/batch-status/{batch_id}",
    response_model=BatchStatusResponse,
    status_code=status.HTTP_200_OK,
)
async def get_batch_status(
    batch_id: str,
    org_id: uuid.UUID = Depends(get_org_id),
    session: AsyncSession = Depends(get_db),
) -> BatchStatusResponse:
    """Get status for all templates in a batch.

    Args:
        batch_id: The batch ID to check.
        org_id: Organization ID from X-Org-ID header.
        session: Database session.

    Returns:
        BatchStatusResponse with aggregate and individual status.
    """
    try:
        # Get all templates in batch
        query = select(TemplateStorage).where(
            TemplateStorage.batch_id == batch_id,
            TemplateStorage.org_id == org_id,
        )

        result = await session.execute(query)
        templates = result.scalars().all()

        if not templates:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Batch not found",
            )

        # Calculate aggregate stats
        total = len(templates)
        # Completed means status is COMPLETED (analysis done), regardless of download_ready
        completed = sum(1 for t in templates if t.status == TemplateStatus.COMPLETED)
        failed = sum(1 for t in templates if t.status == TemplateStatus.FAILED)
        in_progress = sum(
            1 for t in templates
            if t.status in [TemplateStatus.ANALYZING, TemplateStatus.FINALIZING]
        )

        # Get batch status
        batch_status = templates[0].batch_status

        # Check for next batch
        next_batch_id = None
        next_batch_query = select(TemplateStorage.batch_id).where(
            TemplateStorage.previous_batch_id == batch_id,
            TemplateStorage.batch_status == "queued",
        ).distinct().limit(1)
        next_batch_result = await session.execute(next_batch_query)
        next_batch_id = next_batch_result.scalar_one_or_none()

        # Count queued files in next batch
        queued = 0
        if next_batch_id:
            queued_query = select(func.count(TemplateStorage.id)).where(
                TemplateStorage.batch_id == next_batch_id,
            )
            queued_result = await session.execute(queued_query)
            queued = queued_result.scalar_one()

        # Build template status responses
        template_statuses = []
        for t in templates:
            progress = None
            if t.status == TemplateStatus.COMPLETED:
                progress = 100
            elif t.status in [TemplateStatus.ANALYZING, TemplateStatus.FINALIZING]:
                progress = 50
            elif t.status == TemplateStatus.QUEUED:
                progress = 0

            template_statuses.append(
                TemplateStatusResponse(
                    template_id=t.id,
                    filename=t.original_filename,
                    status=t.status.value,
                    progress=progress,
                    detected_variables=t.detected_variables,
                    download_ready=t.download_ready,
                    download_url=t.download_url,
                    error_message=t.error_message,
                    created_at=t.created_at,
                    processing_started_at=t.processing_started_at,
                    processing_completed_at=t.processing_completed_at,
                    batch_id=t.batch_id,
                )
            )

        return BatchStatusResponse(
            batch_id=batch_id,
            batch_status=batch_status,
            total_templates=total,
            completed=completed,
            failed=failed,
            in_progress=in_progress,
            queued=queued,
            next_batch_id=next_batch_id,
            previous_batch_id=templates[0].previous_batch_id,
            templates=template_statuses,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch status check failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch status check failed: {str(e)}",
        ) from e


@router.get(
    "/list-ready",
    response_model=dict[str, Any],
    status_code=status.HTTP_200_OK,
)
async def list_ready_templates(
    download_ready: bool | None = Query(None, description="Filter by download_ready status"),
    status_filter: str | None = Query(None, alias="status", description="Filter by status (e.g., 'completed')"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    org_id: uuid.UUID = Depends(get_org_id),
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """List templates ready for specific actions.

    Filters templates based on their processing status and readiness.
    Used by frontend to get templates ready for injection or download.

    Args:
        download_ready: Filter by download_ready status.
        status_filter: Filter by template status.
        page: Page number.
        page_size: Items per page.
        org_id: Organization ID from X-Org-ID header.
        session: Database session.

    Returns:
        Dict with templates list and total count.
    """
    try:
        query = select(TemplateStorage).where(TemplateStorage.org_id == org_id)

        # Apply filters
        if download_ready is not None:
            query = query.where(TemplateStorage.download_ready == download_ready)

        if status_filter:
            try:
                status_enum = TemplateStatus(status_filter)
                query = query.where(TemplateStorage.status == status_enum)
            except ValueError:
                pass  # Invalid status, ignore filter

        # Count total using the query's selectable (avoids cartesian product warning)
        from sqlalchemy import select as slct
        count_query = slct(func.count()).select_from(query)
        count_result = await session.execute(count_query)
        total = count_result.scalar_one()

        # Apply pagination
        query = query.order_by(TemplateStorage.created_at.desc())
        query = query.offset((page - 1) * page_size).limit(page_size)

        result = await session.execute(query)
        templates = result.scalars().all()

        return {
            "templates": templates,
            "total": total,
            "page": page,
            "page_size": page_size,
        }

    except Exception as e:
        logger.error(f"List ready templates failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"List templates failed: {str(e)}",
        ) from e


@router.post(
    "/inject-random/{template_id}",
    status_code=status.HTTP_410_GONE,
)
async def inject_random_values(
    template_id: uuid.UUID,
    org_id: uuid.UUID = Depends(get_org_id),
) -> dict[str, str]:
    """Inject random values into a template and queue finalization.

    **DEPRECATED**: This endpoint is deprecated and no longer available.
    Random value generation should be done in the frontend using the
    "🎲 Random Fill" button in the template injection UI.

    Use POST /templates/finalize-async with frontend-generated values instead.

    Args:
        template_id: The template to inject values into.
        org_id: Organization ID from X-Org-ID header.

    Returns:
        Error message with migration instructions.
    """
    return {
        "error": "This endpoint has been deprecated.",
        "message": "Random value generation is now handled in the frontend.",
        "instructions": "Use the '🎲 Random Fill' button in the template injection UI, then call POST /templates/finalize-async with the generated values.",
        "alternative_endpoint": f"/templates/finalize-async"
    }


@router.get("/injection-queue", response_model=dict[str, Any], status_code=status.HTTP_200_OK)
async def get_injection_queue(
    org_id: uuid.UUID = Depends(get_org_id),
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Get status of the injection queue.

    Returns:
        Dict with aggregate counts and list of active jobs.
    """
    try:
        # Get counts by status
        count_query = (
            select(TemplateStorage.injection_status, func.count())
            .where(
                TemplateStorage.org_id == org_id,
                TemplateStorage.injection_status.is_not(None),
            )
            .group_by(TemplateStorage.injection_status)
        )

        result = await session.execute(count_query)
        counts = dict(result.all())

        # Get recent jobs
        jobs_query = (
            select(TemplateStorage)
            .where(
                TemplateStorage.org_id == org_id,
                TemplateStorage.injection_status.is_not(None),
            )
            .order_by(TemplateStorage.injection_started_at.desc())
            .limit(20)
        )
        
        jobs_result = await session.execute(jobs_query)
        jobs = jobs_result.scalars().all()
        
        return {
            "queued": counts.get("queued", 0),
            "processing": counts.get("processing", 0),
            "completed": counts.get("completed", 0),
            "failed": counts.get("failed", 0),
            "jobs": [
                {
                    "template_id": job.id,
                    "filename": job.original_filename,
                    "status": job.injection_status,
                    "started_at": job.injection_started_at,
                    "completed_at": job.injection_completed_at,
                    "task_id": job.injection_task_id,
                }
                for job in jobs
            ],
        }
    except Exception as e:
        logger.error(f"Error fetching injection queue: {e}", exc_info=True)
        return {
            "queued": 0,
            "processing": 0,
            "completed": 0,
            "failed": 0,
            "jobs": [],
        }


@router.post(
    "/finalize-async",
    response_model=InjectFinalizeResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def finalize_template_async(
    request: InjectFinalizeRequest,
    org_id: uuid.UUID = Depends(get_org_id),
    session: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> InjectFinalizeResponse:
    """Queue template finalization with variable injection (async).

    This endpoint queues the template for async injection instead of
    processing it synchronously.

    Args:
        request: Request with template_id and variables.
        org_id: Organization ID from X-Org-ID header.
        session: Database session.
        settings: Application settings.

    Returns:
        InjectFinalizeResponse with task_id.
    """
    try:
        template_id = request.template_id
        logger.info(f"Queuing template finalization: {template_id}")

        # Get template to verify it exists and belongs to org
        query = select(TemplateStorage).where(
            TemplateStorage.id == template_id,
            TemplateStorage.org_id == org_id,
        )

        result = await session.execute(query)
        template = result.scalar_one_or_none()

        if not template:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Template not found",
            )

        # Queue Celery task for finalization
        from app.worker import celery_app

        task = celery_app.send_task(
            "app.worker.finalize_template",
            args=[str(template_id), request.variables],
        )

        # Update template status
        try:
            template.injection_status = "queued"
            template.injection_task_id = task.id
            template.injection_started_at = datetime.utcnow()
            template.status = TemplateStatus.FINALIZING  # Also update main status
            
            session.add(template)
            await session.commit()
        except Exception as db_err:
            logger.error(f"Failed to update template status for injection: {db_err}")
            # Continue properly even if DB update fails, knowing task is queued

        logger.info(f"Queued finalization for template {template_id}, task: {task.id}")

        return InjectFinalizeResponse(
            template_id=template_id,
            status="queued",
            task_id=task.id,
            message="Template finalization queued",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Template finalization queue failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Finalization queue failed: {str(e)}",
        ) from e


# =============================================================================
# Prompt Management Endpoints
# =============================================================================


@router.get("/prompts")
async def list_prompts(
    session: AsyncSession = Depends(get_db),
    org_id: uuid.UUID = Depends(get_org_id),
) -> dict[str, Any]:
    """List all analysis prompts for the organization.

    Args:
        session: Database session.
        org_id: Organization ID from dependency.

    Returns:
        Dict with prompts list and total count.
    """
    try:
        from app.db.models import AnalysisPrompt

        query = select(AnalysisPrompt).where(AnalysisPrompt.org_id == org_id)
        result = await session.execute(query)
        prompts = result.scalars().all()

        prompt_list = [
            {
                "id": str(p.id),
                "name": p.name,
                "prompt_text": p.prompt_text,
                "is_default": p.is_default,
                "created_at": p.created_at.isoformat() if p.created_at else None,
                "updated_at": p.updated_at.isoformat() if p.updated_at else None,
            }
            for p in prompts
        ]

        return {"prompts": prompt_list, "total": len(prompt_list)}

    except Exception as e:
        logger.error(f"Failed to list prompts: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list prompts: {str(e)}",
        ) from e


@router.get("/prompts/default")
async def get_default_prompt(
    session: AsyncSession = Depends(get_db),
    org_id: uuid.UUID = Depends(get_org_id),
) -> dict[str, Any]:
    """Get the default analysis prompt for the organization.

    Args:
        session: Database session.
        org_id: Organization ID from dependency.

    Returns:
        Dict with prompt info or system default placeholder.
    """
    try:
        from app.db.models import AnalysisPrompt

        # Look for a default prompt for this org
        query = select(AnalysisPrompt).where(
            AnalysisPrompt.org_id == org_id,
            AnalysisPrompt.is_default == True,
        )
        result = await session.execute(query)
        prompt = result.scalar_one_or_none()

        if prompt:
            return {
                "id": str(prompt.id),
                "name": prompt.name,
                "prompt_text": prompt.prompt_text,
                "is_default": True,
                "source": "organization",
            }
        else:
            # Return system default info
            return {
                "id": None,
                "name": "System Default",
                "prompt_text": None,  # Frontend will show hardcoded default
                "is_default": True,
                "source": "system",
            }

    except Exception as e:
        logger.error(f"Failed to get default prompt: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get default prompt: {str(e)}",
        ) from e


@router.post("/prompts")
async def create_prompt(
    name: str = Body(..., embed=True, min_length=1, max_length=255),
    prompt_text: str = Body(..., embed=True, min_length=1, max_length=8192),
    set_as_default: bool = Body(False, embed=True),
    session: AsyncSession = Depends(get_db),
    org_id: uuid.UUID = Depends(get_org_id),
) -> dict[str, Any]:
    """Create a new analysis prompt.

    If set_as_default=True, this will become the default prompt for
    the organization and any previous default will be unset.

    Args:
        name: Prompt name.
        prompt_text: The prompt content.
        set_as_default: Whether to set as default.
        session: Database session.
        org_id: Organization ID from dependency.

    Returns:
        Created prompt dict.
    """
    try:
        from app.db.models import AnalysisPrompt
        from sqlalchemy import update

        # If setting as default, unset existing defaults first
        if set_as_default:
            await session.execute(
                update(AnalysisPrompt)
                .where(
                    AnalysisPrompt.org_id == org_id,
                    AnalysisPrompt.is_default == True,
                )
                .values(is_default=False)
            )

        # Create new prompt
        new_prompt = AnalysisPrompt(
            name=name,
            prompt_text=prompt_text,
            is_default=set_as_default,
            org_id=org_id,
        )
        session.add(new_prompt)

        # Single commit for both operations
        await session.commit()
        await session.refresh(new_prompt)

        logger.info(f"Created prompt '{name}' (id={new_prompt.id}) for org {org_id}, default={set_as_default}")

        return {
            "id": str(new_prompt.id),
            "name": new_prompt.name,
            "prompt_text": new_prompt.prompt_text,
            "is_default": new_prompt.is_default,
            "created_at": new_prompt.created_at.isoformat() if new_prompt.created_at else None,
        }

    except Exception as e:
        logger.error(f"Failed to create prompt: {e}", exc_info=True)
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create prompt: {str(e)}",
        ) from e


@router.delete("/prompts/{prompt_id}")
async def delete_prompt(
    prompt_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
    org_id: uuid.UUID = Depends(get_org_id),
) -> dict[str, Any]:
    """Delete an analysis prompt.

    Args:
        prompt_id: The prompt ID to delete.
        session: Database session.
        org_id: Organization ID from dependency.

    Returns:
        Success dict.
    """
    try:
        from app.db.models import AnalysisPrompt
        from sqlalchemy import delete

        # Ensure prompt belongs to this org
        query = select(AnalysisPrompt).where(
            AnalysisPrompt.id == prompt_id,
            AnalysisPrompt.org_id == org_id,
        )
        result = await session.execute(query)
        prompt = result.scalar_one_or_none()

        if not prompt:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Prompt not found",
            )

        # Delete the prompt
        await session.execute(
            delete(AnalysisPrompt).where(AnalysisPrompt.id == prompt_id)
        )
        await session.commit()

        logger.info(f"Deleted prompt {prompt_id} for org {org_id}")

        return {"success": True, "id": str(prompt_id)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete prompt: {e}", exc_info=True)
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete prompt: {str(e)}",
        ) from e
