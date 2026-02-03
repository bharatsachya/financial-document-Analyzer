"""Template management API routes.

Handles template analysis, Jinja2 tag injection, and template download.
"""

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db, get_org_id
from app.api.schemas import TemplateAnalysisResponse
from app.core.config import Settings, get_settings
from app.strategies.template_engine import DetectedVariable
from app.strategies.template_engine.analyzer import TemplateAnalyzer
from app.strategies.template_engine.injector import TemplateInjector

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/templates", tags=["templates"])


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
        analyzer = TemplateAnalyzer(use_llm=settings.use_llm_for_templates)
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
    template_id: uuid.UUID = Body(..., description="The template ID from the analyze response"),
    variables: list[DetectedVariable] = Body(..., description="Reviewed list of variables to inject"),
    original_filename: str = Body(..., description="Original filename to locate the temp file"),
    org_id: uuid.UUID = Depends(get_org_id),
    session: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> Any:
    """Finalize template by injecting Jinja2 tags.

    Accepts the reviewed/edited variable list from the analyze endpoint,
    performs the tag injection, and saves the "Gold Master" template.

    Args:
        template_id: The template ID from the analyze response.
        variables: Reviewed list of variables to inject.
        original_filename: Original filename to locate the temp file.
        org_id: Organization ID from X-Org-ID header.
        session: Database session.
        settings: Application settings.

    Returns:
        Dict with template status and download URL.

    Raises:
        HTTPException: If template file not found or injection fails.
    """
    try:
        logger.info(f"Finalizing template: {template_id}")

        # Locate the temporary file
        temp_dir = Path(settings.upload_dir) / "templates" / str(org_id)
        temp_file_path = temp_dir / f"{template_id}_{original_filename}"

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
    org_id: uuid.UUID = Depends(get_org_id),
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    """Download the finalized template.

    Args:
        template_id: The template ID to download.
        org_id: Organization ID from X-Org-ID header.
        settings: Application settings.

    Returns:
        FileResponse with the tagged template.

    Raises:
        HTTPException: If template not found.
    """
    try:
        template_path = (
            Path(settings.upload_dir)
            / "templates"
            / str(org_id)
            / f"{template_id}_tagged.docx"
        )

        if not template_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Template not found",
            )

        return FileResponse(
            path=str(template_path),
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            filename=template_path.name,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Template download failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Download failed",
        ) from e
