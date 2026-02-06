"""Ingestion API routes.

Handles document upload and status tracking for the async processing pipeline.
"""

import logging
import os
import uuid
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError

from app.api.deps import get_db, get_org_id
from app.api.schemas import DocumentListResponse, DocumentStatusResponse, DocumentUploadResponse
from app.core.config import Settings, get_settings
from app.core.factory import ComponentFactory
from app.db.models import Document, DocumentStatus, User
from app.interfaces.parser import BaseParser

# Import Celery task
from app.worker import process_document_task

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingestion"])


# COMMENTED_OUT: Document ingestino is not our concern right now
# @router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_202_ACCEPTED)
# async def upload_document(
#     file: UploadFile = File(..., description="Document file to ingest"),
#     org_id: uuid.UUID = Depends(get_org_id),
#     session: AsyncSession = Depends(get_db),
#     settings: Settings = Depends(get_settings),
# ) -> DocumentUploadResponse:
#     """Upload a document for async processing.
#
#     This endpoint immediately returns after:
#     1. Saving the file to disk
#     2. Creating a database record with QUEUED status
#     3. Dispatching a Celery task for processing
#
#     The client can poll /status to track processing progress.
#
#     Args:
#         file: The uploaded file.
#         org_id: Organization ID from header.
#         session: Database session.
#         settings: Application settings.
#
#     Returns:
#         DocumentUploadResponse with document_id and task_id.
#
#     Raises:
#         HTTPException: If file type is not supported or upload fails.
#     """
#     try:
#         logger.info(f"Starting document upload for org: {org_id}, filename: {file.filename}")
#
#         # Validate file extension
#         try:
#             factory = ComponentFactory(settings)
#             parser = factory.get_parser()
#
#             if not parser.supports_file(file.filename or ""):
#                 logger.warning(f"Unsupported file type: {file.filename}")
#                 raise HTTPException(
#                     status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
#                     detail=f"Unsupported file type. Supported: {parser.supported_extensions}",
#                 )
#         except HTTPException:
#             raise
#         except Exception as e:
#             logger.error(f"Error while validating file type: {e}", exc_info=True)
#             raise HTTPException(
#                 status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#                 detail="Error validating file type",
#             ) from e
#
#         # Generate file path
#         file_id = uuid.uuid4()
#         safe_filename = file.filename or f"upload_{file_id}"
#         file_path = settings.upload_dir / f"{file_id}_{safe_filename}"
#
#         # Ensure upload directory exists
#         try:
#             settings.upload_dir.mkdir(parents=True, exist_ok=True)
#         except Exception as e:
#             logger.error(f"Failed to create upload directory: {e}", exc_info=True)
#             raise HTTPException(
#                 status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#                 detail="Failed to create upload directory",
#             ) from e
#
#         # Save file to disk
#         try:
#             content = await file.read()
#             with open(file_path, "wb") as f:
#                 f.write(content)
#
#             file_size = os.path.getsize(file_path)
#             logger.info(f"Saved file to: {file_path}, size: {file_size} bytes")
#
#         except Exception as e:
#             logger.error(f"Failed to save file: {e}", exc_info=True)
#             raise HTTPException(
#                 status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#                 detail="Failed to save uploaded file",
#             ) from e
#
#         # Create document record
#         try:
#             # Validate that the organization exists
#             from sqlalchemy import select
#             from app.db.models import Organization
#
#             org_query = select(Organization).where(Organization.id == org_id)
#             org_result = await session.execute(org_query)
#             organization = org_result.scalar_one_or_none()
#
#             if not organization:
#                 logger.error(f"Organization not found: {org_id}")
#                 raise HTTPException(
#                     status_code=status.HTTP_404_NOT_FOUND,
#                     detail=f"Organization with ID {org_id} not found",
#                 )
#
#             # Use default test user ID until authentication is implemented
#             # This user is created during database initialization
#             DEFAULT_USER_ID = uuid.UUID("3c5ff1b3-b0d6-4dba-b254-a2be667bbd52")
#
#             document = Document(
#                 filename=safe_filename,
#                 file_path=str(file_path),
#                 file_size=file_size,
#                 mime_type=file.content_type,
#                 status=DocumentStatus.QUEUED,
#                 org_id=org_id,
#                 # TODO: Set user_id from authenticated user
#                 user_id=DEFAULT_USER_ID,  # Using default seeded user
#             )
#
#             session.add(document)
#             await session.commit()
#             await session.refresh(document)
#
#             logger.info(f"Created document record: {document.id}")
#
#         except SQLAlchemyError as e:
#             logger.error(f"Database error while creating document: {e}", exc_info=True)
#             await session.rollback()
#
#             # Clean up uploaded file if database insert failed
#             try:
#                 if os.path.exists(file_path):
#                     os.remove(file_path)
#                     logger.info(f"Cleaned up file after DB error: {file_path}")
#             except Exception as cleanup_error:
#                 logger.error(f"Failed to clean up file: {cleanup_error}", exc_info=True)
#
#             raise HTTPException(
#                 status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#                 detail="Failed to create document record",
#             ) from e
#         except Exception as e:
#             logger.error(f"Unexpected error while creating document: {e}", exc_info=True)
#             await session.rollback()
#             raise HTTPException(
#                 status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#                 detail="Unexpected error creating document",
#             ) from e
#
#         # Dispatch to Celery
#         try:
#             task = process_document_task.delay(str(document.id))
#             task_id = task.id
#
#             logger.info(f"Queued document {document.id} for processing, task: {task_id}")
#
#         except Exception as e:
#             logger.error(f"Failed to queue Celery task: {e}", exc_info=True)
#             # Don't fail the upload if Celery is unavailable
#             # The document can be processed later
#             task_id = None
#
#         return DocumentUploadResponse(
#             document_id=document.id,
#             task_id=task_id,
#             status=DocumentStatus.QUEUED,
#             message="Document queued for processing",
#         )
#
#     except HTTPException:
#         # Re-raise HTTP exceptions as-is
#         raise
#     except Exception as e:
#         logger.error(f"Unexpected error in upload_document: {e}", exc_info=True)
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="An unexpected error occurred",
#         ) from e
#
#
# @router.get("/status", response_model=DocumentListResponse)
# async def get_document_status(
#     org_id: uuid.UUID = Depends(get_org_id),
#     session: AsyncSession = Depends(get_db),
#     page: int = 1,
#     page_size: int = 20,
#     status_filter: DocumentStatus | None = None,
# ) -> DocumentListResponse:
#     """Get status of documents for an organization.
#
#     Args:
#         org_id: Organization ID from header.
#         session: Database session.
#         page: Page number (1-indexed).
#         page_size: Number of results per page.
#         status_filter: Optional filter by document status.
#
#     Returns:
#         DocumentListResponse with paginated documents.
#     """
#     try:
#         logger.info(f"Fetching document status for org: {org_id}, page: {page}")
#
#         from sqlalchemy import select, desc, func
#
#         # Build query
#         query = select(Document).where(Document.org_id == org_id)
#
#         if status_filter:
#             query = query.where(Document.status == status_filter)
#
#         # Order by most recent first
#         query = query.order_by(desc(Document.created_at))
#
#         # Get total count
#         try:
#             count_query = select(func.count()).select_from(query.subquery())
#             total_result = await session.execute(count_query)
#             total = total_result.scalar_one() or 0
#         except Exception as e:
#             logger.error(f"Error getting document count: {e}", exc_info=True)
#             total = 0
#
#         # Apply pagination
#         offset = (page - 1) * page_size
#         query = query.offset(offset).limit(page_size)
#
#         # Execute query
#         try:
#             result = await session.execute(query)
#             documents = result.scalars().all()
#         except Exception as e:
#             logger.error(f"Error fetching documents: {e}", exc_info=True)
#             documents = []
#
#         logger.info(f"Found {len(documents)} documents for org: {org_id}")
#
#         return DocumentListResponse(
#             documents=[DocumentStatusResponse.model_validate(doc) for doc in documents],
#             total=total,
#             page=page,
#             page_size=page_size,
#         )
#
#     except Exception as e:
#         logger.error(f"Unexpected error in get_document_status: {e}", exc_info=True)
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Error fetching document status",
#         ) from e
#
#
# @router.get("/status/{document_id}", response_model=DocumentStatusResponse)
# async def get_single_document_status(
#     document_id: uuid.UUID,
#     org_id: uuid.UUID = Depends(get_org_id),
#     session: AsyncSession = Depends(get_db),
# ) -> DocumentStatusResponse:
#     """Get status of a specific document.
#
#     Args:
#         document_id: The document ID.
#         org_id: Organization ID from header (for tenant isolation).
#         session: Database session.
#
#     Returns:
#         DocumentStatusResponse with document details.
#
#     Raises:
#         HTTPException: If document not found or access denied.
#     """
#     try:
#         logger.info(f"Fetching document {document_id} for org: {org_id}")
#
#         from sqlalchemy import select
#
#         query = select(Document).where(
#             Document.id == document_id,
#             Document.org_id == org_id,
#         )
#
#         result = await session.execute(query)
#         document = result.scalar_one_or_none()
#
#         if not document:
#             logger.warning(f"Document {document_id} not found for org: {org_id}")
#             raise HTTPException(
#                 status_code=status.HTTP_404_NOT_FOUND,
#                 detail="Document not found",
#             )
#
#         logger.info(f"Found document {document_id} with status: {document.status}")
#         return DocumentStatusResponse.model_validate(document)
#
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error fetching document {document_id}: {e}", exc_info=True)
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Error fetching document",
#         ) from e
#
#
# @router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
# async def delete_document(
#     document_id: uuid.UUID,
#     org_id: uuid.UUID = Depends(get_org_id),
#     session: AsyncSession = Depends(get_db),
#     settings: Settings = Depends(get_settings),
# ) -> None:
#     """Delete a document and its associated data.
#
#     Args:
#         document_id: The document ID to delete.
#         org_id: Organization ID from header.
#         session: Database session.
#         settings: Application settings.
#
#     Raises:
#         HTTPException: If document not found.
#     """
#     try:
#         logger.info(f"Deleting document {document_id} for org: {org_id}")
#
#         from sqlalchemy import select
#
#         query = select(Document).where(
#             Document.id == document_id,
#             Document.org_id == org_id,
#         )
#
#         result = await session.execute(query)
#         document = result.scalar_one_or_none()
#
#         if not document:
#             logger.warning(f"Document {document_id} not found for deletion")
#             raise HTTPException(
#                 status_code=status.HTTP_404_NOT_FOUND,
#                 detail="Document not found",
#             )
#
#         # Delete file from disk
#         try:
#             if os.path.exists(document.file_path):
#                 os.remove(document.file_path)
#                 logger.info(f"Deleted file: {document.file_path}")
#             else:
#                 logger.warning(f"File not found for deletion: {document.file_path}")
#         except Exception as e:
#             logger.error(f"Failed to delete file: {e}", exc_info=True)
#             # Continue with database deletion even if file deletion fails
#
#         # Delete from database
#         try:
#             await session.delete(document)
#             await session.commit()
#             logger.info(f"Deleted document {document_id} from database")
#         except SQLAlchemyError as e:
#             logger.error(f"Database error while deleting document: {e}", exc_info=True)
#             await session.rollback()
#             raise HTTPException(
#                 status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#                 detail="Failed to delete document from database",
#             ) from e
#
#         # TODO: Delete vectors from Qdrant
#         logger.info(f"Document {document_id} deleted successfully (vectors not yet implemented)")
#
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Unexpected error deleting document {document_id}: {e}", exc_info=True)
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Error deleting document",
#         ) from e
