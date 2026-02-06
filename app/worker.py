"""Celery worker entry point.

Background worker for processing documents asynchronously.
Uses an event loop to run async tasks within Celery workers.
"""

import asyncio
import logging
import os
import uuid
from datetime import datetime

from celery import Celery, shared_task
from celery.signals import worker_ready
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.factory import ComponentFactory
from app.db.models import Document, DocumentStatus, TemplateStatus, TemplateStorage
from app.db.session import get_async_session
from app.interfaces.chunker import BaseChunker, Chunk
from app.interfaces.embedder import BaseEmbedder
from app.interfaces.parser import BaseParser, Document as ParsedDocument
from app.interfaces.vector_store import BaseVectorStore

logger = logging.getLogger(__name__)

# Initialize Celery app
settings: Settings = get_settings()

celery_app = Celery(
    "doc_intelligence_worker",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.worker"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max per task
    task_soft_time_limit=3300,  # 55 minutes soft limit
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=50,
)


@worker_ready.connect
def on_worker_ready(**kwargs):
    """Log when worker is ready."""
    logger.info("Celery worker is ready and listening for tasks")


def run_async(coro):
    """Run an async coroutine in a new event loop.

    Celery workers don't have a running event loop, so we need
    to create one for async operations.

    Args:
        coro: The async coroutine to run.

    Returns:
        The result of the coroutine.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@shared_task(bind=True, name="app.worker.process_document")
def process_document_task(self, document_id: str) -> dict:
    """Process a document through the ingestion pipeline.

    This Celery task executes the full pipeline:
    1. Parsing -> 2. Chunking -> 3. Embedding -> 4. Vector Store Upsert

    Args:
        self: Celery task instance (for bind=True).
        document_id: The UUID of the document to process.

    Returns:
        Dict with processing results.
    """
    try:
        logger.info(f"Starting document processing: {document_id}")
        return run_async(_process_document_async(document_id))
    except Exception as e:
        logger.exception(f"Fatal error in process_document_task for {document_id}: {e}")
        return {
            "status": "failed",
            "document_id": document_id,
            "error": f"Fatal error: {str(e)}",
        }


async def _process_document_async(document_id: str) -> dict:
    """Async implementation of document processing.

    Args:
        document_id: The UUID of the document to process.

    Returns:
        Dict with processing results.
    """
    settings = get_settings()
    factory = ComponentFactory(settings)

    # Get async database session
    async for session in get_async_session(settings):
        document = None
        try:
            # Fetch document from database
            try:
                from sqlalchemy import select

                query = select(Document).where(Document.id == uuid.UUID(document_id))
                result = await session.execute(query)
                document = result.scalar_one_or_none()

                if not document:
                    logger.error(f"Document {document_id} not found")
                    return {"status": "error", "message": "Document not found"}

                logger.info(f"Processing document {document_id}: {document.filename}")

            except Exception as e:
                logger.error(f"Error fetching document {document_id}: {e}", exc_info=True)
                return {"status": "error", "message": f"Error fetching document: {str(e)}"}

            # Update status to PARSING
            try:
                document.status = DocumentStatus.PARSING
                document.processing_started_at = datetime.utcnow()
                document.error_message = None
                await session.commit()
                logger.info(f"Document {document_id} status updated to PARSING")
            except SQLAlchemyError as e:
                logger.error(f"Database error updating status to PARSING: {e}", exc_info=True)
                await session.rollback()
                return {"status": "error", "message": "Database error"}

            # Step 1: Parse
            try:
                parser = factory.get_parser()
                parsed_documents = await _parse_document(parser, document.file_path)

                if not parsed_documents:
                    raise ValueError("Parser returned no documents")

                logger.info(f"Parsed {len(parsed_documents)} document(s) from {document.filename}")

            except FileNotFoundError as e:
                logger.error(f"File not found: {document.file_path}", exc_info=True)
                document.status = DocumentStatus.FAILED
                document.error_message = f"File not found: {str(e)}"
                await session.commit()
                return {"status": "failed", "document_id": document_id, "error": str(e)}
            except Exception as e:
                logger.error(f"Parse error for document {document_id}: {e}", exc_info=True)
                document.status = DocumentStatus.FAILED
                document.error_message = f"Parse error: {str(e)}"
                await session.commit()
                return {"status": "failed", "document_id": document_id, "error": str(e)}

            # Update status to CHUNKING
            try:
                document.status = DocumentStatus.CHUNKING
                await session.commit()
            except SQLAlchemyError as e:
                logger.error(f"Database error updating status to CHUNKING: {e}", exc_info=True)
                await session.rollback()
                return {"status": "error", "message": "Database error"}

            # Step 2: Chunk
            try:
                chunker = factory.get_chunker()
                all_chunks = await _chunk_documents(chunker, parsed_documents)

                if not all_chunks:
                    raise ValueError("Chunker returned no chunks")

                document.chunk_count = len(all_chunks)
                logger.info(f"Created {len(all_chunks)} chunks from {document.filename}")

            except Exception as e:
                logger.error(f"Chunking error for document {document_id}: {e}", exc_info=True)
                document.status = DocumentStatus.FAILED
                document.error_message = f"Chunking error: {str(e)}"
                await session.commit()
                return {"status": "failed", "document_id": document_id, "error": str(e)}

            # Update status to EMBEDDING
            try:
                document.status = DocumentStatus.EMBEDDING
                await session.commit()
            except SQLAlchemyError as e:
                logger.error(f"Database error updating status to EMBEDDING: {e}", exc_info=True)
                await session.rollback()
                return {"status": "error", "message": "Database error"}

            # Step 3: Embed
            try:
                embedder = factory.get_embedder()
                chunk_texts = [chunk.content for chunk in all_chunks]
                embeddings = await embedder.embed(chunk_texts)

                if len(embeddings) != len(all_chunks):
                    raise ValueError(f"Embedding count mismatch: {len(embeddings)} != {len(all_chunks)}")

                logger.info(f"Generated {len(embeddings)} embeddings")

            except Exception as e:
                logger.error(f"Embedding error for document {document_id}: {e}", exc_info=True)
                document.status = DocumentStatus.FAILED
                document.error_message = f"Embedding error: {str(e)}"
                await session.commit()
                return {"status": "failed", "document_id": document_id, "error": str(e)}

            # Update status to INDEXING
            try:
                document.status = DocumentStatus.INDEXING
                await session.commit()
            except SQLAlchemyError as e:
                logger.error(f"Database error updating status to INDEXING: {e}", exc_info=True)
                await session.rollback()
                return {"status": "error", "message": "Database error"}

            # Step 4: Upsert to vector store
            try:
                vector_store = factory.get_vector_store()
                collection_name = f"org_{document.org_id}"

                # Prepare metadata
                metadata_list = [
                    {
                        **chunk.metadata,
                        "document_id": str(document.id),
                        "filename": document.filename,
                        "org_id": str(document.org_id),
                    }
                    for chunk in all_chunks
                ]

                upsert_result = await vector_store.upsert(
                    collection_name=collection_name,
                    chunks=chunk_texts,
                    embeddings=embeddings,
                    metadata=metadata_list,
                )

                document.vector_count = upsert_result.upserted_count
                logger.info(f"Upserted {upsert_result.upserted_count} vectors to collection {collection_name}")

            except Exception as e:
                logger.error(f"Vector store error for document {document_id}: {e}", exc_info=True)
                document.status = DocumentStatus.FAILED
                document.error_message = f"Vector store error: {str(e)}"
                await session.commit()
                return {"status": "failed", "document_id": document_id, "error": str(e)}

            # Update status to COMPLETED
            try:
                document.status = DocumentStatus.COMPLETED
                document.processing_completed_at = datetime.utcnow()
                await session.commit()

                logger.info(
                    f"Successfully processed document {document_id}: "
                    f"{len(all_chunks)} chunks, {upsert_result.upserted_count} vectors"
                )

                return {
                    "status": "completed",
                    "document_id": str(document.id),
                    "chunk_count": len(all_chunks),
                    "vector_count": upsert_result.upserted_count,
                }

            except SQLAlchemyError as e:
                logger.error(f"Database error updating status to COMPLETED: {e}", exc_info=True)
                await session.rollback()
                return {"status": "error", "message": "Database error"}

        except Exception as e:
            logger.exception(f"Unexpected error processing document {document_id}: {e}")

            # Update status to FAILED if we have a document object
            if document:
                try:
                    document.status = DocumentStatus.FAILED
                    document.error_message = str(e)[:2048]  # Truncate if needed
                    document.processing_completed_at = datetime.utcnow()
                    await session.commit()
                except Exception as commit_error:
                    logger.error(f"Failed to update document to FAILED status: {commit_error}", exc_info=True)
                    await session.rollback()

            return {
                "status": "failed",
                "document_id": document_id,
                "error": str(e),
            }


async def _parse_document(parser: BaseParser, file_path: str) -> list[ParsedDocument]:
    """Parse a document using the configured parser.

    Args:
        parser: The parser instance.
        file_path: Path to the file.

    Returns:
        List of parsed documents.

    Raises:
        FileNotFoundError: If file doesn't exist.
        RuntimeError: If parsing fails.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    return await parser.aload_data(file_path)


async def _chunk_documents(
    chunker: BaseChunker, documents: list[ParsedDocument]
) -> list[Chunk]:
    """Chunk parsed documents.

    Args:
        chunker: The chunker instance.
        documents: List of parsed documents.

    Returns:
        List of all chunks from all documents.
    """
    all_chunks: list[Chunk] = []

    for doc in documents:
        try:
            chunks = chunker.chunk(doc.content)
            # Add document metadata to each chunk
            for chunk in chunks:
                # Merge chunk metadata with document metadata
                combined_metadata = {**doc.metadata, **chunk.metadata}
                all_chunks.append(Chunk(content=chunk.content, metadata=combined_metadata))
        except Exception as e:
            logger.error(f"Error chunking document: {e}", exc_info=True)
            raise

    return all_chunks


@shared_task(name="app.worker.health_check")
def health_check_task() -> dict:
    """Health check task for monitoring worker status.

    Returns:
        Dict with worker health status.
    """
    try:
        logger.debug("Running health check")
        return run_async(_health_check_async())
    except Exception as e:
        logger.exception(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
        }


async def _health_check_async() -> dict:
    """Async health check implementation.

    Returns:
        Dict with worker health status.
    """
    settings = get_settings()

    # Check database connection
    try:
        async for session in get_async_session(settings):
            from sqlalchemy import text

            await session.execute(text("SELECT 1"))
        logger.debug("Database health check passed")
    except Exception as e:
        logger.error(f"Database health check failed: {e}", exc_info=True)
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e),
        }

    # Check Qdrant connection
    try:
        factory = ComponentFactory(settings)
        vector_store = factory.get_vector_store()
        # Simple health check - try to list collections
        from qdrant_client import AsyncQdrantClient
        client: AsyncQdrantClient = vector_store._client  # type: ignore
        await client.get_collections()
        logger.debug("Qdrant health check passed")
    except Exception as e:
        logger.error(f"Qdrant health check failed: {e}", exc_info=True)
        return {
            "status": "unhealthy",
            "qdrant": "disconnected",
            "error": str(e),
        }

    return {
        "status": "healthy",
        "database": "connected",
        "qdrant": "connected",
    }


# =============================================================================
# Template Processing Tasks
# =============================================================================


@shared_task(bind=True, name="app.worker.analyze_template")
def analyze_template_task(self, template_id: str) -> dict:
    """Analyze a template asynchronously to detect variables.

    This Celery task executes template analysis:
    1. Update status to ANALYZING
    2. Run TemplateAnalyzer.analyze()
    3. Store detected_variables in DB
    4. Update status to COMPLETED or FAILED
    5. Check if batch is complete, if so start next batch

    Args:
        self: Celery task instance (for bind=True).
        template_id: The UUID of the template to analyze.

    Returns:
        Dict with processing results.
    """
    try:
        logger.info(f"Starting template analysis: {template_id}")
        return run_async(_analyze_template_async(template_id))
    except Exception as e:
        logger.exception(f"Fatal error in analyze_template_task for {template_id}: {e}")
        return {
            "status": "failed",
            "template_id": template_id,
            "error": f"Fatal error: {str(e)}",
        }


async def _analyze_template_async(template_id: str) -> dict:
    """Async implementation of template analysis.

    Args:
        template_id: The UUID of the template to analyze.

    Returns:
        Dict with processing results.
    """
    settings = get_settings()
    factory = ComponentFactory(settings)

    async for session in get_async_session(settings):
        template = None
        try:
            from sqlalchemy import select
            from app.db.models import AnalysisPrompt

            # Fetch template from database
            query = select(TemplateStorage).where(TemplateStorage.id == uuid.UUID(template_id))
            result = await session.execute(query)
            template = result.scalar_one_or_none()

            if not template:
                logger.error(f"Template {template_id} not found")
                return {"status": "error", "message": "Template not found"}

            logger.info(f"Analyzing template {template_id}: {template.original_filename}")

            # Update status to ANALYZING
            template.status = TemplateStatus.ANALYZING
            template.processing_started_at = datetime.utcnow()
            template.error_message = None
            await session.commit()
            logger.info(f"Template {template_id} status updated to ANALYZING")

            # Fetch the default prompt for this organization from database
            prompt_query = select(AnalysisPrompt).where(
                AnalysisPrompt.org_id == template.org_id,
                AnalysisPrompt.is_default == True,
            )
            prompt_result = await session.execute(prompt_query)
            default_prompt = prompt_result.scalar_one_or_none()

            custom_prompt = default_prompt.prompt_text if default_prompt else None
            if custom_prompt:
                logger.info(f"Using custom prompt '{default_prompt.name}' for template analysis")
            else:
                logger.info("Using system default prompt for template analysis")

            # Run TemplateAnalyzer with custom prompt from database
            from app.strategies.template_engine import TemplateAnalyzer

            analyzer = factory.get_template_analyzer(custom_prompt=custom_prompt)
            detected_variables = await analyzer.analyze(template.file_path)

            # Store detected variables as dict for JSON serialization
            variables_dict = [
                {
                    "original_text": var.original_text,
                    "suggested_variable_name": var.suggested_variable_name,
                    "rationale": var.rationale,
                    "paragraph_index": var.paragraph_index,
                }
                for var in detected_variables
            ]

            template.detected_variables = {"variables": variables_dict}
            template.paragraph_count = len(detected_variables) if detected_variables else 0

            # Update status to COMPLETED
            template.status = TemplateStatus.COMPLETED
            template.processing_completed_at = datetime.utcnow()
            await session.commit()

            logger.info(f"Successfully analyzed template {template_id}: {len(variables_dict)} variables detected")

            # Check if batch is complete and start next batch
            await _check_and_start_next_batch(session, template.batch_id)

            return {
                "status": "completed",
                "template_id": str(template.id),
                "variable_count": len(variables_dict),
                "batch_id": template.batch_id,
            }

        except FileNotFoundError as e:
            logger.error(f"File not found: {template.file_path if template else 'unknown'}", exc_info=True)
            if template:
                template.status = TemplateStatus.FAILED
                template.error_message = f"File not found: {str(e)}"
                await session.commit()
            return {"status": "failed", "template_id": template_id, "error": str(e)}

        except Exception as e:
            logger.error(f"Analysis error for template {template_id}: {e}", exc_info=True)
            if template:
                template.status = TemplateStatus.FAILED
                template.error_message = f"Analysis error: {str(e)}"[:2048]
                await session.commit()
            return {"status": "failed", "template_id": template_id, "error": str(e)}


@shared_task(bind=True, name="app.worker.finalize_template")
def finalize_template_task(self, template_id: str, variables: list[dict]) -> dict:
    """Finalize a template with variable injection asynchronously.

    This Celery task executes template injection:
    1. Update status to FINALIZING
    2. Run TemplateInjector.inject_tags()
    3. Save tagged file path
    4. Set download_ready=True
    5. Update status to COMPLETED
    6. Check if batch is complete, if so start next batch

    Args:
        self: Celery task instance (for bind=True).
        template_id: The UUID of the template to finalize.
        variables: List of variables with values to inject.

    Returns:
        Dict with processing results.
    """
    try:
        logger.info(f"Starting template finalization: {template_id}")
        return run_async(_finalize_template_async(template_id, variables))
    except Exception as e:
        logger.exception(f"Fatal error in finalize_template_task for {template_id}: {e}")
        return {
            "status": "failed",
            "template_id": template_id,
            "error": f"Fatal error: {str(e)}",
        }


async def _finalize_template_async(template_id: str, variables: list[dict]) -> dict:
    """Async implementation of template finalization.

    Args:
        template_id: The UUID of the template to finalize.
        variables: List of variables with values to inject.

    Returns:
        Dict with processing results.
    """
    settings = get_settings()
    factory = ComponentFactory(settings)

    async for session in get_async_session(settings):
        template = None
        try:
            from sqlalchemy import select
            from pathlib import Path

            # Fetch template from database
            query = select(TemplateStorage).where(TemplateStorage.id == uuid.UUID(template_id))
            result = await session.execute(query)
            template = result.scalar_one_or_none()

            if not template:
                logger.error(f"Template {template_id} not found")
                return {"status": "error", "message": "Template not found"}

            logger.info(f"Finalizing template {template_id}: {template.original_filename}")

            # Update status to FINALIZING
            template.status = TemplateStatus.FINALIZING
            template.processing_started_at = datetime.utcnow()
            template.error_message = None
            await session.commit()
            logger.info(f"Template {template_id} status updated to FINALIZING")

            # Run TemplateInjector
            from app.strategies.template_engine import TemplateInjector

            injector = factory.get_template_injector()

            # Generate output path
            input_path = Path(template.file_path)
            output_path = input_path.parent / f"{input_path.stem}_filled{input_path.suffix}"

            # Use inject_values to fill actual values instead of Jinja2 tags
            filled_path = await injector.inject_values(
                file_path=template.file_path,
                variables=variables,
                output_path=str(output_path),
            )

            # Update template with download info
            template.is_tagged = True
            template.download_ready = True
            template.download_url = f"/templates/download/{template_id}"
            template.file_path = filled_path  # Update to filled file path

            # Update status to COMPLETED
            template.status = TemplateStatus.COMPLETED
            template.processing_completed_at = datetime.utcnow()

            # Update injection status
            template.injection_status = "completed"
            template.injection_completed_at = datetime.utcnow()

            await session.commit()

            logger.info(f"Successfully finalized template {template_id}: {filled_path}")

            # Check if batch is complete and start next batch
            await _check_and_start_next_batch(session, template.batch_id)

            return {
                "status": "completed",
                "template_id": str(template.id),
                "download_url": template.download_url,
                "batch_id": template.batch_id,
            }

        except Exception as e:
            logger.error(f"Finalization error for template {template_id}: {e}", exc_info=True)
            if template:
                template.status = TemplateStatus.FAILED
                template.error_message = f"Finalization error: {str(e)}"[:2048]
                
                # Update injection status
                template.injection_status = "failed"
                template.injection_error_message = str(e)[:2048]
                template.injection_completed_at = datetime.utcnow()
                
                await session.commit()
            return {"status": "failed", "template_id": template_id, "error": str(e)}


@shared_task(bind=True, name="app.worker.process_batch")
def process_batch_task(self, batch_id: str) -> dict:
    """Process all templates in a batch.

    This Celery task orchestrates batch processing:
    1. Update batch_status to "processing"
    2. Queue analyze_template_task for each template in batch
    3. Use chord/group to track completion
    4. On completion, check for next_batch_id and start it

    Args:
        self: Celery task instance (for bind=True).
        batch_id: The batch ID to process.

    Returns:
        Dict with batch processing results.
    """
    try:
        logger.info(f"Starting batch processing: {batch_id}")
        return run_async(_process_batch_async(batch_id))
    except Exception as e:
        logger.exception(f"Fatal error in process_batch_task for {batch_id}: {e}")
        return {
            "status": "failed",
            "batch_id": batch_id,
            "error": f"Fatal error: {str(e)}",
        }


async def _process_batch_async(batch_id: str) -> dict:
    """Async implementation of batch processing.

    Args:
        batch_id: The batch ID to process.

    Returns:
        Dict with batch processing results.
    """
    settings = get_settings()

    async for session in get_async_session(settings):
        try:
            from sqlalchemy import select

            # Fetch all templates in the batch
            query = select(TemplateStorage).where(
                TemplateStorage.batch_id == batch_id,
                TemplateStorage.status == TemplateStatus.QUEUED,
            )
            result = await session.execute(query)
            templates = result.scalars().all()

            if not templates:
                logger.info(f"No queued templates found in batch {batch_id}")
                return {"status": "completed", "batch_id": batch_id, "template_count": 0}

            logger.info(f"Found {len(templates)} templates to process in batch {batch_id}")

            # Update batch_status to "processing" for all templates
            for template in templates:
                template.batch_status = "processing"
            await session.commit()

            # Queue analyze tasks for each template
            from celery import group

            job = group(
                analyze_template_task.s(str(template.id))
                for template in templates
            )
            result = job.apply_async()

            logger.info(f"Queued {len(templates)} analyze tasks for batch {batch_id}")

            return {
                "status": "queued",
                "batch_id": batch_id,
                "template_count": len(templates),
                "celery_task_id": result.id,
            }

        except Exception as e:
            logger.error(f"Batch processing error for {batch_id}: {e}", exc_info=True)
            return {"status": "failed", "batch_id": batch_id, "error": str(e)}


async def _check_and_start_next_batch(session: AsyncSession, current_batch_id: str | None) -> None:
    """Check if current batch is complete and start next batch if exists.

    Args:
        session: The async database session.
        current_batch_id: The current batch ID to check.
    """
    if not current_batch_id:
        return

    try:
        from sqlalchemy import select, func

        # Check if all templates in current batch are completed/failed
        query = select(func.count(TemplateStorage.id)).where(
            TemplateStorage.batch_id == current_batch_id,
            TemplateStorage.status.in_([TemplateStatus.QUEUED, TemplateStatus.ANALYZING, TemplateStatus.FINALIZING]),
        )
        result = await session.execute(query)
        pending_count = result.scalar_one()

        if pending_count > 0:
            # Batch still has pending templates
            logger.debug(f"Batch {current_batch_id} still has {pending_count} pending templates")
            return

        # Mark current batch as completed using update() for proper ORM usage
        from sqlalchemy import update
        await session.execute(
            update(TemplateStorage)
            .where(TemplateStorage.batch_id == current_batch_id)
            .values(batch_status="completed")
        )
        await session.commit()
        logger.info(f"Batch {current_batch_id} marked as completed")

        # Find next batch (templates where previous_batch_id == current_batch_id and batch_status == "queued")
        next_batch_query = select(TemplateStorage.batch_id).where(
            TemplateStorage.previous_batch_id == current_batch_id,
            TemplateStorage.batch_status == "queued",
        ).distinct().limit(1)

        next_batch_result = await session.execute(next_batch_query)
        next_batch_id = next_batch_result.scalar_one_or_none()

        if next_batch_id:
            logger.info(f"Found next batch {next_batch_id}, starting processing...")

            # Update next batch status to "processing" using update() for proper ORM usage
            await session.execute(
                update(TemplateStorage)
                .where(TemplateStorage.batch_id == next_batch_id)
                .values(batch_status="processing")
            )
            await session.commit()

            # Start processing next batch
            # Note: We need to call this in a non-blocking way
            # Using celery.send_task to queue the batch processing
            from app.worker import celery_app

            celery_app.send_task("app.worker.process_batch", args=[next_batch_id])
            logger.info(f"Queued batch processing task for {next_batch_id}")

    except Exception as e:
        logger.error(f"Error checking/starting next batch: {e}", exc_info=True)


if __name__ == "__main__":
    # Run the worker
    celery_app.start()
