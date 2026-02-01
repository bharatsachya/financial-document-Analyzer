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
from app.db.models import Document, DocumentStatus
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


if __name__ == "__main__":
    # Run the worker
    celery_app.start()
