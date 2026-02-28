"""Celery worker entry point.

Background worker for processing templates asynchronously.
Uses an event loop to run async tasks within Celery workers.
"""

import asyncio
import logging
import ssl
import uuid
from datetime import datetime
from typing import Any

import httpx
from celery import Celery, shared_task
from celery.signals import worker_ready
import sqlalchemy
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.factory import ComponentFactory
from app.db.models import TemplateStatus, TemplateStorage
from app.db.session import get_async_session

logger = logging.getLogger(__name__)

# Initialize Celery app
settings: Settings = get_settings()

# Configure SSL based on the broker URL scheme
broker_url = settings.celery_broker_url
backend_url = settings.celery_result_backend
broker_ssl_config = None
backend_ssl_config = None

if broker_url.startswith("rediss://"):
    broker_ssl_config = {"ssl_cert_reqs": ssl.CERT_NONE}
if backend_url.startswith("rediss://"):
    backend_ssl_config = {"ssl_cert_reqs": ssl.CERT_NONE}

celery_app = Celery(
    "template_intelligence_worker",
    broker=broker_url,
    backend=backend_url,
    broker_use_ssl=broker_ssl_config,
    redis_backend_use_ssl=backend_ssl_config,
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

    IMPORTANT: Database connections are tied to event loops.
    We close the engine after each task to prevent connection
    reuse across different loops.

    Args:
        coro: The async coroutine to run.

    Returns:
        The result of the coroutine.
    """
    from app.db.session import close_db

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(coro)
        return result
    finally:
        # Close database connections to prevent loop reuse issues
        # Use the global settings that were already loaded at module level
        try:
            loop.run_until_complete(close_db(settings))
        except Exception as e:
            logger.warning(f"Error closing database: {e}")

        # Wait for all pending tasks to complete (e.g., httpx cleanup)
        # This prevents "Event loop is closed" errors from async clients
        try:
            pending = asyncio.all_tasks(loop)
            if pending:
                for task in pending:
                    task.cancel()
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                
            # Properly shutdown async resource generators
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception as e:
            logger.warning(f"Error waiting for pending tasks: {e}")
        finally:
            loop.close()


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

    return {
        "status": "healthy",
        "database": "connected",
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

            # === MEMORY INJECTION: Retrieve learned style rules from Qdrant ===
            try:
                from openai import OpenAI
                from app.db.qdrant_client import query_preferences

                oai = OpenAI(
                    api_key=settings.openrouter_api_key,
                    base_url=settings.openai_base_url,
                )

                # Embed a generic query to retrieve relevant style rules
                embed_resp = oai.embeddings.create(
                    model="openai/text-embedding-3-small",
                    input="template analysis style and formatting rules",
                    encoding_format="float",
                )
                query_vector = embed_resp.data[0].embedding

                # Query Qdrant — use org_id as a proxy for adviser_id in template context
                org_id_str = str(template.org_id)
                style_rules = query_preferences(
                    adviser_id=org_id_str,
                    query_embedding=query_vector,
                    top_k=3,
                    settings=settings,
                )

                if style_rules:
                    rules_text = "\n".join(
                        f"  {i+1}. {r['rule_text']}" for i, r in enumerate(style_rules)
                    )
                    memory_injection = (
                        f"\n\nCRITICAL INSTRUCTIONS (learned from previous feedback):\n"
                        f"{rules_text}\n"
                        f"Apply these rules when analyzing and naming variables."
                    )

                    if custom_prompt:
                        custom_prompt = custom_prompt + memory_injection
                    else:
                        custom_prompt = memory_injection

                    logger.info(
                        f"Injected {len(style_rules)} learned style rules into analysis prompt"
                    )
                else:
                    logger.info("No learned style rules found — using base prompt only")

            except Exception as mem_err:
                # Non-fatal: if Qdrant is down, proceed without memory
                logger.warning(f"Memory retrieval failed (non-fatal): {mem_err}")

            # Run TemplateAnalyzer with (potentially augmented) prompt
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


# =============================================================================
# Report Learning & Preference Tasks  (Director-Typist Model)
# =============================================================================
#
# In the Director-Typist model the user gives *natural-language feedback*
# ("make this more formal"), the AI rewrites, and on approval we store the
# user's feedback string directly as the stylistic rule.  No LLM inference
# step is needed because the user already stated their intent explicitly.
# =============================================================================


@shared_task(bind=True, name="app.worker.store_preference")
def store_preference_task(self, payload: dict) -> dict:
    """Store a learned style preference in Qdrant (Feature 3).

    Pipeline:
      1. Read the user_feedback string from the payload.
      2. Call OpenRouter text-embedding-3-small to embed it.
      3. Upsert the vector + metadata into Qdrant org_style_preferences.

    Because the user explicitly stated the rule ("make it more formal"),
    we skip the expensive LLM-inference step entirely.

    Args:
        self:    Celery task instance.
        payload: Dict with keys:
                   adviser_id, user_feedback, original_text, chosen_text, org_id.

    Returns:
        Dict with status and the Qdrant point ID.
    """
    adviser_id   = payload.get("adviser_id", "unknown")
    user_feedback = payload.get("user_feedback", "")
    original_text = payload.get("original_text", "")
    chosen_text   = payload.get("chosen_text", "")
    org_id        = payload.get("org_id", "")

    if not user_feedback:
        logger.warning(f"Empty user_feedback for adviser {adviser_id} — skipping")
        return {"status": "skipped", "adviser_id": adviser_id, "reason": "empty feedback"}

    try:
        logger.info(f"Storing preference for adviser {adviser_id}: '{user_feedback[:80]}'")

        settings = get_settings()

        # ── Step 1: Embed the user_feedback string ────────────────────────
        from openai import OpenAI

        oai = OpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openai_base_url,
        )
        embed_resp = oai.embeddings.create(
            model="openai/text-embedding-3-small",
            input=user_feedback,
            encoding_format="float",
        )
        embedding_vector = embed_resp.data[0].embedding
        logger.info(f"Embedded user_feedback ({len(embedding_vector)} dims)")

        # ── Step 2: Ensure collection exists, then upsert ─────────────────
        from app.db.qdrant_client import ensure_collection, upsert_preference

        ensure_collection(settings)

        point_id = upsert_preference(
            adviser_id=adviser_id,
            rule_text=user_feedback,
            embedding=embedding_vector,
            org_id=org_id,
            example_original=original_text,
            example_edited=chosen_text,
            settings=settings,
        )

        logger.info(f"✅ Preference stored: point_id={point_id}")

        return {
            "status": "completed",
            "adviser_id": adviser_id,
            "point_id": point_id,
            "rule_text": user_feedback,
        }

    except Exception as e:
        logger.exception(f"store_preference_task failed for {adviser_id}: {e}")
        return {"status": "failed", "adviser_id": adviser_id, "error": str(e)}


# Legacy aliases so existing callers keep working
@shared_task(bind=True, name="app.worker.learn_preference")
def learn_preference_task(self, payload: dict) -> dict:
    """Legacy alias — routes to store_preference_task."""
    return store_preference_task(payload)


@shared_task(bind=True, name="app.worker.infer_preference")
def infer_preference_task(self, payload: dict) -> dict:
    """Legacy alias — routes to store_preference_task."""
    return store_preference_task(payload)


# =============================================================================
# Dual-Layer Feedback Loop
# =============================================================================

@shared_task(
    bind=True,
    name="app.worker.process_feedback_task",
    autoretry_for=(sqlalchemy.exc.OperationalError,),
    retry_kwargs={"max_retries": 3},
    retry_backoff=True,
)
def process_feedback_task(self, payload: dict) -> dict[str, Any]:
    """Process dual-layer feedback: Stylistic and Procedural."""
    logger.info(f"Starting feedback processing: {payload.get('adviser_id')}")
    try:
        result = run_async(_process_feedback_async(payload))
        logger.info(f"Successfully processed feedback for {payload.get('adviser_id')}")
        return result
    except Exception as e:
        logger.error(f"Error processing feedback: {e}", exc_info=True)
        # Re-raise to trigger Celery retry
        raise

async def _process_feedback_async(payload: dict) -> dict[str, Any]:
    """Async execution of the dual-layer feedback loop."""
    from app.core.memory import get_memory_manager
    from app.db.session import get_session_maker
    from app.db.models import ProceduralMemory, EpisodicMemory
    import uuid

    adviser_id = payload.get("adviser_id")
    client_id = payload.get("client_id")
    topic = payload.get("topic")
    stylistic_feedback = payload.get("stylistic_feedback")
    procedural_corrections = payload.get("procedural_corrections", [])

    results = {"semantic": False, "procedural": 0, "episodic": False}

    # Track A: Semantic Memory
    if stylistic_feedback:
        logger.info(f"Processing Semantic Memory for {adviser_id}")
        mem_manager = await get_memory_manager()
        try:
            await mem_manager.add_preference(
                adviser_id=adviser_id,
                text=stylistic_feedback,
                metadata={"topic": topic, "client_id": client_id},
            )
            results["semantic"] = True
        except Exception as e:
            logger.error(f"Failed to save semantic memory: {e}")
            # Non-blocking, continue with Track B & C

    # Track B & C: Procedural and Episodic Memory
    session_maker = get_session_maker()
    async with session_maker() as session:
        # Track B: Procedural Memory
        if procedural_corrections:
            logger.info(f"Processing Procedural Memory for {adviser_id} ({len(procedural_corrections)} corrections)")
            for correction in procedural_corrections:
                proc_entry = ProceduralMemory(
                    id=uuid.uuid4(),
                    adviser_id=adviser_id,
                    variable_name=correction.get("variable_name"),
                    correction_rule=correction.get("correction_rule"),
                )
                session.add(proc_entry)
            results["procedural"] = len(procedural_corrections)

        # Track C: Episodic Memory
        logger.info(f"Processing Episodic Memory for {adviser_id}")
        ep_entry = EpisodicMemory(
            id=uuid.uuid4(),
            adviser_id=adviser_id,
            client_id=client_id,
            event_type="feedback_captured",
            event_metadata={
                "topic": topic,
                "stylistic_feedback_provided": bool(stylistic_feedback),
                "procedural_corrections_count": len(procedural_corrections) if procedural_corrections else 0,
            },
        )
        session.add(ep_entry)
        results["episodic"] = True

        # Commit all DB operations
        await session.commit()

    return {"status": "success", "results": results}


# ─────────────────────────────────────────────────────────────────────────────
# Feature 4 — Smart Generation Interception (RAG from Qdrant)
# ─────────────────────────────────────────────────────────────────────────────


@shared_task(bind=True, name="app.worker.generate_personalized_report")
def generate_personalized_report(
    self, adviser_id: str, prompt: str, org_id: str = ""
) -> str:
    """Generate a report augmented with the adviser's learned preferences.

    RAG pipeline:
      1. Embed the prompt / topic using text-embedding-3-small.
      2. Query Qdrant `org_style_preferences` for top-3 rules (filtered by adviser_id).
      3. Inject retrieved rules into the LLM system prompt.
      4. Call gpt-4o-mini with the augmented prompt.

    Args:
        self:       Celery task instance.
        adviser_id: Adviser for preference lookup.
        prompt:     The section / topic to generate.
        org_id:     Organization ID (optional, for logging).

    Returns:
        Generated report string.
    """
    try:
        logger.info(f"Generating personalized report for adviser {adviser_id}")

        settings = get_settings()

        # ── Step 1: Embed the incoming prompt ─────────────────────────────
        from openai import OpenAI

        oai = OpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openai_base_url,
        )
        embed_resp = oai.embeddings.create(
            model="openai/text-embedding-3-small",
            input=prompt,
            encoding_format="float",
        )
        prompt_vector = embed_resp.data[0].embedding

        # ── Step 2: Query Qdrant for top 3 style rules ───────────────────
        from app.db.qdrant_client import query_preferences

        matches = query_preferences(
            adviser_id=adviser_id,
            query_embedding=prompt_vector,
            top_k=3,
            settings=settings,
        )

        # Build the rules block for prompt injection
        rules_lines = []
        for m in matches:
            rule = m.get("rule_text", "")
            if rule:
                rules_lines.append(f"  - {rule}")
                logger.info(f"Retrieved preference (score={m.get('score', 0):.3f}): {rule}")

        if rules_lines:
            rules_block = (
                "IMPORTANT — Apply these stylistic rules learned from this adviser's "
                "previous feedback:\n" + "\n".join(rules_lines)
            )
        else:
            rules_block = "No specific style preferences found. Use standard professional tone."

        # ── Step 3: Generate with augmented system prompt ─────────────────
        system_prompt = (
            "You are an expert UK financial advisory report writer.\n\n"
            f"{rules_block}\n\n"
            "Write in a clear, compliant style following FCA regulations. "
            "Maintain accuracy and professionalism throughout."
        )

        gen_resp = oai.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=2000,
        )

        report_content = gen_resp.choices[0].message.content or ""
        logger.info(
            f"Generated personalized report for adviser {adviser_id} "
            f"({len(report_content)} chars, {len(matches)} rules applied)"
        )

        return report_content

    except Exception as e:
        logger.exception(
            f"generate_personalized_report failed for adviser {adviser_id}: {e}"
        )
        return f"Error: generation failed — {e}"


if __name__ == "__main__":
    # Run the worker
    celery_app.start()
