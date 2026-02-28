"""Episodic Memory Service for event logging and temporal context.

This is the "when" layer - storing specific events that occurred
at particular times, enabling temporal pattern recognition and audit trails.
"""

import datetime
import logging
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
# Use database.py for the newer models
from app.db.database import EpisodicEvent as EpisodicMemory, EventType

# Try to import FeedbackEvent, fall back if not available
try:
    from app.db.database import FeedbackEvent
    HAS_FEEDBACK_EVENT = True
except ImportError:
    HAS_FEEDBACK_EVENT = False
    # Create a fallback for compatibility
    class FeedbackEvent:
        """Fallback FeedbackEvent for compatibility."""

        def __init__(self, **kwargs):
            pass


logger = logging.getLogger(__name__)


class EpisodicMemoryManager:
    """Manager for episodic memory operations.

    Handles logging and retrieval of events for audit trails
    and temporal context.
    """

    def __init__(
        self,
        settings: Settings | None = None,
    ) -> None:
        """Initialize EpisodicMemoryManager.

        Args:
            settings: Optional application settings.
        """
        self._settings = settings or get_settings()

    async def log_event(
        self,
        session: AsyncSession,
        *,
        adviser_id: str,
        org_id: uuid.UUID,
        event_type: EventType,
        context: dict[str, Any],
        session_id: uuid.UUID | None = None,
        client_id: str | None = None,
        embedding: list[float] | None = None,
    ) -> uuid.UUID:
        """Log an episodic event.

        Args:
            session: Async database session.
            adviser_id: The adviser associated with the event.
            org_id: Organization ID.
            event_type: Type of event (feedback_given, report_generated, etc.).
            context: Full event context as JSONB.
            session_id: Optional session ID to group related events.
            client_id: Optional client ID for the event.
            embedding: Optional embedding for semantic retrieval.

        Returns:
            The UUID of the logged event.

        Raises:
            Exception: If logging fails.
        """
        try:
            event = EpisodicMemory(
                adviser_id=adviser_id,
                org_id=org_id,
                session_id=session_id,
                event_type=event_type,
                context=context,
                embedding=embedding,
            )
            session.add(event)
            await session.commit()
            await session.refresh(event)

            logger.info(
                f"Logged episodic event {event.id} "
                f"(type={event_type}, adviser={adviser_id})"
            )
            return event.id

        except Exception as e:
            await session.rollback()
            logger.error(
                f"Error logging episodic event for adviser {adviser_id}: {e}",
                exc_info=True,
            )
            raise

    async def log_report_generation(
        self,
        session: AsyncSession,
        *,
        adviser_id: str,
        org_id: uuid.UUID,
        client_id: str,
        topic: str,
        generated_text: str,
        extracted_variables: dict[str, Any],
        sources: dict[str, Any],
        session_id: uuid.UUID,
    ) -> uuid.UUID:
        """Log a report generation event.

        Args:
            session: Async database session.
            adviser_id: The adviser who generated the report.
            org_id: Organization ID.
            client_id: Client ID for the report.
            topic: The report topic.
            generated_text: The generated report content.
            extracted_variables: Variables extracted from the template.
            sources: Source information for the report.
            session_id: Session ID to group related events.

        Returns:
            The UUID of the logged event.
        """
        context = {
            "topic": topic,
            "generated_text_length": len(generated_text),
            "extracted_variables": extracted_variables,
            "sources": sources,
        }

        return await self.log_event(
            session=session,
            adviser_id=adviser_id,
            org_id=org_id,
            event_type=EventType.REPORT_GENERATED,
            context=context,
            session_id=session_id,
            client_id=client_id,
        )

    async def log_feedback(
        self,
        session: AsyncSession,
        *,
        adviser_id: str,
        org_id: uuid.UUID,
        original_text: str,
        feedback_text: str,
        chosen_text: str,
        report_section: str | None = None,
        report_type: str | None = None,
        template_id: uuid.UUID | None = None,
        episodic_event_id: uuid.UUID | None = None,
    ) -> uuid.UUID | None:
        """Log a feedback event.

        Args:
            session: Async database session.
            adviser_id: The adviser who provided feedback.
            org_id: Organization ID.
            original_text: Original AI-generated text.
            feedback_text: User's feedback.
            chosen_text: Final chosen/edited text.
            report_section: Optional report section.
            report_type: Optional report type.
            template_id: Optional template ID.
            episodic_event_id: Optional related episodic event ID.

        Returns:
            The UUID of the logged feedback event, or None if not available.
        """
        if not HAS_FEEDBACK_EVENT:
            # Log to episodic memory instead
            return await self.log_event(
                session=session,
                adviser_id=adviser_id,
                org_id=org_id,
                event_type=EventType.FEEDBACK_GIVEN,
                context={
                    "original_text": original_text,
                    "feedback_text": feedback_text,
                    "chosen_text": chosen_text,
                    "report_section": report_section,
                    "report_type": report_type,
                },
            )

        try:
            feedback = FeedbackEvent(
                adviser_id=adviser_id,
                org_id=org_id,
                episodic_event_id=episodic_event_id,
                original_text=original_text,
                feedback_text=feedback_text,
                chosen_text=chosen_text,
                report_section=report_section,
                report_type=report_type,
                template_id=template_id,
            )
            session.add(feedback)
            await session.commit()
            await session.refresh(feedback)

            logger.info(f"Logged feedback event {feedback.id} for adviser {adviser_id}")
            return feedback.id

        except Exception as e:
            await session.rollback()
            logger.error(
                f"Error logging feedback for adviser {adviser_id}: {e}",
                exc_info=True,
            )
            raise

    async def get_recent_events(
        self,
        session: AsyncSession,
        adviser_id: str,
        client_id: str | None = None,
        event_type: EventType | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get recent events for an adviser.

        Args:
            session: Async database session.
            adviser_id: The adviser to get events for.
            client_id: Optional client ID to filter by.
            event_type: Optional event type to filter by.
            limit: Maximum number of events to return.

        Returns:
            List of event dicts with keys: id, event_type, context,
            timestamp, session_id.
        """
        try:
            query = select(EpisodicMemory).where(
                EpisodicMemory.adviser_id == adviser_id
            )

            if client_id is not None:
                # Filter by client_id in context
                # Note: This requires JSONB path query
                query = query.where(
                    EpisodicMemory.context["client_id"].astext == client_id
                )

            if event_type is not None:
                query = query.where(EpisodicMemory.event_type == event_type)

            query = query.order_by(EpisodicMemory.timestamp.desc()).limit(limit)

            result = await session.execute(query)
            events = result.scalars().all()

            logger.info(
                f"Retrieved {len(events)} recent events for adviser {adviser_id}"
            )

            return [
                {
                    "id": str(event.id),
                    "event_type": event.event_type,
                    "context": event.context,
                    "timestamp": event.timestamp.isoformat(),
                    "session_id": str(event.session_id) if event.session_id else None,
                }
                for event in events
            ]

        except Exception as e:
            logger.error(
                f"Error getting recent events for adviser {adviser_id}: {e}",
                exc_info=True,
            )
            return []

    async def get_events_by_session(
        self,
        session: AsyncSession,
        session_id: uuid.UUID,
    ) -> list[EpisodicMemory]:
        """Get all events for a specific session.

        Args:
            session: Async database session.
            session_id: The session ID to retrieve events for.

        Returns:
            List of EpisodicMemory records.
        """
        try:
            query = (
                select(EpisodicMemory)
                .where(EpisodicMemory.session_id == session_id)
                .order_by(EpisodicMemory.timestamp)
            )

            result = await session.execute(query)
            events = result.scalars().all()

            logger.info(
                f"Retrieved {len(events)} events for session {session_id}"
            )
            return events

        except Exception as e:
            logger.error(
                f"Error getting events for session {session_id}: {e}",
                exc_info=True,
            )
            return []


# Global instance
_global_manager: EpisodicMemoryManager | None = None


def get_episodic_memory_manager() -> EpisodicMemoryManager:
    """Get the global EpisodicMemoryManager instance.

    Returns:
        The singleton EpisodicMemoryManager.
    """
    global _global_manager
    if _global_manager is None:
        _global_manager = EpisodicMemoryManager()
    return _global_manager
