"""Procedural Memory Service for logic and learned corrections.

This is the "how" layer - storing and retrieving procedural patterns
that the adviser follows consistently, such as correction rules and
habits learned from their behavior over time.
"""

import datetime
import logging
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.db.database import ProceduralMemory, VariableCorrection

logger = logging.getLogger(__name__)


class ProceduralMemoryManager:
    """Manager for procedural memory operations.

    Handles storage and retrieval of learned correction rules,
    variable corrections, and adviser habits.
    """

    def __init__(
        self,
        settings: Settings | None = None,
    ) -> None:
        """Initialize ProceduralMemoryManager.

        Args:
            settings: Optional application settings.
        """
        self._settings = settings or get_settings()

    async def get_adviser_rules(
        self,
        session: AsyncSession,
        adviser_id: str,
        org_id: uuid.UUID | None = None,
    ) -> list[dict[str, Any]]:
        """Get all procedural rules for an adviser.

        Retrieves stored correction rules and habits for the given adviser.

        Args:
            session: Async database session.
            adviser_id: The adviser to get rules for.
            org_id: Optional org ID for filtering.

        Returns:
            List of rule dicts with keys: key, value, memory_type,
            confidence, created_at.
        """
        try:
            query = select(ProceduralMemory).where(
                ProceduralMemory.adviser_id == adviser_id
            )

            if org_id is not None:
                query = query.where(ProceduralMemory.org_id == org_id)

            result = await session.execute(query.order_by(ProceduralMemory.created_at.desc()))
            rules = result.scalars().all()

            logger.info(
                f"Retrieved {len(rules)} procedural rules for adviser {adviser_id}"
            )

            return [
                {
                    "id": str(rule.id),
                    "key": rule.key,
                    "value": rule.value,
                    "memory_type": rule.memory_type,
                    "confidence": rule.confidence,
                    "access_count": rule.access_count,
                    "created_at": rule.created_at.isoformat(),
                }
                for rule in rules
            ]

        except Exception as e:
            logger.error(
                f"Error getting procedural rules for adviser {adviser_id}: {e}",
                exc_info=True,
            )
            return []

    async def store_correction(
        self,
        session: AsyncSession,
        *,
        adviser_id: str,
        org_id: uuid.UUID,
        variable_name: str,
        incorrect_value: str,
        corrected_value: str,
        template_id: uuid.UUID | None = None,
    ) -> VariableCorrection:
        """Store a variable correction.

        Records corrections made to variable values, enabling
        better auto-injection in the future.

        Args:
            session: Async database session.
            adviser_id: The adviser who made the correction.
            org_id: Organization ID.
            variable_name: The variable being corrected.
            incorrect_value: The incorrect value that was corrected.
            corrected_value: The corrected value.
            template_id: Optional template ID the correction applies to.

        Returns:
            The created VariableCorrection record.

        Raises:
            Exception: If storage fails.
        """
        try:
            # Check if this exact correction already exists
            existing_query = select(VariableCorrection).where(
                VariableCorrection.adviser_id == adviser_id,
                VariableCorrection.variable_name == variable_name,
                VariableCorrection.incorrect_value == incorrect_value,
                VariableCorrection.corrected_value == corrected_value,
            )
            existing = await session.execute(existing_query)
            existing_record = existing.scalar_one_or_none()

            if existing_record:
                # Increment correction count
                existing_record.correction_count += 1
                existing_record.updated_at = datetime.datetime.utcnow()
                await session.commit()
                logger.info(
                    f"Updated existing correction for variable {variable_name}: "
                    f"count={existing_record.correction_count}"
                )
                return existing_record

            # Create new correction record
            correction = VariableCorrection(
                adviser_id=adviser_id,
                org_id=org_id,
                template_id=template_id,
                variable_name=variable_name,
                incorrect_value=incorrect_value,
                corrected_value=corrected_value,
                correction_count=1,
            )
            session.add(correction)
            await session.commit()
            await session.refresh(correction)

            logger.info(
                f"Stored new correction for variable {variable_name}: "
                f"'{incorrect_value}' -> '{corrected_value}'"
            )
            return correction

        except Exception as e:
            await session.rollback()
            logger.error(
                f"Error storing correction for variable {variable_name}: {e}",
                exc_info=True,
            )
            raise

    async def get_corrections_for_variable(
        self,
        session: AsyncSession,
        adviser_id: str,
        variable_name: str,
        limit: int = 10,
    ) -> list[VariableCorrection]:
        """Get corrections for a specific variable.

        Args:
            session: Async database session.
            adviser_id: The adviser to get corrections for.
            variable_name: The variable name.
            limit: Maximum number of corrections to return.

        Returns:
            List of VariableCorrection records.
        """
        try:
            query = (
                select(VariableCorrection)
                .where(
                    VariableCorrection.adviser_id == adviser_id,
                    VariableCorrection.variable_name == variable_name,
                )
                .order_by(VariableCorrection.correction_count.desc())
                .limit(limit)
            )

            result = await session.execute(query)
            corrections = result.scalars().all()

            logger.info(
                f"Retrieved {len(corrections)} corrections for "
                f"variable {variable_name} and adviser {adviser_id}"
            )
            return corrections

        except Exception as e:
            logger.error(
                f"Error getting corrections for variable {variable_name}: {e}",
                exc_info=True,
            )
            return []

    async def apply_corrections(
        self,
        session: AsyncSession,
        adviser_id: str,
        variables: dict[str, Any],
    ) -> dict[str, Any]:
        """Apply learned corrections to variable values.

        Checks for known corrections and applies them to the
        provided variable dictionary.

        Args:
            session: Async database session.
            adviser_id: The adviser whose corrections to apply.
            variables: The variable dictionary to correct.

        Returns:
            The corrected variable dictionary.
        """
        corrected = variables.copy()
        corrections_applied = []

        for var_name, value in variables.items():
            if not isinstance(value, str):
                continue

            # Get corrections for this variable
            corrections = await self.get_corrections_for_variable(
                session=session,
                adviser_id=adviser_id,
                variable_name=var_name,
                limit=1,
            )

            for correction in corrections:
                if value == correction.incorrect_value:
                    corrected[var_name] = correction.corrected_value
                    corrections_applied.append(
                        {
                            "variable": var_name,
                            "original": value,
                            "corrected": correction.corrected_value,
                        }
                    )
                    # Update access count
                    correction.correction_count += 1
                    await session.commit()
                    break

        if corrections_applied:
            logger.info(
                f"Applied {len(corrections_applied)} corrections "
                f"for adviser {adviser_id}"
            )

        return corrected


# Global instance
_global_manager: ProceduralMemoryManager | None = None


def get_procedural_memory_manager() -> ProceduralMemoryManager:
    """Get the global ProceduralMemoryManager instance.

    Returns:
        The singleton ProceduralMemoryManager.
    """
    global _global_manager
    if _global_manager is None:
        _global_manager = ProceduralMemoryManager()
    return _global_manager
