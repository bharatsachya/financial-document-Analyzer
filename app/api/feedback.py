"""Feedback API endpoints for Dual-Layer Feedback Loop.

Handles capturing stylistic edits and procedural corrections from
the "Director-Typist" interface and queueing them for processing.
"""

from fastapi import APIRouter, status
from pydantic import BaseModel, Field

from app.worker import process_feedback_task

router = APIRouter(prefix="/feedback", tags=["feedback"])


class ProceduralCorrection(BaseModel):
    """A procedural correction rule for a specific variable."""

    variable_name: str = Field(description="Name of the variable being corrected")
    correction_rule: str = Field(description="The correction rule to apply (e.g., 'Always capitalize')")


class DualFeedbackPayload(BaseModel):
    """Payload for capturing dual-layer feedback: Stylistic and Procedural."""

    adviser_id: str = Field(
        description="Adviser identifier (e.g., adv_001)",
        min_length=1,
        max_length=50,
    )
    client_id: str = Field(
        description="Client identifier (e.g., client_123)",
        min_length=1,
        max_length=50,
    )
    topic: str = Field(
        description="The topic or context of the feedback (e.g., 'Investment Summary')",
        min_length=1,
    )
    stylistic_feedback: str | None = Field(
        default=None,
        description="Natural-language constraint (e.g., 'Make it more formal')",
    )
    procedural_corrections: list[ProceduralCorrection] | None = Field(
        default=None,
        description="List of specific procedural corrections for variables",
    )


@router.post(
    "/capture",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Capture style and procedural feedback",
    description="Captures feedback and enqueues an asynchronous task to process Track A (Semantic) and Track B (Procedural) memories.",
)
async def capture_feedback(payload: DualFeedbackPayload):
    """Capture dual-layer feedback from the adviser."""
    # Enqueue background task
    task = process_feedback_task.delay(payload.model_dump())

    return {
        "status": "accepted",
        "task_id": task.id,
        "message": "Feedback captured and queued for processing",
    }
