"""Template engine domain models.

Pydantic models specific to template analysis and injection.
These models are moved here to avoid circular imports with the API layer.
"""

from pydantic import BaseModel, Field


class DetectedVariable(BaseModel):
    """A variable detected in the template."""

    original_text: str = Field(description="The exact text found in the document")
    suggested_variable_name: str = Field(description="Suggested snake_case variable name")
    rationale: str = Field(description="Why this was identified as dynamic")
    paragraph_index: int = Field(description="Paragraph position for safe replacement")
