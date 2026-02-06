"""Logic extraction interfaces.

Defines abstract base classes for extracting conditional logic from
financial and legal documents and converting them to Jinja2 template tags.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LogicTag:
    """Represents a conditional logic tag extracted from a document.

    Attributes:
        original_text: The original paragraph text containing the logic.
        paragraph_index: Zero-based index of the paragraph in the document.
        is_conditional: Whether this paragraph contains conditional logic.
        condition_variable: The variable name used in the condition.
        operator: The comparison operator (>, ==, in, >=, <=, <).
        condition_value: The threshold or comparison value.
        jinja_wrapper: The Jinja2 opening tag (e.g., "{% if client_age > 55 %}").
        confidence: Confidence score from 0.0 to 1.0.
        reasoning: Chain-of-thought explanation of the extraction.
    """

    original_text: str
    paragraph_index: int
    is_conditional: bool
    condition_variable: str
    operator: str
    condition_value: str
    jinja_wrapper: str
    confidence: float
    reasoning: str


class BaseLogicExtractor(ABC):
    """Abstract base class for logic extraction strategies.

    Analyzes document paragraphs to detect conditional clauses (e.g.,
    "only applies if savings exceed £1,073,100") and converts them
    to Jinja2 template tags (e.g., "{% if savings > 1073100 %}").
    """

    @abstractmethod
    async def extract_logic(
        self,
        file_path: str,
        available_variables: list[str],
    ) -> list[LogicTag]:
        """Extract conditional logic from a document.

        Args:
            file_path: Path to the document to analyze.
            available_variables: List of variable names detected in the document.
                Extracted conditions must reference one of these variables.

        Returns:
            List of LogicTag objects representing detected conditional logic.
            Only tags with confidence above the threshold are included.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            LogicExtractionError: If extraction fails.
        """
        pass

    @property
    @abstractmethod
    def supported_extensions(self) -> set[str]:
        """Return supported file extensions."""
        pass


class LogicExtractionError(Exception):
    """Exception raised when logic extraction fails."""

    pass
