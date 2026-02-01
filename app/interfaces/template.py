"""Template analysis and injection interfaces.

Defines abstract base classes for the Template Intelligence Engine (TIE).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TemplateDocument:
    """Template document metadata.

    Attributes:
        content: List of paragraph text by index.
        metadata: Additional document metadata.
    """

    content: list[str]
    metadata: dict[str, Any]


class BaseTemplateAnalyzer(ABC):
    """Abstract base class for template analysis strategies.

    Analyzes Word documents to detect dynamic variables that should be
    replaced with Jinja2 tags.
    """

    @abstractmethod
    async def analyze(self, file_path: str) -> list[Any]:
        """Analyze a document to detect dynamic variables.

        Args:
            file_path: Path to the Word template file.

        Returns:
            List of DetectedVariable objects with paragraph indices.

        Raises:
            FileNotFoundError: If file doesn't exist.
            AnalysisError: If analysis fails.
        """

    @property
    @abstractmethod
    def supported_extensions(self) -> set[str]:
        """Return supported file extensions."""


class BaseTemplateInjector(ABC):
    """Abstract base class for tag injection strategies.

    Injects Jinja2 tags into Word templates at detected variable locations.
    """

    @abstractmethod
    async def inject_tags(
        self,
        file_path: str,
        variables: list[Any],
        output_path: str | None = None,
    ) -> str:
        """Inject Jinja2 tags into the template.

        Args:
            file_path: Path to the original template.
            variables: List of DetectedVariable objects to inject.
            output_path: Where to save the tagged template (optional).

        Returns:
            Path to the generated template file.

        Raises:
            InjectionError: If tag injection fails.
        """
