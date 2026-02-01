"""Abstract base class for document parsers.

The Strategy Pattern allows different parsing implementations
to be interchangeable at runtime.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    """Represents a parsed document with metadata.

    Attributes:
        content: The extracted text content from the document.
        metadata: Additional parser-specific information (page count, author, etc.).
        source: The original file path or identifier.
    """

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = ""


class BaseParser(ABC):
    """Abstract base class for document parsing strategies.

    All concrete parser implementations must inherit from this class
    and implement the `aload_data` method.

    Example:
        ```python
        class LlamaParseParser(BaseParser):
            async def aload_data(self, file_path: str) -> list[Document]:
                # Implementation here
                pass
        ```
    """

    @abstractmethod
    async def aload_data(self, file_path: str) -> list[Document]:
        """Asynchronously parse a document file.

        Args:
            file_path: The path to the document file to parse.

        Returns:
            A list of Document objects containing the parsed content and metadata.

        Raises:
            FileNotFoundError: If the file does not exist.
            ParsingError: If the file cannot be parsed.
        """
        ...

    @property
    @abstractmethod
    def supported_extensions(self) -> set[str]:
        """Return the set of file extensions supported by this parser.

        Returns:
            A set of file extensions (e.g., {'.pdf', '.docx'}).
        """
        ...

    def supports_file(self, file_path: str) -> bool:
        """Check if this parser supports the given file.

        Args:
            file_path: The path to the file to check.

        Returns:
            True if the file extension is supported, False otherwise.
        """
        import os

        _, ext = os.path.splitext(file_path)
        return ext.lower() in self.supported_extensions
