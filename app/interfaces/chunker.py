"""Abstract base class for text chunking strategies.

The Strategy Pattern allows different chunking implementations
to be used based on document type or user preference.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """Represents a chunk of text with metadata.

    Attributes:
        content: The text content of the chunk.
        metadata: Additional information about the chunk (section, page, etc.).
    """

    content: str
    metadata: dict


class BaseChunker(ABC):
    """Abstract base class for text chunking strategies.

    All concrete chunker implementations must inherit from this class
    and implement the `chunk` method.

    Example:
        ```python
        class MarkdownHeaderChunker(BaseChunker):
            def chunk(self, text: str) -> list[Chunk]:
                # Split by markdown headers
                pass
        ```
    """

    @abstractmethod
    def chunk(self, text: str, **kwargs: object) -> list[Chunk]:
        """Split text into smaller chunks for processing.

        Args:
            text: The input text to chunk.
            **kwargs: Additional parameters specific to the chunking strategy.

        Returns:
            A list of Chunk objects containing the chunked content and metadata.
        """
        ...

    @property
    @abstractmethod
    def max_chunk_size(self) -> int:
        """Return the maximum chunk size for this chunker.

        Returns:
            The maximum number of characters/tokens per chunk.
        """
        ...
