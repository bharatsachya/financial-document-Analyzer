"""Recursive token-based chunker.

A generalized chunker that recursively splits text using multiple
separators, falling back to character-level splitting when needed.
"""

import logging
from typing import Literal

from app.interfaces.chunker import BaseChunker, Chunk

logger = logging.getLogger(__name__)


class RecursiveTokenChunker(BaseChunker):
    """Recursive chunker that tries multiple split strategies.

    This chunker attempts to split on semantic boundaries first
    (paragraphs, sentences), falling back to smaller units if needed.
    It ensures no chunk exceeds the maximum size.

    Attributes:
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Number of characters to overlap between chunks.
        separators: Ordered list of separators to try, in priority order.
    """

    def __init__(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 128,
        separators: list[str] | None = None,
    ) -> None:
        """Initialize the recursive chunker.

        Args:
            chunk_size: Maximum characters per chunk.
            chunk_overlap: Number of overlapping characters between chunks.
            separators: Ordered separators to try. Defaults to ["\n\n", "\n", ". ", " ", ""].
        """
        self._chunk_size = max(1, chunk_size)
        self._chunk_overlap = max(0, min(chunk_overlap, chunk_size // 2))

        self._separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def chunk(self, text: str, **kwargs: object) -> list[Chunk]:
        """Recursively split text into chunks.

        Args:
            text: The text to chunk.
            **kwargs: Optional override for chunk_size or chunk_overlap.

        Returns:
            A list of Chunks with metadata about position and size.
        """
        chunk_size = kwargs.get("chunk_size", self._chunk_size)
        chunk_overlap = kwargs.get("chunk_overlap", self._chunk_overlap)

        if not text.strip():
            return []

        logger.debug(
            f"Recursive chunking: {len(text)} chars, size={chunk_size}, overlap={chunk_overlap}"
        )

        chunks = self._recursive_split(text, chunk_size, chunk_overlap, self._separators)

        # Add metadata to each chunk
        result = []
        for i, chunk_text in enumerate(chunks):
            result.append(
                Chunk(
                    content=chunk_text,
                    metadata={
                        "chunker": "recursive",
                        "chunk_index": i,
                        "char_count": len(chunk_text),
                        "chunk_size": chunk_size,
                        "overlap": chunk_overlap,
                    },
                )
            )

        logger.info(f"Created {len(result)} chunks using recursive splitting")
        return result

    def _recursive_split(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int,
        separators: list[str],
    ) -> list[str]:
        """Recursively split text using the given separators.

        Args:
            text: The text to split.
            chunk_size: Maximum chunk size.
            chunk_overlap: Overlap between chunks.
            separators: Ordered list of separators to try.

        Returns:
            A list of text chunks.
        """
        # Base case: if text fits in one chunk, return it
        if len(text) <= chunk_size:
            return [text]

        # Try each separator in order
        for separator in separators:
            if separator not in text:
                continue

            # Split by separator
            parts = text.split(separator)

            # Build chunks, respecting overlap
            chunks: list[str] = []
            current_chunk = ""

            for part in parts:
                test_chunk = current_chunk + separator + part if current_chunk else part

                if len(test_chunk) <= chunk_size:
                    current_chunk = test_chunk
                else:
                    # Current chunk is full, add it to results
                    if current_chunk:
                        chunks.append(current_chunk.strip())

                    # Start new chunk with overlap from previous
                    if chunk_overlap > 0 and chunks:
                        overlap_text = chunks[-1][-chunk_overlap:]
                        current_chunk = overlap_text + separator + part
                    else:
                        current_chunk = part

            # Add the last chunk
            if current_chunk:
                chunks.append(current_chunk.strip())

            # Check if we successfully split
            if len(chunks) > 1:
                return chunks

        # Fallback: force split by character count
        logger.warning("Forcing character-level split, no suitable separator found")
        return self._force_split(text, chunk_size, chunk_overlap)

    def _force_split(self, text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
        """Force split text by character count.

        Args:
            text: The text to split.
            chunk_size: Maximum chunk size.
            chunk_overlap: Overlap between chunks.

        Returns:
            A list of text chunks split by character count.
        """
        chunks: list[str] = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)

            # Move start position, accounting for overlap
            start = end - chunk_overlap

        return chunks

    @property
    def max_chunk_size(self) -> int:
        """Return the maximum chunk size."""
        return self._chunk_size
