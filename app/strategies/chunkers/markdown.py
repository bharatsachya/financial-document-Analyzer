"""Markdown header-based chunker.

Splits documents based on markdown heading hierarchy (H1, H2, etc.).
Preserves document structure and context.
"""

import logging
import re
from typing import Literal

from app.interfaces.chunker import BaseChunker, Chunk

logger = logging.getLogger(__name__)


class MarkdownHeaderChunker(BaseChunker):
    """Chunker that splits text by markdown headers.

    This chunker respects the document hierarchy, creating chunks
    based on heading levels (H1, H2, H3, etc.). Each chunk contains
    a section along with its subsections up to a specified depth.

    Attributes:
        max_header_level: The maximum header level to split on (1-6).
        min_chunk_size: Minimum characters for a valid chunk.
        merge_small_chunks: Whether to merge small chunks with siblings.
    """

    def __init__(
        self,
        max_header_level: int = 2,
        min_chunk_size: int = 100,
        merge_small_chunks: bool = True,
    ) -> None:
        """Initialize the markdown header chunker.

        Args:
            max_header_level: Maximum header level to split on (1=H1, 2=H2, etc.).
            min_chunk_size: Minimum characters for a chunk to be valid.
            merge_small_chunks: If True, merge small chunks with adjacent sections.
        """
        self._max_header_level = max(1, min(6, max_header_level))
        self._min_chunk_size = min_chunk_size
        self._merge_small_chunks = merge_small_chunks

        # Build regex pattern for headers at or below the max level
        header_pattern = "|".join(f"{'#' * i} " for i in range(1, self._max_header_level + 1))
        self._header_regex = re.compile(rf"^(?:{header_pattern})(.+)$", re.MULTILINE)

    def chunk(self, text: str, **kwargs: object) -> list[Chunk]:
        """Split markdown text by headers.

        Args:
            text: The markdown text to chunk.
            **kwargs: Additional parameters (unused).

        Returns:
            A list of Chunks with header information in metadata.
        """
        if not text.strip():
            return []

        logger.debug(f"Chunking {len(text)} characters by markdown headers")

        chunks: list[Chunk] = []

        # Find all header positions
        matches = list(self._header_regex.finditer(text))
        if not matches:
            # No headers found, return entire text as single chunk
            return [
                Chunk(
                    content=text.strip(),
                    metadata={
                        "chunker": "markdown",
                        "header_level": 0,
                        "section": "root",
                    },
                )
            ]

        # Create chunks for each section
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

            header_text = match.group(1).strip()
            header_level = len(match.group(0).split()) - 1
            section_content = text[start:end].strip()

            # Skip if below minimum size
            if len(section_content) < self._min_chunk_size:
                continue

            chunks.append(
                Chunk(
                    content=section_content,
                    metadata={
                        "chunker": "markdown",
                        "header_level": header_level,
                        "section": header_text,
                        "char_count": len(section_content),
                    },
                )
            )

        # Handle merging of small chunks if enabled
        if self._merge_small_chunks:
            chunks = self._merge_chunks(chunks)

        logger.info(f"Created {len(chunks)} chunks from markdown text")
        return chunks

    def _merge_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Merge small chunks with their siblings.

        Args:
            chunks: The list of chunks to potentially merge.

        Returns:
            A list of chunks after merging.
        """
        if len(chunks) <= 1:
            return chunks

        merged: list[Chunk] = []
        i = 0

        while i < len(chunks):
            current = chunks[i]

            # If current chunk is too small, merge with next
            if (
                len(current.content) < self._min_chunk_size
                and i + 1 < len(chunks)
            ):
                next_chunk = chunks[i + 1]
                merged_content = current.content + "\n\n" + next_chunk.content

                merged.append(
                    Chunk(
                        content=merged_content,
                        metadata={
                            **current.metadata,
                            "merged": True,
                            "char_count": len(merged_content),
                        },
                    )
                )
                i += 2
            else:
                merged.append(current)
                i += 1

        return merged

    @property
    def max_chunk_size(self) -> int:
        """Return the maximum chunk size.

        Note: For markdown chunking, this is approximate as chunks
        are based on document structure rather than fixed size.
        """
        return 4096  # Approximate maximum
