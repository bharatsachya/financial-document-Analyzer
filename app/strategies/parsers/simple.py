"""Simple text-based document parser.

A fallback parser for plain text files that doesn't require
external API calls.
"""

import logging
from pathlib import Path

from app.interfaces.parser import BaseParser, Document

logger = logging.getLogger(__name__)


class SimpleTextParser(BaseParser):
    """Simple parser for plain text and markdown files.

    This is a lightweight fallback parser that reads files as-is
    without any complex extraction logic.
    """

    def __init__(
        self,
        encoding: str = "utf-8",
    ) -> None:
        """Initialize the simple text parser.

        Args:
            encoding: The character encoding to use when reading files.
        """
        self._encoding = encoding

    async def aload_data(self, file_path: str) -> list[Document]:
        """Read a plain text document.

        Args:
            file_path: Path to the text file.

        Returns:
            A list containing a single Document with the file's content.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            UnicodeDecodeError: If the file encoding is incorrect.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Reading text file: {file_path}")

        try:
            async with open(path, "r", encoding=self._encoding) as f:  # noqa: ASYNC101
                content = await f.read()

            documents = [
                Document(
                    content=content,
                    metadata={
                        "parser": "simple_text",
                        "source_file": path.name,
                        "file_size": path.stat().st_size,
                    },
                    source=file_path,
                )
            ]

            logger.info(f"Successfully read {len(content)} characters from {file_path}")
            return documents

        except FileNotFoundError:
            raise
        except UnicodeDecodeError as e:
            logger.error(f"Encoding error reading {file_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            raise RuntimeError(f"Reading failed: {e}") from e

    @property
    def supported_extensions(self) -> set[str]:
        """Return supported file extensions."""
        return {".txt", ".md", ".csv", ".json", ".xml"}
