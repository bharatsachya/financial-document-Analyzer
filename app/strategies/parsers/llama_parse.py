"""LlamaParse-based document parser.

Uses LlamaParse for high-quality document extraction from PDFs, DOCX, and other formats.
"""

import logging
from pathlib import Path

from app.interfaces.parser import BaseParser, Document

logger = logging.getLogger(__name__)


class LlamaParseParser(BaseParser):
    """Parser implementation using LlamaParse API.

    This parser provides high-quality extraction from complex documents
    including PDFs with tables, headers, and multi-column layouts.

    Attributes:
        api_key: The LlamaParse API key.
        result_type: The format of results ("markdown" or "text").
        language: Language hint for better OCR accuracy.
    """

    def __init__(
        self,
        api_key: str,
        result_type: str = "markdown",
        language: str = "en",
        parsing_instruction: str | None = None,
    ) -> None:
        """Initialize the LlamaParse parser.

        Args:
            api_key: Your LlamaParse API key from https://cloud.llamaindex.ai/
            result_type: Output format - "markdown" or "text".
            language: ISO language code for better OCR.
            parsing_instruction: Optional custom parsing instructions.
        """
        self._api_key = api_key
        self._result_type = result_type
        self._language = language
        self._parsing_instruction = parsing_instruction
        self._client = None

    def _get_client(self):
        """Lazy-initialize the LlamaParse client.

        Returns:
            The initialized LlamaParse client instance.
        """
        if self._client is None:
            try:
                from llama_parse import LlamaParse

                self._client = LlamaParse(
                    api_key=self._api_key,
                    result_type=self._result_type,
                    language=self._language,
                    parsing_instruction=self._parsing_instruction,
                    verbose=False,
                )
            except ImportError as e:
                raise ImportError(
                    "llama-parse is not installed. "
                    "Install it with: pip install llama-parse"
                ) from e
        return self._client

    async def aload_data(self, file_path: str) -> list[Document]:
        """Parse a document using LlamaParse.

        Args:
            file_path: Path to the document file.

        Returns:
            A list of Document objects with parsed content and metadata.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            RuntimeError: If parsing fails.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Parsing file: {file_path}")

        try:
            client = self._get_client()

            # Use the async load_data method
            documents = await client.aload_data(file_path)

            result: list[Document] = []
            for doc in documents:
                result.append(
                    Document(
                        content=doc.text,
                        metadata={
                            **(doc.metadata or {}),
                            "parser": "llama_parse",
                            "source_file": path.name,
                        },
                        source=file_path,
                    )
                )

            logger.info(f"Successfully parsed {len(result)} document(s) from {file_path}")
            return result

        except FileNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {e}")
            raise RuntimeError(f"Parsing failed: {e}") from e

    @property
    def supported_extensions(self) -> set[str]:
        """Return supported file extensions."""
        return {
            ".pdf",
            ".docx",
            ".doc",
            ".pptx",
            ".ppt",
            ".xlsx",
            ".xls",
            ".txt",
            ".md",
        }
