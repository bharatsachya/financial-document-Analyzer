"""Component Factory for strategy instantiation.

The Factory Pattern allows the application to instantiate
different strategy implementations at runtime based on
configuration or environment variables.
"""

import logging

from app.core.config import Settings, get_settings
from app.interfaces.chunker import BaseChunker
from app.interfaces.embedder import BaseEmbedder
from app.interfaces.parser import BaseParser
from app.interfaces.template import BaseTemplateAnalyzer, BaseTemplateInjector
from app.interfaces.vector_store import BaseVectorStore
from app.strategies.chunkers import MarkdownHeaderChunker, RecursiveTokenChunker
from app.strategies.embedders import OpenAIEmbedder
from app.strategies.parsers import LlamaParseParser, SimpleTextParser
from app.strategies.template_engine import TemplateAnalyzer, TemplateInjector
from app.strategies.vector_stores import QdrantStore

logger = logging.getLogger(__name__)


class ComponentFactory:
    """Factory for creating component instances based on configuration.

    This factory implements the Factory Pattern, allowing strategy
    selection at runtime without modifying core application code.

    Example:
        ```python
        settings = get_settings()
        factory = ComponentFactory(settings)

        parser = factory.get_parser()
        chunker = factory.get_chunker()
        embedder = factory.get_embedder()
        vector_store = factory.get_vector_store()
        ```
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the factory with optional settings.

        Args:
            settings: Application settings. If None, uses global settings.
        """
        self._settings = settings or get_settings()
        self._parser_cache: BaseParser | None = None
        self._chunker_cache: BaseChunker | None = None
        self._embedder_cache: BaseEmbedder | None = None
        self._vector_store_cache: BaseVectorStore | None = None
        self._template_analyzer_cache: BaseTemplateAnalyzer | None = None
        self._template_injector_cache: BaseTemplateInjector | None = None

    def get_parser(self, parser_type: str | None = None) -> BaseParser:
        """Get a parser instance based on the specified type.

        Args:
            parser_type: The parser type to instantiate. If None, uses settings.

        Returns:
            A BaseParser implementation instance.

        Raises:
            ValueError: If the parser type is unknown.
        """
        if self._parser_cache is None or parser_type is not None:
            parser_type = parser_type or self._settings.parser_type

            logger.info(f"Instantiating parser: {parser_type}")

            match parser_type:
                case "llama_parse":
                    if not self._settings.llama_parse_api_key:
                        raise ValueError("LLAMA_PARSE_API_KEY is required for LlamaParse")
                    self._parser_cache = LlamaParseParser(
                        api_key=self._settings.llama_parse_api_key,
                        result_type="markdown",
                    )
                case "simple":
                    self._parser_cache = SimpleTextParser()
                case _:
                    raise ValueError(
                        f"Unknown parser type: {parser_type}. "
                        f"Valid options: 'llama_parse', 'simple'"
                    )

        return self._parser_cache

    def get_chunker(
        self,
        chunker_type: str | None = None,
        chunk_size: int = 1024,
        chunk_overlap: int = 128,
    ) -> BaseChunker:
        """Get a chunker instance based on the specified type.

        Args:
            chunker_type: The chunker type to instantiate. If None, uses settings.
            chunk_size: Maximum chunk size for recursive chunker.
            chunk_overlap: Overlap between chunks for recursive chunker.

        Returns:
            A BaseChunker implementation instance.

        Raises:
            ValueError: If the chunker type is unknown.
        """
        if self._chunker_cache is None or chunker_type is not None:
            chunker_type = chunker_type or self._settings.chunker_type

            logger.info(f"Instantiating chunker: {chunker_type}")

            match chunker_type:
                case "markdown":
                    self._chunker_cache = MarkdownHeaderChunker(
                        max_header_level=2,
                        min_chunk_size=100,
                        merge_small_chunks=True,
                    )
                case "recursive":
                    self._chunker_cache = RecursiveTokenChunker(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                case _:
                    raise ValueError(
                        f"Unknown chunker type: {chunker_type}. "
                        f"Valid options: 'markdown', 'recursive'"
                    )

        return self._chunker_cache

    def get_embedder(self, embedder_type: str | None = None) -> BaseEmbedder:
        """Get an embedder instance based on the specified type.

        Args:
            embedder_type: The embedder type to instantiate. If None, uses settings.

        Returns:
            A BaseEmbedder implementation instance.

        Raises:
            ValueError: If the embedder type is unknown.
        """
        if self._embedder_cache is None or embedder_type is not None:
            embedder_type = embedder_type or self._settings.embedder_type

            logger.info(f"Instantiating embedder: {embedder_type}")

            match embedder_type:
                case "openai":
                    if not self._settings.openrouter_api_key:
                        raise ValueError("OPENROUTER_API_KEY is required for embeddings")
                    self._embedder_cache = OpenAIEmbedder(
                        api_key=self._settings.openrouter_api_key,
                        model=self._settings.llm_embedding_model,
                        base_url=self._settings.openai_base_url,
                        organization=None,
                    )
                case _:
                    raise ValueError(
                        f"Unknown embedder type: {embedder_type}. "
                        f"Valid options: 'openai'"
                    )

        return self._embedder_cache

    def get_vector_store(
        self, vector_store_type: str | None = None
    ) -> BaseVectorStore:
        """Get a vector store instance based on the specified type.

        Args:
            vector_store_type: The vector store type to instantiate. If None, uses settings.

        Returns:
            A BaseVectorStore implementation instance.

        Raises:
            ValueError: If the vector store type is unknown.
        """
        if self._vector_store_cache is None or vector_store_type is not None:
            vector_store_type = vector_store_type or self._settings.vector_store_type

            logger.info(f"Instantiating vector store: {vector_store_type}")

            match vector_store_type:
                case "qdrant":
                    self._vector_store_cache = QdrantStore(
                        url=self._settings.qdrant_url,
                        api_key=self._settings.qdrant_api_key,
                        distance_metric="cosine",
                    )
                case _:
                    raise ValueError(
                        f"Unknown vector store type: {vector_store_type}. "
                        f"Valid options: 'qdrant'"
                    )

        return self._vector_store_cache

    def get_template_analyzer(self, custom_prompt: str | None = None) -> BaseTemplateAnalyzer:
        """Get a template analyzer instance.

        Args:
            custom_prompt: Optional custom prompt for LLM analysis.
                When provided, bypasses cache to ensure prompt isolation.

        Returns:
            A BaseTemplateAnalyzer implementation instance.

        Raises:
            ValueError: If template analyzer configuration is invalid.
        """
        # Bypass cache when custom prompt is provided for prompt isolation
        if custom_prompt is not None:
            logger.info("Instantiating template analyzer with custom prompt")
            return TemplateAnalyzer(
                use_llm=self._settings.use_llm_for_templates,
                openai_api_key=self._settings.openrouter_api_key,
                base_url=self._settings.openai_base_url,
                model=self._settings.llm_chat_model,
                custom_prompt=custom_prompt,
            )

        if self._template_analyzer_cache is None:
            logger.info("Instantiating template analyzer")

            self._template_analyzer_cache = TemplateAnalyzer(
                use_llm=self._settings.use_llm_for_templates,
                openai_api_key=self._settings.openrouter_api_key,
                base_url=self._settings.openai_base_url,
                model=self._settings.llm_chat_model,
            )

        return self._template_analyzer_cache

    def get_template_injector(self) -> BaseTemplateInjector:
        """Get a template injector instance.

        Returns:
            A BaseTemplateInjector implementation instance.
        """
        if self._template_injector_cache is None:
            logger.info("Instantiating template injector")

            self._template_injector_cache = TemplateInjector()

        return self._template_injector_cache

    def clear_cache(self) -> None:
        """Clear all cached component instances.

        This forces new instances to be created on next access.
        Useful for testing or when settings change.
        """
        self._parser_cache = None
        self._chunker_cache = None
        self._embedder_cache = None
        self._vector_store_cache = None
        self._template_analyzer_cache = None
        self._template_injector_cache = None
        logger.debug("Component factory cache cleared")


# Global factory instance
_factory: ComponentFactory | None = None


def get_factory() -> ComponentFactory:
    """Get or create the global ComponentFactory instance.

    Returns:
        The singleton ComponentFactory instance.
    """
    global _factory
    if _factory is None:
        _factory = ComponentFactory()
    return _factory
