"""Abstract base classes for document processing strategies."""

from app.interfaces.chunker import BaseChunker
from app.interfaces.embedder import BaseEmbedder
from app.interfaces.parser import BaseParser
from app.interfaces.template import BaseTemplateAnalyzer, BaseTemplateInjector, TemplateDocument
from app.interfaces.vector_store import BaseVectorStore

__all__ = [
    "BaseParser",
    "BaseChunker",
    "BaseEmbedder",
    "BaseVectorStore",
    "BaseTemplateAnalyzer",
    "BaseTemplateInjector",
    "TemplateDocument",
]
