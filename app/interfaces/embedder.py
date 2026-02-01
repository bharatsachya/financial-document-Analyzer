"""Abstract base class for text embedding strategies.

The Strategy Pattern allows different embedding models
to be used interchangeably.
"""

from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """Abstract base class for text embedding strategies.

    All concrete embedder implementations must inherit from this class
    and implement the required methods.

    Example:
        ```python
        class OpenAIEmbedder(BaseEmbedder):
            async def embed(self, texts: list[str]) -> list[list[float]]:
                # Call OpenAI API
                pass

            @property
            def dimension(self) -> int:
                return 1536  # text-embedding-3-small
        ```
    """

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: A list of strings to generate embeddings for.

        Returns:
            A list of embedding vectors, where each vector is a list of floats.

        Raises:
            EmbeddingError: If the embedding generation fails.
        """
        ...

    @abstractmethod
    async def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: The string to generate an embedding for.

        Returns:
            An embedding vector as a list of floats.
        """
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimension of the embedding vectors.

        Returns:
            The number of dimensions in the embedding vectors.
        """
        ...
