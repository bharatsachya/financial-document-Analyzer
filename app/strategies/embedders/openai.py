"""OpenAI-based text embedder.

Uses OpenAI's embedding API to generate high-quality text embeddings.
"""

import logging
from collections.abc import Awaitable

from openai import AsyncOpenAI, OpenAIError

from app.interfaces.embedder import BaseEmbedder

logger = logging.getLogger(__name__)


class OpenAIEmbedder(BaseEmbedder):
    """Embedder implementation using OpenAI's embedding API.

    Supports various OpenAI embedding models including:
    - text-embedding-3-small (1536 dimensions)
    - text-embedding-3-large (3072 dimensions)
    - text-embedding-ada-002 (1536 dimensions)

    Attributes:
        client: The async OpenAI client instance.
        model: The embedding model to use.
        dimension: The dimension of the embedding vectors.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: str | None = None,
        organization: str | None = None,
    ) -> None:
        """Initialize the OpenAI embedder.

        Args:
            api_key: Your OpenAI API key.
            model: The embedding model to use.
            base_url: Optional custom base URL for the API.
            organization: Optional OpenAI organization ID.
        """
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
        )
        self._model = model

        # Map models to their dimensions
        self._model_dimensions: dict[str, int] = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }

        if model not in self._model_dimensions:
            logger.warning(
                f"Unknown model '{model}', defaulting to dimension 1536. "
                f"Known models: {list(self._model_dimensions.keys())}"
            )
            self._dimension = 1536
        else:
            self._dimension = self._model_dimensions[model]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: A list of strings to embed.

        Returns:
            A list of embedding vectors.

        Raises:
            OpenAIError: If the API call fails.
            ValueError: If texts list is empty.
        """
        if not texts:
            return []

        logger.debug(f"Generating embeddings for {len(texts)} text(s) using {self._model}")

        try:
            response = await self._client.embeddings.create(
                input=texts,
                model=self._model,
            )

            embeddings = [item.embedding for item in response.data]
            logger.info(f"Successfully generated {len(embeddings)} embeddings")

            return embeddings

        except OpenAIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during embedding: {e}")
            raise RuntimeError(f"Embedding failed: {e}") from e

    async def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: The string to embed.

        Returns:
            An embedding vector.

        Raises:
            OpenAIError: If the API call fails.
            ValueError: If text is empty.
        """
        if not text.strip():
            raise ValueError("Cannot embed empty text")

        embeddings = await self.embed([text])
        return embeddings[0]

    @property
    def dimension(self) -> int:
        """Return the dimension of the embedding vectors."""
        return self._dimension

    @property
    def model(self) -> str:
        """Return the model name."""
        return self._model
