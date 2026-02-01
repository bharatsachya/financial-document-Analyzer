"""Application configuration using Pydantic v2 Settings.

Centralized configuration that loads from environment variables
and provides type-safe access throughout the application.
"""

import logging
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    All settings have sensible defaults and can be overridden via
    environment variables or a .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://docuser:docpass@localhost:5432/docplatform",
        description="PostgreSQL connection URL with asyncpg driver.",
    )

    # Vector Store
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant server URL.",
    )
    qdrant_api_key: str | None = Field(
        default=None,
        description="Optional Qdrant API key for authentication.",
    )

    # Redis / Celery
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL for Celery broker and backend.",
    )
    celery_broker_url: str = Field(
        default="redis://localhost:6379/0",
        description="Celery message broker URL.",
    )
    celery_result_backend: str = Field(
        default="redis://localhost:6379/0",
        description="Celery result backend URL.",
    )

    # OpenAI
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key for embeddings.",
    )
    openai_base_url: str | None = Field(
        default=None,
        description="Optional custom OpenAI base URL.",
    )
    openai_organization: str | None = Field(
        default=None,
        description="Optional OpenAI organization ID.",
    )

    # LlamaParse
    llama_parse_api_key: str = Field(
        default="",
        description="LlamaParse API key for document parsing.",
    )

    # Strategy Selection
    parser_type: str = Field(
        default="llama_parse",
        description="Parser strategy to use: 'llama_parse' or 'simple'.",
    )
    chunker_type: str = Field(
        default="markdown",
        description="Chunker strategy to use: 'markdown' or 'recursive'.",
    )
    embedder_type: str = Field(
        default="openai",
        description="Embedder strategy to use: 'openai'.",
    )
    vector_store_type: str = Field(
        default="qdrant",
        description="Vector store strategy to use: 'qdrant'.",
    )

    # File Storage
    upload_dir: Path = Field(
        default=Path("./uploads"),
        description="Directory for uploaded files.",
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR.",
    )

    @field_validator("upload_dir")
    @classmethod
    def ensure_upload_dir(cls, v: Path) -> Path:
        """Ensure upload directory exists."""
        v.mkdir(parents=True, exist_ok=True)
        return v.resolve()

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, v: str) -> str:
        """Normalize log level to uppercase."""
        return v.upper()

    def configure_logging(self) -> None:
        """Configure global logging based on settings."""
        import structlog

        level = getattr(logging, self.log_level, logging.INFO)

        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

        logging.basicConfig(
            format="%(message)s",
            level=level,
        )

        logger.setLevel(level)


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get or create the global Settings instance.

    Returns:
        The singleton Settings instance.
    """
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.configure_logging()
    return _settings
