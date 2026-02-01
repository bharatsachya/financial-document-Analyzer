"""Database session management for async PostgreSQL.

Provides async session creation and dependency injection for FastAPI.
"""

import logging
from collections.abc import AsyncGenerator

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlmodel import SQLModel, select

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


# Global engine and session maker
_engine: AsyncEngine | None = None
_async_session_maker: async_sessionmaker[AsyncSession] | None = None


def get_engine(settings: Settings | None = None) -> AsyncEngine:
    """Get or create the async database engine.

    Args:
        settings: Optional settings. If None, uses global settings.

    Returns:
        The async SQLAlchemy engine.
    """
    global _engine

    try:
        if _engine is None:
            settings = settings or get_settings()

            logger.info(f"Creating async database engine: {settings.database_url}")

            _engine = create_async_engine(
                settings.database_url,
                echo=settings.log_level == "DEBUG",
                future=True,
                pool_pre_ping=True,
                pool_size=10,
                max_overflow=20,
            )

            logger.info("Database engine created successfully")

        return _engine

    except Exception as e:
        logger.error(f"Failed to create database engine: {e}", exc_info=True)
        raise


def get_session_maker(settings: Settings | None = None) -> async_sessionmaker[AsyncSession]:
    """Get or create the async session maker.

    Args:
        settings: Optional settings. If None, uses global settings.

    Returns:
        The async session maker.
    """
    global _async_session_maker

    try:
        if _async_session_maker is None:
            engine = get_engine(settings)

            _async_session_maker = async_sessionmaker(
                engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autocommit=False,
                autoflush=False,
            )

            logger.info("Session maker created successfully")

        return _async_session_maker

    except Exception as e:
        logger.error(f"Failed to create session maker: {e}", exc_info=True)
        raise


async def get_async_session(settings: Settings | None = None) -> AsyncGenerator[AsyncSession, None]:
    """Dependency injection for FastAPI to get async database sessions.

    Args:
        settings: Optional settings. If None, uses global settings.

    Yields:
        An async database session.

    Example:
        ```python
        @app.get("/users/{user_id}")
        async def get_user(user_id: str, session: AsyncSession = Depends(get_async_session)):
            result = await session.execute(select(User).where(User.id == user_id))
            return result.scalar_one_or_none()
        ```
    """
    session_maker = get_session_maker(settings)
    async with session_maker() as session:
        try:
            yield session
        except SQLAlchemyError as e:
            logger.error(f"Database error: {e}", exc_info=True)
            await session.rollback()
            raise
        except Exception as e:
            logger.error(f"Unexpected error in session: {e}", exc_info=True)
            await session.rollback()
            raise
        finally:
            try:
                await session.close()
            except Exception as e:
                logger.error(f"Error closing session: {e}", exc_info=True)


async def create_all_tables(settings: Settings | None = None) -> None:
    """Create all database tables.

    This uses SQLAlchemy's metadata.create_all for table creation.
    In production, consider using Alembic for migrations.

    Args:
        settings: Optional settings. If None, uses global settings.
    """
    try:
        # Import models to ensure they're registered with SQLModel metadata
        from app.db import models  # noqa: F401

        engine = get_engine(settings)

        logger.info("Creating database tables...")

        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

        logger.info("Database tables created successfully")

    except Exception as e:
        logger.error(f"Failed to create database tables: {e}", exc_info=True)
        raise


async def drop_all_tables(settings: Settings | None = None) -> None:
    """Drop all database tables.

    WARNING: This will delete all data. Use with caution,
    typically only in test environments.

    Args:
        settings: Optional settings. If None, uses global settings.
    """
    try:
        engine = get_engine(settings)

        logger.warning("Dropping all database tables...")

        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.drop_all)

        logger.warning("All database tables dropped")

    except Exception as e:
        logger.error(f"Failed to drop database tables: {e}", exc_info=True)
        raise


async def init_db(settings: Settings | None = None) -> None:
    """Initialize the database with default data.

    Creates tables and seeds initial data for development.

    Args:
        settings: Optional settings. If None, uses global settings.
    """
    try:
        # Import here to avoid circular dependency
        from app.db.models import Organization, User

        await create_all_tables(settings)

        # Check if we need to seed initial data
        session_maker = get_session_maker(settings)
        async with session_maker() as session:
            try:
                # Check if any organizations exist
                result = await session.execute(select(Organization).limit(1))
                org_exists = result.scalar_one_or_none() is not None

                if not org_exists:
                    logger.info("Seeding initial organization...")

                    # Create a default organization with a specific ID for consistency
                    import uuid
                    default_org = Organization(
                        id=uuid.UUID("bf0d03fb-d8ea-4377-a991-b3b5818e71ec"),
                        name="Default Organization",
                        slug="default",
                    )
                    session.add(default_org)
                    await session.commit()

                    logger.info(f"Created default organization: {default_org.id}")

                    # Create a default user for testing
                    logger.info("Seeding initial user...")
                    import uuid

                    default_user = User(
                        id=uuid.UUID("3c5ff1b3-b0d6-4dba-b254-a2be667bbd52"),
                        email="test@example.com",
                        full_name="Test User",
                        hashed_password="$2b$12$test",  # Placeholder, not for production
                        organization_id=default_org.id,
                        is_active=True,
                    )
                    session.add(default_user)
                    await session.commit()

                    logger.info(f"Created default user: {default_user.id}")

                await session.commit()

            except Exception as e:
                logger.error(f"Error seeding initial data: {e}", exc_info=True)
                await session.rollback()
                raise

    except Exception as e:
        logger.error(f"Failed to initialize database: {e}", exc_info=True)
        raise


async def close_db(settings: Settings | None = None) -> None:
    """Close the database engine and all connections.

    Args:
        settings: Optional settings. If None, uses global settings.
    """
    global _engine, _async_session_maker

    try:
        if _engine is not None:
            logger.info("Closing database engine...")
            await _engine.dispose()
            _engine = None
            _async_session_maker = None
            logger.info("Database engine closed")

    except Exception as e:
        logger.error(f"Error closing database engine: {e}", exc_info=True)
        raise
