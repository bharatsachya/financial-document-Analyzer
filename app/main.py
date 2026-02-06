"""FastAPI application entry point.

Main application setup with middleware, routing, and lifecycle management.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.templates import router as templates_router
from app.api.schemas import ErrorResponse
from app.core.config import Settings, get_settings
from app.core.logging_config import setup_logging
from app.db.session import close_db, create_all_tables, init_db

# Initialize logging before importing other modules
setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.

    Handles startup and shutdown events for proper resource management.
    """
    settings: Settings = app.state.settings

    # Startup
    logger.info("Starting Template Intelligence Engine API...")

    # Initialize database
    try:
        logger.info("Initializing database...")
        await init_db(settings)
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}", exc_info=True)
        raise

    yield

    # Shutdown
    logger.info("Shutting down Template Intelligence Engine API...")

    # Close database connections
    try:
        await close_db(settings)
        logger.info("Database connections closed")
    except Exception as e:
        logger.error(f"Error closing database: {e}", exc_info=True)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        settings: Optional settings. If None, loads from environment.

    Returns:
        Configured FastAPI application instance.
    """
    try:
        settings = settings or get_settings()

        app = FastAPI(
            title="Template Intelligence Engine",
            description="Template analysis and variable injection platform",
            version="0.1.0",
            lifespan=lifespan,
            docs_url="/docs",
            redoc_url="/redoc",
        )

        # Store settings in app state
        app.state.settings = settings

        # CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Include routers
        try:
            app.include_router(templates_router)
            logger.info("Registered templates router")
        except Exception as e:
            logger.error(f"Failed to include router: {e}", exc_info=True)
            raise

        # Health check endpoint
        @app.get("/health", tags=["health"])
        async def health_check():
            """Health check endpoint for load balancers and monitoring."""
            return {
                "status": "healthy",
                "service": "template-intelligence-api",
                "version": "0.1.0",
            }

        # Exception handlers
        @app.exception_handler(RequestValidationError)
        async def validation_exception_handler(request, exc):
            """Handle Pydantic validation errors."""
            logger.warning(f"Validation error: {exc.errors()}")
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={
                    "detail": "Validation error",
                    "errors": exc.errors(),
                },
            )

        @app.exception_handler(Exception)
        async def general_exception_handler(request, exc):
            """Handle uncaught exceptions."""
            logger.error(f"Unhandled exception: {exc}", exc_info=True)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=ErrorResponse(
                    detail="Internal server error",
                    error_code="INTERNAL_ERROR",
                ).model_dump(),
            )

        logger.info("FastAPI application created successfully")
        return app

    except Exception as e:
        logger.error(f"Failed to create FastAPI app: {e}", exc_info=True)
        raise


# Create the app instance
try:
    app = create_app()
except Exception as e:
    logger.error(f"Fatal error creating app: {e}", exc_info=True)
    raise


if __name__ == "__main__":
    import uvicorn

    try:
        settings = get_settings()
        logger.info(f"Starting uvicorn server on port 8000...")
        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level=settings.log_level.lower(),
        )
    except Exception as e:
        logger.error(f"Failed to start uvicorn: {e}", exc_info=True)
        raise
