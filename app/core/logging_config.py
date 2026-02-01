"""Centralized logging configuration for the application.

Provides structured logging with separate files for:
- info.log: General application logs (INFO level and above)
- error.log: Error logs only (ERROR level and above)
"""

import logging
import sys
from pathlib import Path

from app.core.config import get_settings


def setup_logging() -> logging.Logger:
    """Configure application logging with file and console handlers.

    Creates separate log files for info and error levels.
    Logs are written to the logs/ directory in the project root.

    Returns:
        The configured root logger.
    """
    settings = get_settings()

    # Create logs directory
    log_dir = Path(__file__).parent.parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)

    info_log_path = log_dir / "info.log"
    error_log_path = log_dir / "error.log"

    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    simple_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if settings.log_level == "DEBUG" else logging.INFO)

    # Clear existing handlers
    root_logger.handlers.clear()

    # File handler for INFO and above (info.log)
    info_handler = logging.FileHandler(info_log_path, encoding="utf-8")
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(info_handler)

    # File handler for ERROR and above (error.log)
    error_handler = logging.FileHandler(error_log_path, encoding="utf-8")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(error_handler)

    # Console handler for development
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if settings.log_level == "DEBUG" else logging.INFO)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name.

    Args:
        name: The name for the logger (typically __name__ of the module).

    Returns:
        A configured logger instance.
    """
    return logging.getLogger(name)


# Initialize logging on import
setup_logging()
