"""Database initialization script.

Run this script to create database tables and seed initial data.

Usage:
    python -m scripts.init_db
    or
    python scripts/init_db.py (after pip install -e .)
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.core.config import get_settings
from app.db.session import init_db


async def main() -> None:
    """Initialize the database."""
    settings = get_settings()
    await init_db(settings)
    print("Database initialized successfully!")


if __name__ == "__main__":
    asyncio.run(main())
