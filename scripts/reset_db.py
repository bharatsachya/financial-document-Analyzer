"""Database reset script.

Run this script to drop all tables and reinitialize the database.
This will delete all data and recreate tables with the correct organization ID.

Usage:
    poetry run python -m scripts.reset_db
    or
    poetry run python scripts/reset_db.py (after pip install -e .)
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.core.config import get_settings
from app.db.session import drop_all_tables, init_db


async def main() -> None:
    """Reset the database by dropping all tables and reinitializing."""
    settings = get_settings()
    
    print("Dropping all database tables...")
    await drop_all_tables(settings)
    print("All tables dropped successfully!")
    
    print("Initializing database with correct organization ID...")
    await init_db(settings)
    print("Database reinitialized successfully!")


if __name__ == "__main__":
    asyncio.run(main())
