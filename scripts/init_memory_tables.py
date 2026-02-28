"""Initialize Multi-Tier Memory database tables.

Creates tables for Procedural Memory (logic/corrections),
Episodic Memory (event logging), and Feedback Events.

Usage:
    python -m scripts.init_memory_tables
    or
    python scripts/init_memory_tables.py
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.core.config import get_settings
from app.db.session import get_engine, get_session_maker


async def main() -> None:
    """Initialize the multi-tier memory database tables."""
    settings = get_settings()

    print("Creating multi-tier memory tables...")

    # Import the database models to register them with SQLModel metadata
    from app.db.database import (
        EpisodicEvent,
        EpisodicMemory,
        EventFeedbackEvent,
        FeedbackEvent,
        MemoryType,
        ProceduralMemory,
        VariableCorrection,
    )

    engine = get_engine(settings)

    print("Creating tables (this will create new tables and skip existing ones)...")
    async with engine.begin() as conn:
        await conn.run_sync(
            lambda sync_conn: EpisodicEvent.metadata.create_all(
                sync_conn, checkfirst=True
            )
        )

    print("Multi-tier memory tables created successfully!")
    print("\nTables created:")
    print("  - procedural_memory: Stores learned logic and corrections")
    print("  - variable_corrections: Tracks variable value corrections")
    print("  - episodic_events: Logs events for temporal context")
    print("  - feedback_events: Detailed feedback capture for learning")


if __name__ == "__main__":
    asyncio.run(main())
