"""Initialize the Qdrant org_style_preferences collection.

Run this once to set up the vector collection for storing
learned stylistic rules from adviser edits.

Usage:
    python -m scripts.init_qdrant_collection
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.qdrant_client import QdrantStyleMemory

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


async def main() -> None:
    """Create the org_style_preferences collection in Qdrant."""
    logger.info("Initializing Qdrant style preferences collection...")

    memory = QdrantStyleMemory()
    await memory.ensure_collection()

    logger.info("✅ Qdrant collection ready!")


if __name__ == "__main__":
    asyncio.run(main())
