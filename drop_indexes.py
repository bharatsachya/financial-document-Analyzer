import asyncio
import asyncpg

async def main():
    conn = await asyncpg.connect('postgresql://docuser:docpass@localhost:5432/docplatform')
    try:
        print("Dropping leftover indexes...")
        indexes = [
            "ix_procedural_memory_adviser_id",
            "ix_procedural_memory_memory_type",
            "ix_procedural_memory_org_id",
            "idx_procedural_adviser_memory_type",
            "idx_procedural_adviser_org"
        ]
        for idx in indexes:
            try:
                await conn.execute(f"DROP INDEX IF EXISTS {idx} CASCADE;")
                print(f"Dropped {idx}")
            except Exception as e:
                print(f"Failed to drop {idx}: {e}")
                
        print("Indexes dropped.")
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(main())
