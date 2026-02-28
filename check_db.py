import asyncio
import asyncpg

async def main():
    conn = await asyncpg.connect('postgresql://docuser:docpass@localhost:5432/docplatform')
    try:
        # Check if index exists
        print("Checking for indexes...")
        indexes = await conn.fetch("SELECT indexname, tablename FROM pg_indexes WHERE indexname LIKE '%procedural_memory%';")
        for idx in indexes:
            print(f"Found index {idx['indexname']} on table {idx['tablename']}")
            
        print("Checking for tables...")
        tables = await conn.fetch("SELECT tablename FROM pg_tables WHERE tablename LIKE '%procedural_memory%';")
        for tbl in tables:
            print(f"Found table {tbl['tablename']}")

        print("\nAttempting to drop everything clean...")
        await conn.execute("DROP TABLE IF EXISTS procedural_memory CASCADE;")
        for idx in indexes:
            try:
                await conn.execute(f"DROP INDEX IF EXISTS {idx['indexname']} CASCADE;")
                print(f"Dropped index {idx['indexname']}")
            except Exception as e:
                print(f"Could not drop {idx['indexname']}: {e}")
                
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(main())
