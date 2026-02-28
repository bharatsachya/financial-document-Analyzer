import asyncio
import asyncpg

async def main():
    conn = await asyncpg.connect('postgresql://docuser:docpass@localhost:5432/docplatform')
    try:
        print("Dropping old procedural_memory table...")
        await conn.execute("DROP TABLE IF EXISTS procedural_memory CASCADE;")
        print("Table dropped. Restart the api server to auto-recreate it via create_all_tables.")
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(main())
