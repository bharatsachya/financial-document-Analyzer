import asyncio
import asyncpg
import uuid

async def main():
    conn = await asyncpg.connect('postgresql://docuser:docpass@localhost:5432/docplatform')
    try:
        # Check if the column exists
        exists = await conn.fetchval(
            "SELECT 1 FROM information_schema.columns WHERE table_name='procedural_memory' AND column_name='org_id';"
        )
        if not exists:
            # Add column
            print("Adding org_id column...")
            await conn.execute("ALTER TABLE procedural_memory ADD COLUMN org_id UUID;")
            
            # Update existing rows to the default org_id (bf0d03fb-d8ea-4377-a991-b3b5818e71ec)
            print("Setting default org_id for existing rows...")
            await conn.execute("UPDATE procedural_memory SET org_id = 'bf0d03fb-d8ea-4377-a991-b3b5818e71ec';")
            
            # Set NOT NULL constraint
            print("Setting NOT NULL constraint...")
            await conn.execute("ALTER TABLE procedural_memory ALTER COLUMN org_id SET NOT NULL;")
            
            # Add foreign key 
            try:
                print("Adding Foreign Key constraint...")
                await conn.execute("ALTER TABLE procedural_memory ADD CONSTRAINT fk_procedural_memory_org_id FOREIGN KEY (org_id) REFERENCES organizations(id) ON DELETE CASCADE;")
            except Exception as e:
                print(f"Failed to add foreign key, it might already exist or organizations table is missing: {e}")
                
            print("Successfully migrated procedural_memory table!")
        else:
            print("Column org_id already exists.")
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(main())
