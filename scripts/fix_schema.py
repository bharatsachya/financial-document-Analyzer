import asyncio
import sys
from pathlib import Path
from sqlalchemy import text

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.session import get_engine
from app.core.config import get_settings

async def main():
    settings = get_settings()
    engine = get_engine(settings)
    
    print(f"Connecting to database: {settings.database_url}")
    
    async with engine.begin() as conn:
        print("Fixing database schema...")
        
        # 1. Inspect and Ensure Enum type exists
        print("Checking TemplateStatus enum...")
        result = await conn.execute(text("SELECT enumlabel FROM pg_enum JOIN pg_type ON pg_enum.enumtypid = pg_type.oid WHERE typname = 'templatestatus'"))
        existing_labels = [row[0] for row in result.fetchall()]
        print(f"Existing templatestatus labels: {existing_labels}")

        if not existing_labels:
            print("Creating templatestatus type...")
            await conn.execute(text("CREATE TYPE templatestatus AS ENUM ('queued', 'analyzing', 'finalizing', 'completed', 'failed');"))
        else:
             # Check if 'queued' is in labels
             if 'queued' not in existing_labels and 'QUEUED' in existing_labels:
                 print("WARNING: Enum uses uppercase. Adjusting default...")
                 default_val = "'QUEUED'"
             else:
                 default_val = "'queued'"

        # 2. Add status column if missing
        print("Checking 'status' column...")
        # Determine default value logic based on inspection result or fallback
        # If the loop above ran, we know what to use. If not (e.g. type just created), use 'queued'.
        
        use_default = "'queued'"
        if existing_labels and 'QUEUED' in existing_labels and 'queued' not in existing_labels:
             use_default = "'QUEUED'"
        
        await conn.execute(text(f"""
            ALTER TABLE templates 
            ADD COLUMN IF NOT EXISTS status templatestatus DEFAULT {use_default}::templatestatus;
        """))

        # 3. Add other missing columns
        columns = [
            ("injection_status", "VARCHAR(50)"),
            ("injection_task_id", "VARCHAR(255)"),
            ("injection_started_at", "TIMESTAMP WITH TIME ZONE"),
            ("injection_completed_at", "TIMESTAMP WITH TIME ZONE"),
            ("injection_error_message", "VARCHAR(2048)"),
            ("download_ready", "BOOLEAN DEFAULT FALSE"),
            ("download_url", "VARCHAR(1024)"),
            ("batch_id", "VARCHAR(255)"),
            ("previous_batch_id", "VARCHAR(255)"),
            ("batch_status", "VARCHAR(50)"),
            ("is_tagged", "BOOLEAN DEFAULT FALSE"),
            ("paragraph_count", "INTEGER DEFAULT 0"),
            ("detected_variables", "JSONB"),
            ("analysis_method", "VARCHAR(50) DEFAULT 'regex'"),
            ("source_document_id", "UUID"),
            ("task_id", "VARCHAR(255)"),
            ("processing_started_at", "TIMESTAMP WITH TIME ZONE"),
            ("processing_completed_at", "TIMESTAMP WITH TIME ZONE"),
            ("error_message", "VARCHAR(2048)")
        ]
        
        for col_name, col_type in columns:
            try:
                await conn.execute(text(f"""
                    ALTER TABLE templates 
                    ADD COLUMN IF NOT EXISTS {col_name} {col_type};
                """))
                print(f"Checked/Added column: {col_name}")
            except Exception as e:
                print(f"Error checking/adding column {col_name}: {e}")
        
        print("Schema fix complete!")

if __name__ == "__main__":
    asyncio.run(main())
