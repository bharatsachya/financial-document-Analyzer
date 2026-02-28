import asyncio
from sqlalchemy import select
from app.db.database import TemplateStorage
from app.api.deps import get_db
from app.core.config import get_settings

async def main():
    async for session in get_db():
        query = select(TemplateStorage)
        result = await session.execute(query)
        template = result.scalars().first()
        if template:
            print(f"Template ID: {template.id}")
            print(f"File Path: {template.file_path}")
            try:
                import docx
                doc = docx.Document(template.file_path)
                text = "\n".join([p.text for p in doc.paragraphs])
                print(f"Extracted {len(text)} characters of text.")
            except Exception as e:
                print(f"Error loading docx: {e}")
        else:
            print("No templates found")
        break

if __name__ == "__main__":
    asyncio.run(main())
