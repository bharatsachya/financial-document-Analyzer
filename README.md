# Document Intelligence Platform

A high-performance, event-driven document ingestion pipeline designed for the UK Financial Advisory market. It processes unstructured financial documents (PDFs, Docx) into structured, queryable knowledge chunks in Qdrant.

## Architecture Overview

This system follows the **Strategy Pattern** and **Factory Pattern** for maximum extensibility.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Frontend (Streamlit)                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                   │
│  │ File Upload  │  │ Status View  │  │ Settings     │                   │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘                   │
│         │                  │                                               │
│         ▼                  ▼                                               │
│         APIClient (X-Org-ID header for multi-tenancy)                     │
└──────────────────────────────┬────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         FastAPI Backend                                  │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                          API Routers                              │   │
│  │  ┌──────────────┐  ┌──────────────────┐  ┌──────────────────┐    │   │
│  │  │   ingest.py  │  │ organizations.py │  │    users.py      │    │   │
│  │  │  /ingest/*   │  │  /organizations  │  │    /users        │    │   │
│  │  └──────┬───────┘  └──────────────────┘  └──────────────────┘    │   │
│  │         │                                                           │   │
│  │         ▼                                                           │   │
│  │  ┌─────────────────────────────────────────────────────────────┐  │   │
│  │  │              Upload Document Flow (Fire-and-Forget)         │  │   │
│  │  │  1. Validate file & organization                            │  │   │
│  │  │  2. Save to uploads/                                         │  │   │
│  │  │  3. Create Document record in PostgreSQL                     │  │   │
│  │  │  4. Queue task to Redis (Celery) → Return 202 Accepted       │  │   │
│  │  └─────────────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────┬────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Message Queue (Redis)                             │
│                    Celery Task Queue (Background Jobs)                   │
└──────────────────────────────┬────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       Celery Worker (Async Processing)                   │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    Document Processing Pipeline                   │   │
│  │                                                                   │   │
│  │  1. ┌─────────────┐   2. ┌─────────────┐   3. ┌─────────────┐   │   │
│  │     │   Parser    │  →   │  Chunker    │  →   │  Embedder   │   │   │
│  │     │ (LlamaParse)│      │ (Markdown)  │      │  (OpenAI)   │   │   │
│  │     └─────────────┘      └─────────────┘      └─────────────┘   │   │
│  │           │                      │                     │         │   │
│  │           ▼                      ▼                     ▼         │   │
│  │  ┌─────────────────────────────────────────────────────────────┐  │   │
│  │  │                    4. Vector Store                          │  │   │
│  │  │                     (Qdrant)                               │  │   │
│  │  │            - Upsert vectors with org_id                     │  │   │
│  │  │            - Update document status to COMPLETED            │  │   │
│  │  └─────────────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          Data Storage Layer                              │
│  ┌──────────────────────┐  ┌──────────────────────────────────────┐    │
│  │    PostgreSQL        │  │           Qdrant                      │    │
│  │  (Metadata & State)  │  │    (Vector Embeddings)                │    │
│  │                      │  │                                      │    │
│  │  - organizations     │  │  - Collections by org_id             │    │
│  │  - users             │  │  - Vectors with tenant isolation     │    │
│  │  - documents         │  │                                      │    │
│  │  - document_chunks   │  │                                      │    │
│  └──────────────────────┘  └──────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Architectural Patterns

**Strategy Pattern** (Plugin System)
```
app/interfaces/          # Abstract Base Classes
├── parser.py           → BaseParser
├── chunker.py          → BaseChunker
├── embedder.py         → BaseEmbedder
└── vector_store.py     → BaseVectorStore

app/strategies/         # Concrete Implementations
├── parsers/            → LlamaParseParser, SimpleParser
├── chunkers/           → MarkdownChunker, RecursiveChunker
└── vector_stores/      → QdrantVectorStore
```

**Factory Pattern** (Runtime Configuration)
```python
parser = ComponentFactory.get_parser("llama_parse")
chunker = ComponentFactory.get_chunker("markdown")
embedder = ComponentFactory.get_embedder("openai")
```

### Core Components

- **FastAPI**: Asynchronous REST API with fire-and-forget upload
- **PostgreSQL**: Metadata and state management (Async SQLAlchemy)
- **Qdrant**: Vector database for semantic search
- **Redis**: Message broker for Celery
- **Celery**: Async background task processing
- **LlamaParse**: High-quality document extraction

### Core Components

- **FastAPI**: Asynchronous REST API with fire-and-forget upload
- **PostgreSQL**: Metadata and state management (Async SQLAlchemy)
- **Qdrant**: Vector database for semantic search
- **Redis**: Message broker for Celery
- **Celery**: Async background task processing
- **LlamaParse**: High-quality document extraction

## Quick Start

### 1. Start Infrastructure Services

```bash
docker-compose up -d
```

This starts PostgreSQL, Qdrant, and Redis.

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

Required environment variables:
- `OPENAI_API_KEY`: For embeddings
- `LLAMA_PARSE_API_KEY`: For document parsing (or use simple parser)

### 3. Install Dependencies

```bash
poetry install
```

Or with pip:
```bash
pip install -e .
```

### 4. Initialize Database

```bash
python -c "import asyncio; from app.db.session import init_db; asyncio.run(init_db())"
```

### 5. Start Services

In separate terminals:

```bash
# Terminal 1: FastAPI Backend
uvicorn app.main:app --reload

# Terminal 2: Celery Worker
celery -A app.worker.celery_app worker --loglevel=info

# Terminal 3: Streamlit Frontend
streamlit run frontend/app.py
```

## Project Structure

```
.
├── app/
│   ├── api/              # FastAPI routers and dependencies
│   │   ├── deps.py       # Dependency injection
│   │   ├── ingest.py     # Document upload/status endpoints
│   │   └── schemas.py    # Pydantic models
│   ├── core/             # Configuration and factory
│   │   ├── config.py     # Settings with Pydantic
│   │   └── factory.py    # ComponentFactory for strategies
│   ├── db/               # Database models and session
│   │   ├── models.py     # SQLModel definitions
│   │   └── session.py    # Async session management
│   ├── interfaces/       # Abstract base classes
│   │   ├── parser.py     # BaseParser
│   │   ├── chunker.py    # BaseChunker
│   │   ├── embedder.py   # BaseEmbedder
│   │   └── vector_store.py  # BaseVectorStore
│   ├── strategies/       # Concrete implementations
│   │   ├── parsers/      # LlamaParse, Simple
│   │   ├── chunkers/     # Markdown, Recursive
│   │   ├── embedders/    # OpenAI
│   │   └── vector_stores/  # Qdrant
│   ├── main.py           # FastAPI app entry point
│   └── worker.py         # Celery worker entry point
├── frontend/
│   └── app.py            # Streamlit UI
├── docker-compose.yaml   # Infrastructure services
├── pyproject.toml        # Python dependencies
└── CLAUDE.md             # Developer instructions
```

## API Endpoints

### Upload Document

```bash
POST /ingest/upload
Headers: X-Org-ID: <your-org-id>
Content-Type: multipart/form-data

# Returns 202 Accepted
{
  "document_id": "uuid",
  "task_id": "celery-task-id",
  "status": "queued"
}
```

### Get Document Status

```bash
GET /ingest/status?page=1&page_size=20
Headers: X-Org-ID: <your-org-id>

# Returns
{
  "documents": [...],
  "total": 10,
  "page": 1,
  "page_size": 20
}
```

## Adding New Strategies

The system is **open for extension, closed for modification**.

### Add a New Parser

1. Inherit from `BaseParser`:

```python
# app/strategies/parsers/custom.py
from app.interfaces.parser import BaseParser, Document

class CustomParser(BaseParser):
    async def aload_data(self, file_path: str) -> list[Document]:
        # Your implementation
        pass

    @property
    def supported_extensions(self) -> set[str]:
        return {".custom"}
```

2. Register in `ComponentFactory.get_parser()`:

```python
case "custom":
    self._parser_cache = CustomParser()
```

3. Set `PARSER_TYPE=custom` in `.env`

No modification of existing code required!

## Development Commands

### Linting & Formatting

```bash
ruff check .
black .
```

### Type Checking

```bash
mypy .
```

### Testing

```bash
pytest tests/unit
pytest tests/integration
```

## Processing Pipeline

Documents flow through these states:

```
QUEUED → PARSING → CHUNKING → EMBEDDING → INDEXING → COMPLETED
                  |___________________________|
                              ↓
                           FAILED
```

Each state transition is persisted to PostgreSQL for tracking.

## Tenant Isolation

All operations are scoped by `org_id`:
- Qdrant collections are named `org_{org_id}`
- Database queries filter by `org_id`
- API requires `X-Org-ID` header

## License

MIT
# financial-document-Analyzer
