# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

### IMPORTANT CONSIDERATIONS

- **Code Quality**: Ensure code is highly modular, type-safe (Pydantic v2), and follows the **Strategy & Factory Patterns**.
- **Async First**: Use `async/await` for all I/O bound operations (Database, API calls, Vector Store).
- **Strict Typing**: Use Python type hints everywhere. No `Any` unless absolutely necessary.
- **Logging**: Use the configured `structlog` or standard `logging` module. Do not use `print`.

### Productivity Hacks

- **Reduce Tool Calls**:
  - Use `tree -L 2` to understand the directory structure instead of searching file by file.
  - Assume standard FastAPI/Celery structures unless the tree suggests otherwise.
- **Codebase Context**:
  - Use `grep -r "class .*Base"` to find abstract base classes and interfaces.
  - Read `app/main.py` and `app/worker.py` first to understand the entry points.

## Development Commands

### Core Commands

- **Backend Dev**: `uvicorn app.main:app --reload` - Start FastAPI with auto-reload
- **Frontend Dev**: `streamlit run frontend/app.py` - Start Streamlit UI
- **Celery Worker**: `celery -A app.worker.celery_app worker --loglevel=info` - Start background worker
- **Lint/Format**: `ruff check .` / `black .` - Python linting and formatting

### Infrastructure (Docker)

- **Start Services**: `docker-compose up -d` - Starts PostgreSQL, Qdrant, Redis
- **Stop Services**: `docker-compose down`
- **View Logs**: `docker-compose logs -f`

### Testing

- **Unit Tests**: `pytest tests/unit`
- **Integration Tests**: `pytest tests/integration`
- **Type Check**: `mypy .`

## System Architecture

This is a **high-performance, event-driven document ingestion pipeline** designed for the UK Financial Advisory market. It processes unstructured financial documents (PDFs, Docx) into structured, queryable knowledge chunks in Qdrant.

### Core Data Models (SQLModel)

- **Organization**: Tenant isolation root.
- **User**: Authenticated entity belonging to an Organization.
- **Document**: Tracks file status (`PENDING` -> `PARSING` -> `CHUNKING` -> `INDEXING` -> `COMPLETED`).

### Processing Flow (Async Pipeline)

User Upload (Streamlit) → API (FastAPI) → Redis Queue (Celery) ↓ Worker Process ↓ 1. Strategy Selection (Factory) 2. Parsing (LlamaParse) 3. Semantic Chunking (Markdown/Recursive) 4. Embedding (OpenAI/Cohere) 5. Vector Upsert (Qdrant)


### Key Architectural Patterns

- **Strategy Pattern**: Core logic (Parsing, Chunking, Embedding) is defined via Abstract Base Classes (`interfaces/`).
- **Factory Pattern**: `ComponentFactory` instantiates specific strategies at runtime based on config.
- **Tenant Isolation**: All Qdrant payloads MUST include `org_id`. Retrieval queries must filter by `org_id`.

## Infrastructure Dependencies

- **FastAPI**: Asynchronous REST API.
- **PostgreSQL**: Metadata and state management (Async SQLAlchemy).
- **Qdrant**: Vector database for semantic search.
- **Redis**: Message broker for Celery.
- **LlamaParse**: Primary engine for complex document extraction.

## Development Patterns

### Ingestion Pipeline **[CRITICAL]**

- **Fire-and-Forget**: The API returns immediately after queuing the task. Polling handles status updates.
- **Idempotency**: Ensure tasks can be retried without duplicating vectors (use deterministic IDs where possible).
- **Error Handling**: 
  - If a step fails, catch the exception, log the stack trace, and set Document status to `FAILED` with an `error_message`.
  - Do not leave documents in `PROCESSING` indefinitely.

### Strategy Implementation

When adding a new capability (e.g., a new Parser):
1.  Inherit from the Base Class (e.g., `BaseParser`).
2.  Implement the required methods (e.g., `aload_data`).
3.  Register the new class in the `ComponentFactory`.
4.  Do NOT modify existing implementations (Open/Closed Principle).

### State Management

- **PostgreSQL** is the source of truth for *Status*.
- **Qdrant** is the source of truth for *Knowledge*.
- **Optimistic Locking**: Not strictly required yet, but keep transactions short.

## File Structure

. ├── app/ │ ├── api/ # FastAPI Routers │ │ ├── auth.py │ │ ├── ingest.py │ │ └── deps.py │ ├── core/ # Config & Factories │ │ ├── config.py │ │ └── factory.py │ ├── db/ # Database setup │ │ ├── session.py │ │ └── models.py │ ├── interfaces/ # Abstract Base Classes │ │ ├── parser.py │ │ ├── chunker.py │ │ └── vector_store.py │ ├── strategies/ # Concrete Implementations │ │ ├── parsers/ │ │ │ ├── llama_parse.py │ │ │ └── simple.py │ │ ├── chunkers/ │ │ └── vector_stores/ │ ├── main.py # API Entrypoint │ └── worker.py # Celery Worker Entrypoint ├── frontend/ │ └── app.py # Streamlit UI ├── tests/ ├── docker-compose.yaml ├── pyproject.toml └── README.md


## Development Principles

### KISS (Keep It Simple, Stupid)

- Use standard libraries where possible.
- Complex logic belongs in `strategies/`, not in `api/` routes.

### Open/Closed Principle

- Classes should be open for extension but closed for modification.
- New file formats = New Parser Strategy, not `if/else` chains in `main.py`.

### Development Guidelines

- **Ask First**: If changing a Base Class interface, discuss it first. It affects all strategies.
- **Async Everywhere**: Do not use blocking I/O (like standard `requests`) inside async routes or Celery tasks. Use `httpx` or async SDK clients.
- **Explanation**: Explain your approach step-by-step before writing code, especially when implementing complex logic.
- **No Mocks**: Unless specifically asked, write actual implementation code that connects to services (assuming env vars are present).