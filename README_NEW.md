# Template Intelligence Platform

A high-performance template analysis and injection system designed for the UK Financial Advisory market. It converts static Word documents into dynamic Jinja2 templates, enabling automated document generation with client-specific data through intelligent variable detection and tag injection.

## Architecture Overview

This system follows the **Strategy Pattern** and **Factory Pattern** for maximum extensibility, with an **async queue-based architecture** for scalable batch processing.

### Current Architecture: Async Queue-Based Template Workflow

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                          Frontend (Streamlit)                                    │
│  ┌──────────────┐  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────┐ │
│  │ Upload       │  │ Processing       │  │ Inject           │  │ Download    │ │
│  │ (Multi-file) │  │ Status Tab       │  │ Variables Tab    │  │ Tab         │ │
│  │              │  │ (Auto-refresh)    │  │ (Random filler)  │  │             │ │
│  └──────┬───────┘  └────────┬─────────┘  └────────┬─────────┘  └──────┬──────┘ │
│         │                    │                    │                     │         │
│         ▼                    ▼                    ▼                     ▼         │
│         APIClient (X-Org-ID header for multi-tenancy, batch_id tracking)          │
└─────────────────────────────────────┬──────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         FastAPI Backend                                         │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                    Template API Routes (Async)                           │   │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────┐        │   │
│  │  │ POST             │  │ GET              │  │ GET            │        │   │
│  │  │ /analyze-batch   │  │ /batch-status/   │  │ /list-ready    │        │   │
│  │  │ (Queue to Redis) │  │ {batch_id}       │  │ (Filter by     │        │   │
│  │  └────────┬─────────┘  └────────┬─────────┘  │  status)       │        │   │
│  │           │                     │            └────────┬───────┘        │   │
│  │  ┌────────▼─────────┐  ┌────────▼─────────┐  ┌────────▼────────┐        │   │
│  │  │ POST             │  │ POST             │  │ POST            │        │   │
│  │  │ /queue-next-batch│  │ /inject-random/  │  │ /finalize-async │        │   │
│  │  │ (Sequential)     │  │ {template_id}    │  │ (Queue task)    │        │   │
│  │  └──────────────────┘  └──────────────────┘  └─────────────────┘        │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────┬───────────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         Celery Worker (Async Processing)                         │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                         Celery Tasks                                     │   │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐       │   │
│  │  │ analyze_template │  │ finalize_template│  │ process_batch    │       │   │
│  │  │ - Status: QUEUED │  │ - Status: COMPL  │  │ - Orchestrate    │       │   │
│  │  │   → ANALYZING   │  │   → FINALIZING   │  │ - Auto-start     │       │   │
│  │  │   → COMPLETED   │  │   → COMPLETED    │  │   next batch     │       │   │
│  │  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘       │   │
│  │           │                     │                     │                  │   │
│  │  ┌────────▼─────────┐  ┌────────▼─────────┐  ┌────────▼─────────┐       │   │
│  │  │TemplateAnalyzer  │  │TemplateInjector  │  │Batch Manager     │       │   │
│  │  │- Detect vars     │  │- Inject Jinja2   │  │- Check complete  │       │   │
│  │  │- Store in DB     │  │- Save file       │  │- Start next      │       │   │
│  │  └──────────────────┘  └──────────────────┘  └──────────────────┘       │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                          Data Storage Layer                                      │
│  ┌──────────────────────┐  ┌──────────────────────┐  ┌────────────────────┐    │
│  │    PostgreSQL        │  │      Redis           │  │   File System      │    │
│  │  (Template Metadata) │  │   (Task Queue)       │  │ (Template Storage) │    │
│  │                      │  │                      │  │                    │    │
│  │  - organizations     │  │  - Celery broker     │  │  - Original .docx  │    │
│  │  - users             │  │  - Task state       │  │  - Tagged .docx    │    │
│  │  - templates         │  │  - Result backend   │  │  - uploads/        │    │
│  │  - status (enum)     │  │                      │  │    templates/      │    │
│  │  - batch_id          │  │                      │  │    {org_id}/       │    │
│  │  - detected_vars     │  │                      │  │                    │    │
│  └──────────────────────┘  └──────────────────────┘  └────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Batch Processing Flow

```
Upload Files → Create Batch → Queue Celery Tasks → Process Async → Poll Status → Download
                      │
                      ▼
              If batch in progress:
              Queue as next batch
              (auto-starts when complete)
```

### Template Status Pipeline

```
QUEUED → ANALYZING → COMPLETED → FINALIZING → COMPLETED (with download_ready=True)
  │          │           │            │                │
  │          │           │            │                └── Ready for download
  │          │           │            └── Injecting variables
  │          │           └── Analysis complete, ready for injection
  │          └── Processing template
  └── Waiting in queue
```

### Key Features

- **Multi-File Batch Upload**: Upload multiple Word documents at once
- **Async Background Processing**: Non-blocking API with Celery workers
- **Real-Time Status Updates**: Auto-refresh Processing Status tab
- **Sequential Batch Queue**: Next batch auto-starts when current completes
- **Random Value Generation**: UK-style random data for testing
- **Style Preservation**: Uses python-docx runs to maintain original formatting
- **Multi-Tenant Isolation**: org_id scoping for all operations

### Key Architectural Patterns

**Strategy Pattern** (Plugin System)
```
app/interfaces/          # Abstract Base Classes
└── template.py         → BaseTemplateAnalyzer, BaseTemplateInjector

app/strategies/         # Concrete Implementations
└── template_engine/    → TemplateAnalyzer, TemplateInjector
    ├── analyzer.py     # Variable detection (LLM/Regex)
    ├── injector.py     # Jinja2 tag injection
    └── models.py       # Pydantic models
```

**Factory Pattern** (Runtime Configuration)
```python
analyzer = ComponentFactory.get_template_analyzer()
injector = ComponentFactory.get_template_injector()
```

### Core Components

- **FastAPI**: Asynchronous REST API for template operations
- **Celery**: Background task queue for async processing
- **Redis**: Celery message broker
- **PostgreSQL**: Template metadata storage (Async SQLAlchemy)
- **python-docx**: Word document manipulation with style preservation
- **Streamlit**: Interactive tabbed frontend for template workflow
- **OpenAI/OpenRouter**: Optional LLM-powered variable detection

## Quick Start

### 1. Start Infrastructure

```bash
docker-compose up -d
```

This starts PostgreSQL, Redis, and Qdrant (optional for templates).

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your configuration
```

Required environment variables:
- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection for Celery

Optional:
- `USE_LLM_FOR_TEMPLATES`: Enable LLM-based template analysis (default: false, uses regex patterns)
- `OPENROUTER_API_KEY`: For LLM-powered analysis
- `LLM_CHAT_MODEL`: Model to use (default: openai/gpt-4o)

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
streamlit run frontend/template_ui.py
```

## Project Structure

```
.
├── app/
│   ├── api/              # FastAPI routers
│   │   ├── templates.py  # Template analysis, batch, injection endpoints
│   │   ├── schemas.py    # Pydantic models for API
│   │   └── deps.py       # Dependency injection (org_id, db session)
│   ├── core/             # Configuration and factory
│   │   ├── config.py     # Settings with Pydantic
│   │   └── factory.py    # ComponentFactory for strategies
│   ├── db/               # Database models and session
│   │   ├── models.py     # SQLModel definitions (TemplateStatus, TemplateStorage)
│   │   └── session.py    # Async session management
│   ├── interfaces/       # Abstract base classes
│   │   └── template.py   # BaseTemplateAnalyzer, BaseTemplateInjector
│   ├── strategies/       # Concrete implementations
│   │   └── template_engine/  # Template analysis & injection
│   │       ├── analyzer.py     # LLM/Regex variable detection
│   │       ├── injector.py     # Jinja2 tag injection
│   │       └── models.py       # DetectedVariable Pydantic model
│   ├── worker.py         # Celery task definitions
│   └── main.py           # FastAPI app entry point
├── frontend/
│   ├── app.py            # Main Streamlit UI
│   └── template_ui.py    # Template workflow UI with tabs
├── uploads/              # Temporary file storage
│   └── templates/        # Uploaded and tagged .docx files
├── docker-compose.yaml   # Infrastructure services
├── pyproject.toml        # Python dependencies
├── CLAUDE.md             # Developer instructions
└── README.md             # This file
```

## Frontend Tabs

### 1. Upload Tab

Upload multiple Word documents for batch processing:
- Accept multiple `.docx` files at once
- Shows current batch status (processing/queued)
- Automatically queues files as next batch if current batch is in progress
- Returns batch ID for tracking

### 2. Processing Status Tab

Real-time monitoring of template processing:
- Aggregate progress bar (X of Y completed)
- Individual file status with expandable details
- Error messages for failed templates
- Next batch information
- Download buttons for completed templates
- Auto-refreshes every 3 seconds

### 3. Inject Variables Tab

Review and inject variables into analyzed templates:
- Select from templates ready for injection
- Edit detected variable values
- **Fill Random Values** button - generates UK-style test data
- **Inject Variables** button - queues async injection
- **Random & Inject** button - one-click random injection

### 4. Download Tab

Download all completed templates:
- Lists all templates with `download_ready=True`
- Shows completion timestamp and variable count
- Direct download buttons for each template

## API Endpoints

### Batch Upload

```bash
POST /templates/analyze-batch
Headers: X-Org-ID: <your-org-id>
Content-Type: multipart/form-data

# Upload multiple .docx files
files: [<file1>, <file2>, ...]

# Returns
{
  "batch_id": "batch_abc123",
  "template_count": 3,
  "message": "Uploaded 3 templates for processing"
}
```

### Queue Next Batch

```bash
POST /templates/queue-next-batch?current_batch_id=<batch_id>
Headers: X-Org-ID: <your-org-id>
Content-Type: multipart/form-data

# Upload files for next batch (auto-starts when current completes)
files: [<file1>, ...]

# Returns
{
  "batch_id": "batch_def456",
  "template_count": 2,
  "message": "Queued 2 templates for next batch"
}
```

### Batch Status

```bash
GET /templates/batch-status/{batch_id}
Headers: X-Org-ID: <your-org-id>

# Returns
{
  "batch_id": "batch_abc123",
  "batch_status": "processing",
  "total_templates": 3,
  "completed": 1,
  "failed": 0,
  "in_progress": 2,
  "queued": 2,
  "next_batch_id": "batch_def456",
  "templates": [
    {
      "template_id": "uuid",
      "filename": "template.docx",
      "status": "completed",
      "progress": 100,
      "detected_variables": {...},
      "download_ready": true,
      "error_message": null
    },
    ...
  ]
}
```

### Template Status

```bash
GET /templates/status/{template_id}
Headers: X-Org-ID: <your-org-id>

# Returns
{
  "template_id": "uuid",
  "filename": "template.docx",
  "status": "completed",
  "progress": 100,
  "detected_variables": {...},
  "download_ready": true,
  "error_message": null
}
```

### List Ready Templates

```bash
GET /templates/list-ready?status=completed&download_ready=false&page=1&page_size=20
Headers: X-Org-ID: <your-org-id>

# Returns templates ready for injection
{
  "templates": [...],
  "total": 5,
  "page": 1,
  "page_size": 20
}
```

### Inject Random Values

```bash
POST /templates/inject-random/{template_id}
Headers: X-Org-ID: <your-org-id>

# Generates random values and queues injection
# Returns
{
  "template_id": "uuid",
  "status": "queued",
  "task_id": "celery-task-id",
  "message": "Template finalization with random values queued"
}
```

### Finalize Template (Async)

```bash
POST /templates/finalize-async
Headers: X-Org-ID: <your-org-id>
Content-Type: application/json

{
  "template_id": "uuid",
  "variables": [
    {
      "suggested_variable_name": "client_name",
      "original_text": "Mr. John Smith",
      "value": "Jane Doe"
    }
  ]
}

# Returns
{
  "template_id": "uuid",
  "status": "queued",
  "task_id": "celery-task-id",
  "message": "Template finalization queued"
}
```

### Download Template

```bash
GET /templates/download/{template_id}
Headers: X-Org-ID: <your-org-id>

# Downloads the tagged .docx with Jinja2 variables
```

## Celery Tasks

### analyze_template

Analyzes a template asynchronously to detect variables.

```python
@shared_task(bind=True, name="app.worker.analyze_template")
def analyze_template_task(self, template_id: str) -> dict
```

**Flow:**
1. Update status to ANALYZING
2. Run TemplateAnalyzer.analyze()
3. Store detected_variables in DB
4. Update status to COMPLETED or FAILED
5. Check if batch is complete, start next batch

### finalize_template

Finalizes a template with variable injection asynchronously.

```python
@shared_task(bind=True, name="app.worker.finalize_template")
def finalize_template_task(self, template_id: str, variables: list[dict]) -> dict
```

**Flow:**
1. Update status to FINALIZING
2. Run TemplateInjector.inject_tags()
3. Save tagged file path
4. Set download_ready=True
5. Update status to COMPLETED
6. Check if batch is complete, start next batch

### process_batch

Orchestrates batch processing.

```python
@shared_task(bind=True, name="app.worker.process_batch")
def process_batch_task(self, batch_id: str) -> dict
```

**Flow:**
1. Update batch_status to "processing"
2. Queue analyze_template_task for each template in batch
3. Track completion with Celery group
4. On completion, check for next_batch_id and start it

## Database Models

### TemplateStatus Enum

```python
class TemplateStatus(str, enum.Enum):
    QUEUED = "queued"        # Waiting in queue
    ANALYZING = "analyzing"  # Detecting variables
    FINALIZING = "finalizing"  # Injecting Jinja2 tags
    COMPLETED = "completed"  # Processing complete
    FAILED = "failed"        # Processing failed
```

### TemplateStorage Model

```python
class TemplateStorage(SQLModel, table=True):
    id: uuid.UUID
    org_id: uuid.UUID
    name: str
    original_filename: str
    file_path: str

    # Analysis results
    detected_variables: dict[str, Any] | None  # JSONB
    paragraph_count: int | None
    is_tagged: bool = False

    # Processing status
    status: TemplateStatus = TemplateStatus.QUEUED
    task_id: str | None
    processing_started_at: datetime | None
    processing_completed_at: datetime | None
    error_message: str | None

    # Batch tracking
    batch_id: str | None
    previous_batch_id: str | None  # Links sequential batches
    batch_status: str | None  # "processing", "queued", "completed"

    # Download tracking
    download_ready: bool = False
    download_url: str | None
```

## Supported Patterns (Regex-based)

| Pattern | Variable Name |
|---------|---------------|
| `Mr/Mrs/Ms/Dr First Last` | `client_full_name` |
| `DD/MM/YYYY` | `date` |
| `£1,234.56` | `monetary_amount` |
| `50%` | `percentage` |
| `123 Street Road` | `address_line` |

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

## Tenant Isolation

All operations are scoped by `org_id`:
- Database queries filter by `org_id`
- File storage organized by `uploads/templates/{org_id}/`
- API requires `X-Org-ID` header
- Batches are isolated per organization

## License

MIT
