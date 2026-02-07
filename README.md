# Template Intelligence Platform

A high-performance, event-driven template analysis and injection system designed for the UK Financial Advisory market. It converts static Word documents into dynamic templates through intelligent variable detection, supporting both synchronous and asynchronous processing workflows with Celery workers.

## Architecture Overview

This system follows the **Strategy Pattern** and **Factory Pattern** for maximum extensibility, with full support for async background processing.

### Current Architecture: Async Queue-Based Workflow

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         Frontend (Streamlit)                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                     │
│  │ Upload .docx │  │ Edit Vars    │  │ Download     │                     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                     │
│         │                  │                  │                             │
│         ▼                  ▼                  ▼                             │
│         APIClient (X-Org-ID header for multi-tenancy)                       │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                          FastAPI Backend                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Template API Routes                              │   │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │   │
│  │  │ POST /analyze    │  │ POST /finalize   │  │ GET /download    │  │   │
│  │  │ Queue to Celery  │  │ Queue to Celery  │  │ Tagged File      │  │   │
│  │  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘  │   │
│  │           │                     │                     │            │   │
│  │           ▼                     ▼                     ▼            │   │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐ │   │
│  │  │Celery - Analyze  │  │Celery - Finalize │  │File Storage      │ │   │
│  │  │Template Task     │  │Template Task     │  │uploads/templates/│ │   │
│  │  └──────────────────┘  └──────────────────┘  └──────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                       Background Worker Layer                               │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │                     Celery Worker Processes                          │ │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │ │
│  │  │TemplateAnalyzer  │  │TemplateInjector  │  │Batch Processing  │  │ │
│  │  │- LLM or Regex    │  │- python-docx     │  │- Chain Batches  │  │ │
│  │  │- Paragraph scan  │  │- Style preserve  │  │- Status tracking │  │ │
│  │  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘  │ │
│  │           │                     │                     │            │ │
│  └───────────┼─────────────────────┼─────────────────────┼────────────┘ │
│              ▼                     ▼                     ▼              │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    Shared Dependencies                            │   │
│  │  - PostgreSQL (Aiven Cloud w/ SSL)                                │   │
│  │  - Redis (Upstash Cloud w/ SSL)                                   │   │
│  │  - OpenRouter LLM (for intelligent analysis)                      │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────┘
```

### Key Benefits of Async Architecture

- **Non-blocking API responses** for large documents
- **Parallel processing** of multiple templates via worker pool
- **Automatic retry logic** for failed operations
- **Horizontal worker scaling** based on queue depth
- **Batch processing** support with sequential workflow

### Cloud Infrastructure

The system is configured for cloud deployment with managed services:

| Service | Provider | Purpose |
|---------|----------|---------|
| **PostgreSQL** | Aiven | Template metadata (asyncpg w/ SSL) |
| **Redis** | Upstash | Celery broker & backend (rediss:// w/ SSL) |
| **LLM** | OpenRouter | Intelligent variable detection (optional) |

## Quick Start

### 1. Clone and Install

```bash
git clone <repository-url>
cd Challenge
poetry install
```

Or with pip:
```bash
pip install -e .
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

#### Required Environment Variables

```bash
# Database (Aiven PostgreSQL - requires SSL)
DATABASE_URL=postgresql+asyncpg://user:pass@host:port/db?ssl=require

# Redis/ Celery (Upstash Redis - requires SSL)
CELERY_BROKER_URL=rediss://default:token@host:port/0
CELERY_RESULT_BACKEND=rediss://default:token@host:port/0

# Multi-tenancy
ORG_ID=bf0d03fb-d8ea-4377-a991-b3b5818e71ec  # Default org UUID

# API Configuration
API_BASE_URL=http://localhost:8000  # For Streamlit frontend
```

#### Optional Environment Variables

```bash
# LLM Configuration (for intelligent analysis)
OPENROUTER_API_KEY=sk-or-v1-...
OPENAI_BASE_URL=https://openrouter.ai/api/v1
LLM_CHAT_MODEL=qwen/qwen-2.5-7b-instruct
USE_LLM_FOR_TEMPLATES=true  # false = regex patterns only

# File Storage
UPLOAD_DIR=./uploads

# Logging
LOG_LEVEL=INFO
```

### 3. Initialize Database

```bash
# Using init_db script
python -m scripts.init_db

# Or via Python
python -c "import asyncio; from app.db.session import init_db; asyncio.run(init_db())"
```

### 4. Start Services

Using `make` (recommended):

```bash
# Terminal 1: FastAPI Backend
make api

# Terminal 2: Celery Worker
make worker

# Terminal 3: Streamlit Frontend
make frontend
```

Or manually:

```bash
# Terminal 1: FastAPI Backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Celery Worker
celery -A app.worker.celery_app worker --loglevel=info

# Terminal 3: Streamlit Frontend
streamlit run frontend/app.py
```

## Project Structure

```
.
├── app/
│   ├── api/              # FastAPI routers
│   │   └── templates.py  # Template analysis & injection endpoints
│   ├── core/             # Configuration and factory
│   │   ├── config.py     # Settings with Pydantic
│   │   ├── factory.py    # ComponentFactory for strategies
│   │   └── logging_config.py  # Structlog configuration
│   ├── db/               # Database models and session
│   │   ├── models.py     # SQLModel definitions
│   │   └── session.py    # Async session management
│   ├── interfaces/       # Abstract base classes
│   │   └── template.py   # BaseTemplateAnalyzer, BaseTemplateInjector
│   ├── strategies/       # Concrete implementations
│   │   └── template_engine/
│   │       ├── analyzer.py     # LLM/Regex variable detection
│   │       ├── injector.py     # Value injection
│   │       └── models.py       # Pydantic models
│   ├── main.py           # FastAPI app entry point
│   └── worker.py         # Celery worker entry point
├── frontend/
│   ├── app.py            # Main Streamlit UI
│   └── template_ui.py    # Template workflow UI
├── uploads/              # File storage
│   └── templates/        # Organized by org_id
├── tests/
│   ├── unit/             # Unit tests
│   └── integration/      # Integration tests
├── scripts/
│   └── init_db.py        # Database initialization
├── Makefile              # Development commands
├── docker-compose.yaml   # Local infrastructure (optional)
├── pyproject.toml        # Python dependencies
├── CLAUDE.md             # Developer instructions
└── README.md             # This file
```

## API Endpoints

### Health Check

```bash
GET /health

# Returns
{
  "status": "healthy",
  "service": "template-intelligence-api",
  "version": "0.1.0"
}
```

### Analyze Template (Async)

Detects variables in a Word document. Returns immediately with template_id for status polling.

```bash
POST /templates/analyze
Headers: X-Org-ID: <your-org-id>
Content-Type: multipart/form-data

# Upload .docx template
# Returns immediately with template_id for polling
{
  "template_id": "uuid",
  "filename": "template.docx",
  "status": "queued"
}
```

### Check Template Status

```bash
GET /templates/{template_id}/status
Headers: X-Org-ID: <your-org-id>

# Returns processing status
{
  "template_id": "uuid",
  "status": "analyzing" | "completed" | "failed",
  "detected_variables": [...],  # Available when completed
  "error_message": null
}
```

### Finalize Template (Async)

Injects actual values (not Jinja2 tags) after reviewing detected variables.

```bash
POST /templates/finalize
Headers: X-Org-ID: <your-org-id>
Content-Type: application/json

{
  "template_id": "uuid",
  "variables": [
    {"original_text": "...", "value": "..."}
  ]
}

# Returns immediately
{
  "template_id": "uuid",
  "status": "queued"
}
```

### Download Template

```bash
GET /templates/download/{template_id}
Headers: X-Org-ID: <your-org-id>

# Downloads the processed .docx with injected values
```

## Development Commands

### All Commands

```bash
make help              # Show all available commands
```

### Infrastructure

```bash
make dev               # Start all services (docker-compose)
make down              # Stop all services
make logs              # Show Docker logs
make ps                # Show running containers
```

### Application Services

```bash
make api               # Start FastAPI backend (port 8000)
make worker            # Start Celery worker
make frontend          # Start Streamlit frontend
```

### Code Quality

```bash
make lint              # Run ruff linter
make format            # Format code with ruff & black
make typecheck         # Run mypy type checker
```

### Testing

```bash
make test              # Run all tests
make test-unit         # Run unit tests only
make test-integration  # Run integration tests only
```

### Utilities

```bash
make clean             # Clean cache files
make db                # Initialize database
```

## Template Intelligence Engine (TIE)

The TIE system provides intelligent Word template analysis and value injection for document generation workflows.

### Features

- **Automatic Variable Detection**: Identifies dynamic content (names, dates, amounts) using regex patterns or LLM analysis
- **Async Processing**: Non-blocking template analysis via Celery workers
- **Batch Support**: Process multiple templates with sequential workflows
- **Style Preservation**: Uses python-docx runs to maintain original formatting
- **Human Review Workflow**: Upload → Analyze → Review → Finalize → Download

### Supported Patterns (Regex-based)

| Pattern | Variable Name |
|---------|---------------|
| `Mr/Mrs/Ms/Dr First Last` | `client_full_name` |
| `DD/MM/YYYY` | `date` |
| `£1,234.56` | `monetary_amount` |
| `50%` | `percentage` |
| `123 Street Road` | `address_line` |

### TIE Workflow

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Upload .docx   │───▶│  Analyze &      │───▶│  Poll Status    │
│                 │    │  Detect Vars    │    │  (Async Task)   │
└─────────────────┘    └─────────────────┘    └────────┬────────┘
                                                       │
                       ┌─────────────────┐             │
                       │  Review & Edit  │◀────────────┘
                       │  Variables      │
                       └────────┬────────┘
                                │
                       ┌────────▼────────┐    ┌─────────────────┐
                       │  Finalize &     │───▶│  Poll Status    │
                       │  Inject Values  │    │  (Async Task)   │
                       └────────┬────────┘    └────────┬────────┘
                                │                      │
                       ┌────────▼────────┐             │
                       │  Download       │◀────────────┘
                       │  Filled .docx   │
                       └─────────────────┘
```

## Key Architectural Patterns

### Strategy Pattern (Plugin System)

```
app/interfaces/           # Abstract Base Classes
└── template.py          → BaseTemplateAnalyzer, BaseTemplateInjector

app/strategies/          # Concrete Implementations
└── template_engine/     → TemplateAnalyzer, TemplateInjector
    ├── analyzer.py      # Variable detection (LLM/Regex)
    ├── injector.py      # Value injection
    └── models.py        # Pydantic models
```

### Factory Pattern (Runtime Configuration)

```python
from app.core.factory import ComponentFactory

analyzer = factory.get_template_analyzer(custom_prompt="...")
injector = factory.get_template_injector()
```

## Tenant Isolation

All operations are scoped by `org_id`:
- Database queries filter by `org_id`
- File storage organized by `uploads/templates/{org_id}/`
- API requires `X-Org-ID` header

## Deployment

### Cloud Environment Variables

For production deployment, update these environment variables:

```bash
# Database
DATABASE_URL=postgresql+asyncpg://...?ssl=require

# Celery (Upstash)
CELERY_BROKER_URL=rediss://...
CELERY_RESULT_BACKEND=rediss://...

# API
API_BASE_URL=https://your-app.onrender.com

# LLM
OPENROUTER_API_KEY=sk-or-v1-...
```

### SSL Configuration Notes

- **PostgreSQL**: Uses `ssl=require` query parameter for asyncpg
- **Redis (Upstash)**: Uses `rediss://` scheme with `ssl.CERT_NONE` for cert validation
- **Celery**: Dynamically configures SSL based on URL scheme in worker.py

## Troubleshooting

### Celery Worker Clock Drift Warning

If you see "Substantial drift from celery" warnings, restart the worker:

```bash
# Sync system time (macOS)
sudo sntp -sS time.apple.com

# Restart worker
make worker
```

### "API Disconnected" in Streamlit

Ensure the FastAPI backend is running:

```bash
curl http://localhost:8000/health
```

Should return `{"status":"healthy",...}`

### Database Connection Errors

For `sslmode` errors with Aiven PostgreSQL, ensure your URL uses:
- `ssl=require` (asyncpg) NOT `sslmode=require` (psycopg2)

## License

MIT
