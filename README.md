# Template Intelligence Platform

A high-performance template analysis and injection system designed for the UK Financial Advisory market. It converts static Word documents into dynamic Jinja2 templates, enabling automated document generation with client-specific data through intelligent variable detection and tag injection.

## Architecture Overview

This system follows the **Strategy Pattern** and **Factory Pattern** for maximum extensibility.

### Current Architecture: Synchronous Template Workflow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Frontend (Streamlit)                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                   │
│  │ Upload .docx │  │ Edit Vars    │  │ Download     │                   │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                   │
│         │                  │                  │                           │
│         ▼                  ▼                  ▼                           │
│         APIClient (X-Org-ID header for multi-tenancy)                     │
└──────────────────────────────┬────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         FastAPI Backend                                  │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    Template API Routes                            │   │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────┐  │   │
│  │  │ POST /analyze    │  │ POST /finalize   │  │ GET /download  │  │   │
│  │  │ Detect Variables │  │ Inject Jinja2    │  │ Tagged File    │  │   │
│  │  └────────┬─────────┘  └────────┬─────────┘  └────────┬───────┘  │   │
│  │           │                     │                     │          │   │
│  │           ▼                     ▼                     ▼          │   │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐ │   │
│  │  │TemplateAnalyzer  │  │TemplateInjector  │  │File Storage      │ │   │
│  │  │- LLM or Regex    │  │- python-docx     │  │uploads/templates/│ │   │
│  │  │- Paragraph scan  │  │- Style preserve  │  │                  │ │   │
│  │  └──────────────────┘  └──────────────────┘  └──────────────────┘ │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────┬────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          Data Storage Layer                              │
│  ┌──────────────────────┐  ┌──────────────────────────────────────┐    │
│  │    PostgreSQL        │  │           File System                │    │
│  │  (Template Metadata) │  │    (Template Storage)                │    │
│  │                      │  │                                      │    │
│  │  - organizations     │  │  - Original .docx files              │    │
│  │  - users             │  │  - Tagged .docx files                │    │
│  │  - templates         │  │  - uploads/templates/{org_id}/       │    │
│  └──────────────────────┘  └──────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

### Async Queue Architecture (Future Design)

For high-volume template processing, the system can be extended with Celery queues:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Upload .docx   │───▶│  Queue to       │───▶│  Worker         │
│                 │    │  Redis (Celery) │    │  Analyzes Async │
└─────────────────┘    └─────────────────┘    └────────┬────────┘
                                                       │
                       ┌─────────────────┐             │
                       │  Poll Status    │◀────────────┘
                       │  (GET /status)   │
                       └────────┬────────┘
                                │
                       ┌────────▼────────┐
                       │  Review & Edit  │
                       │  Variables      │
                       └────────┬────────┘
                                │
                       ┌────────▼────────┐    ┌─────────────────┐
                       │  Finalize &     │───▶│  Worker         │
                       │  Inject Async   │    │  Injects Jinja2 │
                       └─────────────────┘    └─────────────────┘
```

**Benefits of Async Queue:**
- Non-blocking API responses for large documents
- Parallel processing of multiple templates
- Retry logic for failed operations
- Worker scaling based on queue depth

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
- **PostgreSQL**: Template metadata storage (Async SQLAlchemy)
- **python-docx**: Word document manipulation with style preservation
- **Streamlit**: Interactive frontend for template workflow
- **OpenAI/OpenRouter**: Optional LLM-powered variable detection

## Quick Start

### 1. Start Database

```bash
docker-compose up -d postgresql
```

This starts PostgreSQL for template metadata storage.

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

Required environment variables:
- `DATABASE_URL`: PostgreSQL connection string

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

# Terminal 2: Streamlit Frontend
streamlit run frontend/template_ui.py
```

## Project Structure

```
.
├── app/
│   ├── api/              # FastAPI routers
│   │   ├── templates.py  # Template analysis & injection endpoints
│   │   └── deps.py       # Dependency injection (org_id, db session)
│   ├── core/             # Configuration and factory
│   │   ├── config.py     # Settings with Pydantic
│   │   └── factory.py    # ComponentFactory for strategies
│   ├── db/               # Database models and session
│   │   ├── models.py     # SQLModel definitions (Organization, TemplateStorage)
│   │   └── session.py    # Async session management
│   ├── interfaces/       # Abstract base classes
│   │   └── template.py   # BaseTemplateAnalyzer, BaseTemplateInjector
│   ├── strategies/       # Concrete implementations
│   │   └── template_engine/  # Template analysis & injection
│   │       ├── analyzer.py     # LLM/Regex variable detection
│   │       ├── injector.py     # Jinja2 tag injection
│   │       └── models.py       # DetectedVariable Pydantic model
│   └── main.py           # FastAPI app entry point
├── frontend/
│   ├── app.py            # Main Streamlit UI
│   └── template_ui.py    # Dedicated template workflow UI
├── uploads/              # Temporary file storage
│   └── templates/        # Uploaded and tagged .docx files
├── docker-compose.yaml   # Infrastructure services (PostgreSQL)
├── pyproject.toml        # Python dependencies
└── CLAUDE.md             # Developer instructions
```

## API Endpoints

### Analyze Template

Detects variables in a Word document that should be converted to Jinja2 tags.

```bash
POST /templates/analyze
Headers: X-Org-ID: <your-org-id>
Content-Type: multipart/form-data

# Upload .docx template
# Returns detected variables for review
{
  "template_id": "uuid",
  "filename": "template.docx",
  "detected_variables": [
    {
      "original_text": "Mr. John Smith",
      "suggested_variable_name": "client_full_name",
      "rationale": "Detected pattern matching client_full_name",
      "paragraph_index": 5
    }
  ],
  "total_paragraphs": 20,
  "analyzed_at": "2026-02-02T12:00:00Z"
}
```

### Finalize Template

Injects Jinja2 tags into the template after reviewing the detected variables.

```bash
POST /templates/finalize
Headers: X-Org-ID: <your-org-id>
Content-Type: application/json

{
  "template_id": "uuid",
  "variables": [...],  # Reviewed variables from analyze response
  "original_filename": "template.docx"
}

# Returns download URL for tagged template
{
  "template_id": "uuid",
  "status": "finalized",
  "download_url": "/templates/download/{template_id}",
  "variable_count": 5
}
```

### Download Template

```bash
GET /templates/download/{template_id}
Headers: X-Org-ID: <your-org-id>

# Downloads the tagged .docx with Jinja2 variables
```

### Save Template (Optional Storage)

Store analyzed templates in PostgreSQL for later retrieval.

```bash
POST /templates/save
Headers: X-Org-ID: <your-org-id>
Content-Type: application/json

{
  "name": "Client Report Template",
  "original_filename": "template.docx",
  "description": "Annual client summary report",
  "template_id": "uuid",
  "detected_variables": [...]
}

# Returns saved template record
{
  "id": "uuid",
  "name": "Client Report Template",
  "org_id": "org-uuid",
  "created_at": "2026-02-02T12:00:00Z"
}
```

### List Stored Templates

```bash
GET /templates/stored?page=1&page_size=20
Headers: X-Org-ID: <your-org-id>

# Returns
{
  "templates": [...],
  "total": 10,
  "page": 1
}
```

## Template Intelligence Engine (TIE)

The TIE system provides intelligent Word template analysis and Jinja2 variable injection for document generation workflows.

### Features

- **Automatic Variable Detection**: Identifies dynamic content (names, dates, amounts) using regex patterns or LLM analysis
- **Paragraph Index Tracking**: Preserves precise locations for handling duplicate text occurrences
- **Style Preservation**: Uses python-docx runs to maintain original formatting
- **Human Review Workflow**: Analyze → Review → Finalize → Download

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
│  Upload .docx   │───▶│  Analyze &      │───▶│  Review & Edit  │
│                 │    │  Detect Vars    │    │  Variables      │
└─────────────────┘    └─────────────────┘    └────────┬────────┘
                                                       │
                                                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Download       │◀───│  Finalize &     │◀───│  Approve List   │
│  Tagged .docx   │    │  Inject Tags    │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Usage Example

```python
from app.strategies.template_engine import TemplateAnalyzer, TemplateInjector

# Analyze a template
analyzer = TemplateAnalyzer(use_llm=False)
variables = await analyzer.analyze("template.docx")

# Variables are detected with paragraph indices for precise replacement
# [
#   DetectedVariable(
#     original_text="Mr. John Smith",
#     suggested_variable_name="client_full_name",
#     rationale="Detected pattern matching client_full_name",
#     paragraph_index=5
#   ),
#   ...
# ]

# Inject Jinja2 tags
injector = TemplateInjector()
tagged_path = await injector.inject_tags(
    file_path="template.docx",
    variables=variables,
    output_path="template_tagged.docx"
)
```

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

## License

MIT
