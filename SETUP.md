# Quick Setup Guide

Get the Template Intelligence Platform running in 5 minutes.

## Prerequisites

| Tool | Version | Check |
|------|---------|-------|
| Python | 3.12+ | `python --version` |
| Poetry | Latest | `poetry --version` |
| Docker | Latest | `docker --version` |

## 1. Start Infrastructure (One Command)

```bash
docker-compose up -d
```

**What this starts:**
- PostgreSQL 16 (port 5432)
- Redis 7 (port 6379)
- Qdrant (ports 6333/6334)

## 2. Install Dependencies

```bash
# Option A: Poetry (recommended)
poetry install

# Option B: pip
pip install -e .
```

## 3. Configure Environment

Create `.env` file in the project root:

```bash
# Minimal .env for local development
DATABASE_URL=postgresql+asyncpg://docuser:docpass@localhost:5432/docplatform
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
ORG_ID=bf0d03fb-d8ea-4377-a991-b3b5818e71ec
API_BASE_URL=http://localhost:8000
```

## 4. Initialize Database

```bash
make db
```

## 5. Start Services (3 Terminals)

```bash
# Terminal 1: API Backend
make api

# Terminal 2: Celery Worker
make worker

# Terminal 3: Streamlit Frontend
make frontend
```

## 6. Open & Test

1. Open http://localhost:8501
2. Look for "✅ API Connected" in sidebar
3. Upload a `.docx` template

---

## Cloud Deployment (Render/Railway)

### Update `.env` for cloud:

```bash
# Database (Aiven/Neon)
DATABASE_URL=postgresql+asyncpg://user:pass@host:port/db?ssl=require

# Redis (Upstash)
CELERY_BROKER_URL=rediss://default:token@host:port/0
CELERY_RESULT_BACKEND=rediss://default:token@host:port/0

# API URL
API_BASE_URL=https://your-app.onrender.com
```

### SSL is Auto-Configured

The codebase automatically detects `rediss://` and configures SSL for Celery.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| API Disconnected | Run `make api` |
| Worker won't start | Run `docker-compose up -d` |
| Database errors | Run `make db` |
| Clock drift warning | Restart worker (harmless) |
| API is asleep | Click API URL in sidebar to wake it |

---

## All Commands

```bash
make help     # Show all commands
make dev      # Start Docker
make down     # Stop Docker
make api      # Start FastAPI
make worker   # Start Celery
make frontend # Start Streamlit
make db       # Init database
make logs     # Docker logs
make clean    # Clean cache
```

---

## Next Steps

- [README.md](README.md) - Full documentation
- [CLAUDE.md](CLAUDE.md) - Developer guide
