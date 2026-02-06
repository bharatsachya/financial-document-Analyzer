#!/bin/bash

# 1. SETUP: Create uploads folder (since it doesn't exist in git)
echo "📂 Creating upload directories..."
mkdir -p uploads/templates

# 2. DATABASE: Initialize DB (equivalent to 'make db')
# We run this every time to ensure tables exist on the fresh Neon DB
echo "🗄️ Initializing database..."
python -m scripts.init_db

# 3. WORKER: Start Celery in background (equivalent to 'make worker')
# The '&' is crucial—it lets the script continue to the next step
echo "👷 Starting Celery worker..."
celery -A app.worker.celery_app worker --loglevel=info --concurrency=1 &

# 4. API: Start FastAPI (equivalent to 'make api')
# We use $PORT provided by Render
echo "🚀 Starting FastAPI server..."
uvicorn app.main:app --host 0.0.0.0 --port $PORT