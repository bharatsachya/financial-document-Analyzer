#!/bin/bash

# Template Intelligence Platform - Quick Setup Script
# This script automates the entire setup process

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC}  Template Intelligence Platform - Quick Setup            ${BLUE}║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_step() {
    echo -e "${GREEN}➤${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✖${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        print_error "$1 is not installed"
        return 1
    fi
    return 0
}

# Main setup
print_header

# Check prerequisites
print_step "Checking prerequisites..."

MISSING=0

if ! check_command "python3"; then
    MISSING=1
fi

if ! check_command "docker"; then
    MISSING=1
fi

if ! check_command "docker-compose"; then
    # Try docker compose (v2)
    if ! docker compose version &> /dev/null; then
        MISSING=1
    fi
fi

if [ $MISSING -eq 1 ]; then
    print_error "Missing prerequisites. Please install Python 3.12+, Docker, and Docker Compose."
    exit 1
fi

print_success "Prerequisites met"

# Check Python version
PYTHON_VERSION=$(python3 --version | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 12 ]); then
    print_error "Python 3.12+ is required (found $PYTHON_VERSION)"
    exit 1
fi

print_success "Python $PYTHON_VERSION detected"

# Step 1: Start Docker infrastructure
print_step "Starting Docker infrastructure..."
docker-compose up -d

# Wait for services to be healthy
print_step "Waiting for services to be ready..."
sleep 5

# Check if services are running
if ! docker-compose ps | grep -q "Up"; then
    print_error "Docker services failed to start"
    docker-compose logs
    exit 1
fi

print_success "Docker infrastructure started"

# Step 2: Install dependencies
print_step "Installing Python dependencies..."

if command -v "poetry" &> /dev/null; then
    print_step "Using Poetry..."
    poetry install
elif command -v "pip3" &> /dev/null; then
    print_step "Using pip..."
    pip3 install -e .
else
    print_error "Neither Poetry nor pip3 found"
    exit 1
fi

print_success "Dependencies installed"

# Step 3: Check/create .env file
print_step "Checking environment configuration..."

if [ ! -f ".env" ]; then
    print_warning ".env file not found"

    # Check if .env.example exists
    if [ -f ".env.example" ]; then
        print_step "Creating .env from .env.example..."
        cp .env.example .env
    else
        print_step "Creating minimal .env file..."
        cat > .env << 'EOF'
# Database (local Docker)
DATABASE_URL=postgresql+asyncpg://docuser:docpass@localhost:5432/docplatform

# Redis (local Docker)
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# Organization
ORG_ID=bf0d03fb-d8ea-4377-a991-b3b5818e71ec

# API
API_BASE_URL=http://localhost:8000

# Logging
LOG_LEVEL=INFO
EOF
    fi

    print_success ".env file created"
else
    print_success ".env file exists"
fi

# Step 4: Initialize database
print_step "Initializing database..."

if [ -f "scripts/init_db.py" ]; then
    python3 -m scripts.init_db
else
    python3 -c "import asyncio; from app.db.session import init_db; asyncio.run(init_db())"
fi

print_success "Database initialized"

# Summary
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║${NC}  ${GREEN}Setup Complete!${NC}                                         ${GREEN}║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "To start the application, run ${BLUE}3 separate terminals${NC}:"
echo ""
echo -e "  ${YELLOW}Terminal 1 - API Backend:${NC}"
echo -e "    ${BLUE}make api${NC}"
echo -e "    or: ${BLUE}uvicorn app.main:app --reload${NC}"
echo ""
echo -e "  ${YELLOW}Terminal 2 - Celery Worker:${NC}"
echo -e "    ${BLUE}make worker${NC}"
echo -e "    or: ${BLUE}celery -A app.worker.celery_app worker --loglevel=info${NC}"
echo ""
echo -e "  ${YELLOW}Terminal 3 - Streamlit Frontend:${NC}"
echo -e "    ${BLUE}make frontend${NC}"
echo -e "    or: ${BLUE}streamlit run frontend/app.py${NC}"
echo ""
echo -e "Then open: ${BLUE}http://localhost:8501${NC}"
echo ""
echo -e "For all commands, run: ${BLUE}make help${NC}"
echo ""
