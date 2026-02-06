.PHONY: help install dev db api worker frontend lint test clean

help: ## Show this help message
	@echo "Document Intelligence Platform - Development Commands"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	poetry install

dev: ## Start all services in development mode
	docker-compose up -d
	@echo "Infrastructure services started. Run 'make api', 'make worker', and 'make frontend' in separate terminals."

db: ## Initialize database
	python -m scripts.init_db

api: ## Start FastAPI backend
	poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

worker: ## Start Celery worker
	poetry run celery -A app.worker.celery_app worker --loglevel=info

frontend: ## Start Streamlit frontend
	poetry run streamlit run frontend/app.py

lint: ## Run linter
	ruff check .

format: ## Format code
	ruff check --fix .
	black .

typecheck: ## Run type checker
	mypy .

test: ## Run all tests
	pytest

test-unit: ## Run unit tests
	pytest tests/unit

test-integration: ## Run integration tests
	pytest tests/integration

clean: ## Clean up generated files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf .pytest_cache .ruff_cache .mypy_cache dist build *.egg-info

down: ## Stop all services
	docker-compose down

logs: ## Show Docker logs
	docker-compose logs -f

ps: ## Show running containers
	docker-compose ps
