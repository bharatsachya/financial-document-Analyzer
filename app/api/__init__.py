"""FastAPI routers and dependencies."""

from app.api.deps import get_db, get_org_id
from app.api.templates import router as templates_router

__all__ = [
    "get_db",
    "get_org_id",
    "templates_router",
]
