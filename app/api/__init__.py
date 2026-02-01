"""FastAPI routers and dependencies."""

from app.api.deps import (
    get_current_active_user,
    get_current_user,
    get_db,
    get_org_id,
)
from app.api.ingest import router as ingest_router
from app.api.organizations import router as organizations_router
from app.api.templates import router as templates_router
from app.api.users import router as users_router

__all__ = [
    "get_current_user",
    "get_current_active_user",
    "get_db",
    "get_org_id",
    "ingest_router",
    "organizations_router",
    "templates_router",
    "users_router",
]
