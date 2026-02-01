"""Streamlit frontend for Document Intelligence Platform.

Provides a UI for uploading documents and tracking processing status.
"""

import io
import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
ORG_ID = os.getenv("ORG_ID", "bf0d03fb-d8ea-4377-a991-b3b5818e71ec")  # Default org ID matching database

# Page config
st.set_page_config(
    page_title="Document Intelligence Platform",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# API Client
# =============================================================================


class APIClient:
    """Simple async-style API client for the Streamlit frontend."""

    def __init__(self, base_url: str, org_id: str):
        """Initialize the API client.

        Args:
            base_url: Base URL of the API.
            org_id: Organization ID for multi-tenancy.
        """
        self.base_url = base_url.rstrip("/")
        self.org_id = org_id
        self.headers = {
            "X-Org-ID": org_id,
        }

    def upload_document(self, file: UploadedFile) -> dict[str, Any]:
        """Upload a document for processing.

        Args:
            file: The uploaded file from Streamlit.

        Returns:
            API response dict.
        """
        url = f"{self.base_url}/ingest/upload"

        # Prepare files for upload
        files = {"file": (file.name, file.getvalue(), file.type)}

        try:
            response = httpx.post(url, headers=self.headers, files=files, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Upload failed: {e.response.status_code} - {e.response.text}")
            st.error(f"Upload failed: {e.response.status_code}")
            return {}
        except Exception as e:
            logger.error(f"Upload error: {e}")
            st.error(f"Upload error: {e}")
            return {}

    def get_document_status(self, page: int = 1, page_size: int = 20) -> dict[str, Any]:
        """Get status of documents.

        Args:
            page: Page number.
            page_size: Results per page.

        Returns:
            API response dict with documents.
        """
        url = f"{self.base_url}/ingest/status"
        params = {"page": page, "page_size": page_size}

        try:
            response = httpx.get(url, headers=self.headers, params=params, timeout=10.0)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Status fetch failed: {e.response.status_code}")
            return {"documents": [], "total": 0, "page": page, "page_size": page_size}
        except Exception as e:
            logger.error(f"Status fetch error: {e}")
            return {"documents": [], "total": 0, "page": page, "page_size": page_size}

    def health_check(self) -> bool:
        """Check if the API is healthy.

        Returns:
            True if API is healthy, False otherwise.
        """
        try:
            response = httpx.get(f"{self.base_url}/health", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False


# =============================================================================
# UI Components
# =============================================================================


def render_sidebar(client: APIClient) -> None:
    """Render the sidebar with org info and connection status.

    Args:
        client: The API client instance.
    """
    with st.sidebar:
        st.title("📄 Doc Intelligence")

        st.divider()

        # Connection status
        is_healthy = client.health_check()
        if is_healthy:
            st.success("✅ API Connected")
        else:
            st.error("❌ API Disconnected")
            st.info(f"API URL: {API_BASE_URL}")

        st.divider()

        # Organization info
        st.subheader("Organization")
        st.text_input(
            "Organization ID",
            value=client.org_id,
            disabled=True,
            help="Your organization ID for multi-tenancy isolation",
        )

        st.divider()

        # Instructions
        st.subheader("Instructions")
        st.markdown("""
        1. **Upload Documents**: Drag & drop or browse for files
        2. **Track Progress**: Monitor processing status in real-time
        3. **Supported Formats**: PDF, DOCX, TXT, MD

        Documents are processed asynchronously:
        - Queued → Parsing → Chunking → Embedding → Indexed
        """)

        st.divider()

        # Settings link
        st.caption(f"API: `{API_BASE_URL}`")


def render_upload_section(client: APIClient) -> None:
    """Render the file upload section.

    Args:
        client: The API client instance.
    """
    st.subheader("📤 Upload Document")

    col1, col2 = st.columns([3, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["pdf", "docx", "txt", "md"],
            label_visibility="collapsed",
            help="Supported formats: PDF, DOCX, TXT, Markdown",
        )

    with col2:
        st.write("")
        st.write("")
        upload_button = st.button("Upload", type="primary", use_container_width=True)

    if uploaded_file and upload_button:
        with st.spinner("Uploading document..."):
            result = client.upload_document(uploaded_file)

            if result:
                st.success(f"Document uploaded! Task ID: `{result.get('task_id', 'N/A')}`")
                st.rerun()


def render_status_table(client: APIClient) -> None:
    """Render the document status table.

    Args:
        client: The API client instance.
    """
    st.subheader("📊 Document Status")

    # Fetch status
    status_data = client.get_document_status(page=1, page_size=50)

    if not status_data.get("documents"):
        st.info("No documents found. Upload a document to get started.")
        return

    # Display stats
    col1, col2, col3, col4 = st.columns(4)
    documents = status_data.get("documents", [])

    status_counts = {}
    for doc in documents:
        status = doc.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

    with col1:
        st.metric("Total", status_data.get("total", 0))

    with col2:
        st.metric("Queued", status_counts.get("queued", 0))

    with col3:
        completed = status_counts.get("completed", 0)
        st.metric("Completed", completed, delta_color="normal" if completed > 0 else "off")

    with col4:
        failed = status_counts.get("failed", 0)
        st.metric("Failed", failed, delta_color="inverse" if failed > 0 else "normal")

    st.divider()

    # Status badges
    def status_badge(status: str) -> str:
        """Return an emoji/status badge for a document status."""
        badges = {
            "queued": "🔄 Queued",
            "parsing": "📖 Parsing",
            "chunking": "✂️ Chunking",
            "embedding": "🔢 Embedding",
            "indexing": "📇 Indexing",
            "completed": "✅ Completed",
            "failed": "❌ Failed",
        }
        return badges.get(status.lower(), status)

    # Prepare table data
    table_data = []
    for doc in documents:
        created_at = doc.get("created_at", "")
        if created_at:
            try:
                dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                created_at = dt.strftime("%Y-%m-%d %H:%M")
            except Exception:
                pass

        table_data.append({
            "Filename": doc.get("filename", "unknown")[:30],
            "Status": status_badge(doc.get("status", "unknown")),
            "Chunks": doc.get("chunk_count") or "-",
            "Vectors": doc.get("vector_count") or "-",
            "Uploaded": created_at,
        })

    st.dataframe(
        table_data,
        use_container_width=True,
        hide_index=True,
    )


def render_document_detail(client: APIClient, document_id: str) -> None:
    """Render detailed view for a specific document.

    Args:
        client: The API client instance.
        document_id: The document ID to view.
    """
    try:
        url = f"{client.base_url}/ingest/status/{document_id}"
        response = httpx.get(url, headers=client.headers, timeout=10.0)
        response.raise_for_status()
        doc = response.json()

        st.subheader(f"Document: {doc.get('filename')}")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Status:**", status_badge(doc.get("status")))
            st.write("**Uploaded:**", doc.get("created_at"))
            st.write("**File size:**", f"{doc.get('file_size', 0):,} bytes")

        with col2:
            st.write("**Chunks:**", doc.get("chunk_count") or "Processing...")
            st.write("**Vectors:**", doc.get("vector_count") or "Processing...")
            st.write("**MIME type:**", doc.get("mime_type") or "unknown")

        if doc.get("error_message"):
            st.error(f"Error: {doc['error_message']}")

    except Exception as e:
        st.error(f"Failed to load document: {e}")


def status_badge(status: str | None) -> str:
    """Return a status badge with emoji."""
    if not status:
        return "❓ Unknown"

    badges = {
        "queued": "🔄 Queued",
        "parsing": "📖 Parsing",
        "chunking": "✂️ Chunking",
        "embedding": "🔢 Embedding",
        "indexing": "📇 Indexing",
        "completed": "✅ Completed",
        "failed": "❌ Failed",
    }
    return badges.get(status.lower(), f"❓ {status}")


# =============================================================================
# Main App
# =============================================================================


def main() -> None:
    """Main application entry point."""

    # Initialize API client
    client = APIClient(API_BASE_URL, ORG_ID)

    # Render sidebar
    render_sidebar(client)

    # Main content area
    st.title("Document Ingestion Dashboard")

    # Tab navigation
    tab1, tab2 = st.tabs(["Upload & Monitor", "Settings"])

    with tab1:
        # Upload section
        render_upload_section(client)

        st.divider()

        # Status table
        render_status_table(client)

    with tab2:
        st.subheader("⚙️ Settings")

        st.write("**API Configuration**")
        st.text_input("API Base URL", value=API_BASE_URL, disabled=True)

        st.write("**Organization**")
        st.text_input("Organization ID", value=ORG_ID, disabled=True)

        st.write("**Strategy Configuration**")
        st.info("Strategies are configured server-side via environment variables.")

        st.write("**Current Strategies**")
        st.json({
            "parser": "llama_parse",
            "chunker": "markdown",
            "embedder": "openai",
            "vector_store": "qdrant",
        })


if __name__ == "__main__":
    main()
