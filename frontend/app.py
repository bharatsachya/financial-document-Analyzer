"""Streamlit frontend for Document Intelligence Platform.

Provides a unified UI for both Document Ingestion and Template Engine.
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
ORG_ID = os.getenv("ORG_ID", "bf0d03fb-d8ea-4377-a991-b3b5818e71ec")

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
# API Clients
# =============================================================================


class APIClient:
    """API client for document ingestion endpoints."""

    def __init__(self, base_url: str, org_id: str):
        self.base_url = base_url.rstrip("/")
        self.org_id = org_id
        self.headers = {"X-Org-ID": org_id}

    def upload_document(self, file: UploadedFile) -> dict[str, Any]:
        url = f"{self.base_url}/ingest/upload"
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
        try:
            response = httpx.get(f"{self.base_url}/health", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False


class TemplateAPIClient:
    """API client for template engine endpoints."""

    def __init__(self, base_url: str, org_id: str):
        self.base_url = base_url.rstrip("/")
        self.org_id = org_id
        self.headers = {"X-Org-ID": org_id}

    def health_check(self) -> bool:
        try:
            response = httpx.get(f"{self.base_url}/health", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False

    def analyze_template(self, file: UploadedFile) -> dict[str, Any] | None:
        """Analyze a Word template to detect dynamic variables."""
        url = f"{self.base_url}/templates/analyze"
        files = {"file": (file.name, file.getvalue(), file.type)}
        try:
            # Progress bar for analysis
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Step 1: Uploading file
            progress_bar.progress(10)
            status_text.text("📤 Uploading file...")
            time.sleep(0.2)

            # Step 2: Analyzing paragraphs
            progress_bar.progress(30)
            status_text.text("🔍 Analyzing paragraphs...")

            response = httpx.post(url, headers=self.headers, files=files, timeout=60.0)
            response.raise_for_status()

            # Step 3: Finalizing results
            progress_bar.progress(80)
            status_text.text("✅ Finalizing results...")
            time.sleep(0.2)

            progress_bar.progress(100)
            status_text.text("Complete!")
            time.sleep(0.3)

            progress_bar.empty()
            status_text.empty()

            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Analysis failed: {e.response.status_code} - {e.response.text}")
            st.error(f"Analysis failed: {e.response.status_code}")
            st.error(e.response.text)
            return None
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            st.error(f"Analysis error: {e}")
            return None

    def finalize_template(
        self,
        template_id: str,
        variables: list[dict[str, Any]],
        original_filename: str,
    ) -> dict[str, Any] | None:
        """Finalize template by injecting Jinja2 tags."""
        url = f"{self.base_url}/templates/finalize"
        payload = {
            "template_id": template_id,
            "variables": variables,
            "original_filename": original_filename,
        }
        try:
            # Progress bar for finalization
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Step 1: Preparing injection
            progress_bar.progress(10)
            status_text.text("🔧 Preparing Jinja2 tag injection...")
            time.sleep(0.2)

            # Step 2: Injecting tags
            progress_bar.progress(30)
            status_text.text(f"💉 Injecting {len(variables)} variables...")

            response = httpx.post(url, headers=self.headers, json=payload, timeout=30.0)
            response.raise_for_status()

            # Step 3: Processing document
            progress_bar.progress(70)
            status_text.text("📄 Processing document...")
            time.sleep(0.2)

            # Step 4: Saving template
            progress_bar.progress(90)
            status_text.text("💾 Saving template...")
            time.sleep(0.2)

            progress_bar.progress(100)
            status_text.text("Complete!")
            time.sleep(0.3)

            progress_bar.empty()
            status_text.empty()

            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Finalization failed: {e.response.status_code} - {e.response.text}")
            st.error(f"Finalization failed: {e.response.status_code}")
            st.error(e.response.text)
            return None
        except Exception as e:
            logger.error(f"Finalization error: {e}")
            st.error(f"Finalization error: {e}")
            return None

    def download_template(self, template_id: str) -> bytes | None:
        """Download the finalized template."""
        url = f"{self.base_url}/templates/download/{template_id}"
        try:
            response = httpx.get(url, headers=self.headers, timeout=30.0)
            response.raise_for_status()
            return response.content
        except httpx.HTTPStatusError as e:
            logger.error(f"Download failed: {e.response.status_code}")
            st.error(f"Download failed: {e.response.status_code}")
            return None
            return None
        except Exception as e:
            logger.error(f"Download error: {e}")
            st.error(f"Download error: {e}")
            return None

    def analyze_batch(self, files: list[UploadedFile]) -> str | None:
        """Upload multiple files for batch processing.

        Args:
            files: List of uploaded files.

        Returns:
            batch_id string, or None if failed.
        """
        url = f"{self.base_url}/templates/analyze-batch"

        try:
            # Prepare multipart files
            files_data = [
                ("files", (f.name, f.getvalue(), f.type))
                for f in files
            ]

            with st.spinner(f"Uploading {len(files)} templates..."):
                response = httpx.post(url, headers=self.headers, files=files_data, timeout=120.0)
                response.raise_for_status()
                result = response.json()
                return result.get("batch_id")
        except httpx.HTTPStatusError as e:
            logger.error(f"Batch upload failed: {e.response.status_code} - {e.response.text}")
            st.error(f"Batch upload failed: {e.response.status_code}")
            return None
        except Exception as e:
            logger.error(f"Batch upload error: {e}")
            st.error(f"Batch upload error: {e}")
            return None

    def queue_next_batch(self, files: list[UploadedFile], current_batch_id: str) -> str | None:
        """Queue files for next batch (auto-starts when current completes).

        Args:
            files: List of uploaded files.
            current_batch_id: The currently processing batch ID.

        Returns:
            next_batch_id string, or None if failed.
        """
        url = f"{self.base_url}/templates/queue-next-batch?current_batch_id={current_batch_id}"

        try:
            files_data = [
                ("files", (f.name, f.getvalue(), f.type))
                for f in files
            ]

            with st.spinner(f"Queuing {len(files)} templates for next batch..."):
                response = httpx.post(url, headers=self.headers, files=files_data, timeout=120.0)
                response.raise_for_status()
                result = response.json()
                return result.get("batch_id")
        except httpx.HTTPStatusError as e:
            logger.error(f"Queue next batch failed: {e.response.status_code} - {e.response.text}")
            st.error(f"Queue next batch failed: {e.response.status_code}")
            return None
        except Exception as e:
            logger.error(f"Queue next batch error: {e}")
            st.error(f"Queue next batch error: {e}")
            return None

    def get_batch_status(self, batch_id: str) -> dict | None:
        """Get status for all templates in a batch.

        Args:
            batch_id: The batch ID to check.

        Returns:
            Batch status dict with aggregate stats and template list, or None if failed.
        """
        url = f"{self.base_url}/templates/batch-status/{batch_id}"

        try:
            response = httpx.get(url, headers=self.headers, timeout=10.0)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Batch status check failed: {e.response.status_code}")
            return None
        except Exception as e:
            logger.error(f"Batch status error: {e}")
            return None

    def get_ready_for_injection(self) -> list[dict] | None:
        """Get templates ready for variable injection.

        Returns:
            List of templates with status=completed and download_ready=False.
        """
        url = f"{self.base_url}/templates/list-ready"
        params = {"status": "completed", "page_size": 100}  # Fetch plenty
        
        try:
            response = httpx.get(url, headers=self.headers, params=params, timeout=10.0)
            response.raise_for_status()
            data = response.json()
            return data.get("templates", [])
        except httpx.HTTPStatusError as e:
            logger.error(f"Failed to fetch templates for injection: {e.response.status_code}")
            return None
        except Exception as e:
            logger.error(f"Error fetching templates for injection: {e}")
            return None
            
    def get_ready_for_download(self) -> list[dict] | None:
        """Get templates ready for download.

        Returns:
            List of templates with download_ready=True.
        """
        url = f"{self.base_url}/templates/list-ready"
        params = {"download_ready": True, "page_size": 100}
        
        try:
            response = httpx.get(url, headers=self.headers, params=params, timeout=10.0)
            response.raise_for_status()
            data = response.json()
            templates = data.get("templates", [])

            # Ensure download_url is complete with org_id for browser downloads
            for t in templates:
                if t.get("download_url") and not t.get("download_url").startswith("http"):
                    base_url = f"{self.base_url}{t['download_url']}"
                    # Add org_id as query parameter for browser downloads (browsers can't send custom headers)
                    t["download_url"] = f"{base_url}?org_id={self.org_id}"

            return templates
        except httpx.HTTPStatusError as e:
            logger.error(f"Failed to fetch templates for download: {e.response.status_code}")
            return None
        except Exception as e:
            logger.error(f"Error fetching templates for download: {e}")
            return None

    def inject_random_values(self, template_id: str) -> bool:
        """Inject random values and queue finalization.

        Args:
            template_id: The template ID to inject values into.

        Returns:
            True if queued successfully, False otherwise.
        """
        url = f"{self.base_url}/templates/inject-random/{template_id}"

        try:
            with st.spinner("Generating random values and queuing injection..."):
                response = httpx.post(url, headers=self.headers, timeout=30.0)
                response.raise_for_status()
                return True
        except httpx.HTTPStatusError as e:
            logger.error(f"Random injection failed: {e.response.status_code} - {e.response.text}")
            st.error(f"Random injection failed: {e.response.status_code}")
            return False
        except Exception as e:
            logger.error(f"Random injection error: {e}")
            st.error(f"Random injection error: {e}")
            return False

    def finalize_template_async(
        self,
        template_id: str,
        variables: list[dict],
    ) -> str | None:
        """Queue template finalization with variable injection (async).

        Args:
            template_id: The template ID to finalize.
            variables: Variables with values to inject.

        Returns:
            task_id if queued successfully, None otherwise.
        """
        url = f"{self.base_url}/templates/finalize-async"

        payload = {
            "template_id": template_id,
            "variables": variables,
        }

        try:
            with st.spinner("Queueing template finalization..."):
                response = httpx.post(url, headers=self.headers, json=payload, timeout=30.0)
                response.raise_for_status()
                return response.json().get("task_id")
        except httpx.HTTPStatusError as e:
            logger.error(f"Async finalization failed: {e.response.status_code} - {e.response.text}")
            st.error(f"Finalization failed: {e.response.status_code}")
            return None
        except Exception as e:
            logger.error(f"Async finalization error: {e}")
            st.error(f"Finalization error: {e}")
            return None
    def get_injection_queue(self) -> dict | None:
        """Get injection queue status.

        Returns:
            Dict with queue stats and jobs.
        """
        url = f"{self.base_url}/templates/injection-queue"

        try:
            response = httpx.get(url, headers=self.headers, timeout=10.0)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Queue status check failed: {e.response.status_code}")
            return None
        except Exception as e:
            logger.error(f"Queue status error: {e}")
            return None

    def save_template(
        self,
        template_id: str,
        variables: list[dict[str, Any]],
        original_filename: str,
        name: str | None = None,
        description: str | None = None,
        paragraph_count: int | None = None,
    ) -> dict[str, Any] | None:
        """Save an analyzed template to storage for later use.

        Args:
            template_id: The template ID from analysis.
            variables: List of detected variables.
            original_filename: Original filename.
            name: Custom name for the template.
            description: Template description.
            paragraph_count: Total paragraph count.

        Returns:
            Saved template data or None if failed.
        """
        url = f"{self.base_url}/templates/save"
        payload = {
            "template_id": template_id,
            "variables": variables,
            "original_filename": original_filename,
            "name": name,
            "description": description,
            "paragraph_count": paragraph_count,
        }
        try:
            response = httpx.post(url, headers=self.headers, json=payload, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Save template failed: {e.response.status_code} - {e.response.text}")
            st.error(f"Save template failed: {e.response.status_code}")
            return None
        except Exception as e:
            logger.error(f"Save template error: {e}")
            st.error(f"Save template error: {e}")
            return None

    def get_stored_templates(self, page: int = 1, page_size: int = 100) -> list[dict[str, Any]]:
        """Get list of stored templates.

        Args:
            page: Page number.
            page_size: Items per page.

        Returns:
            List of stored templates.
        """
        url = f"{self.base_url}/templates/stored"
        params = {"page": page, "page_size": page_size}

        try:
            response = httpx.get(url, headers=self.headers, params=params, timeout=10.0)
            response.raise_for_status()
            data = response.json()

            templates = data.get("templates", [])
            logger.info(f"Retrieved {len(templates)} stored templates")
            return templates
        except httpx.HTTPStatusError as e:
            logger.error(f"Fetch stored templates failed: {e.response.status_code}")
            return []
        except Exception as e:
            logger.error(f"Fetch stored templates error: {e}")
            return []

    def get_stored_template(self, template_id: str) -> dict[str, Any] | None:
        """Get a specific stored template.

        Args:
            template_id: The template ID.

        Returns:
            Template data or None if not found.
        """
        url = f"{self.base_url}/templates/stored/{template_id}"

        try:
            response = httpx.get(url, headers=self.headers, timeout=10.0)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            logger.error(f"Fetch template failed: {e.response.status_code}")
            return None
        except Exception as e:
            logger.error(f"Fetch template error: {e}")
            return None

    def get_template_status(self, template_id: str) -> dict[str, Any] | None:
        """Get status for a specific template.

        Args:
            template_id: The template ID to check.

        Returns:
            Template status dict or None if failed.
        """
        url = f"{self.base_url}/templates/status/{template_id}"

        try:
            response = httpx.get(url, headers=self.headers, timeout=10.0)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Template status check failed: {e.response.status_code}")
            return None
        except Exception as e:
            logger.error(f"Template status error: {e}")
            return None

    def get_default_prompt(self) -> dict[str, Any]:
        """Get the default analysis prompt.

        Returns:
            Dict with default prompt data.
        """
        url = f"{self.base_url}/templates/prompts/default"

        try:
            response = httpx.get(url, headers=self.headers, timeout=10.0)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Get default prompt failed: {e.response.status_code}")
        except Exception as e:
            logger.error(f"Get default prompt error: {e}")

        # Fallback to system default
        return {
            "id": "system_default",
            "name": "System Default",
            "source": "system",
            "is_default": True,
            "prompt_text": """You are an expert Document Intelligence Engineer. Analyze the text segment and identify potential variables that should be replaced with dynamic values. Look for patterns like:
- Names (person, company, client)
- Dates (birth, incorporation, expiry)
- Addresses
- Monetary amounts
- Percentages
- Phone numbers
- Email addresses

For each detected variable, suggest a meaningful variable name in snake_case format."""
        }

    def get_prompts(self) -> list[dict[str, Any]]:
        """Get all saved custom prompts.

        Returns:
            List of saved prompts.
        """
        url = f"{self.base_url}/templates/prompts"

        try:
            response = httpx.get(url, headers=self.headers, timeout=10.0)
            response.raise_for_status()
            data = response.json()
            return data.get("prompts", [])
        except httpx.HTTPStatusError as e:
            logger.error(f"Get prompts failed: {e.response.status_code}")
        except Exception as e:
            logger.error(f"Get prompts error: {e}")

        return []

    def delete_prompt(self, prompt_id: str) -> bool:
        """Delete a saved prompt.

        Args:
            prompt_id: The prompt ID to delete.

        Returns:
            True if deleted, False otherwise.
        """
        url = f"{self.base_url}/templates/prompts/{prompt_id}"

        try:
            response = httpx.delete(url, headers=self.headers, timeout=10.0)
            response.raise_for_status()
            return True
        except httpx.HTTPStatusError as e:
            logger.error(f"Delete prompt failed: {e.response.status_code}")
        except Exception as e:
            logger.error(f"Delete prompt error: {e}")

        return False

    def save_prompt(
        self,
        name: str,
        prompt_text: str,
        set_as_default: bool = False,
    ) -> dict[str, Any] | None:
        """Save a custom extraction prompt.

        Args:
            name: Name for the prompt.
            prompt_text: The prompt text content.
            set_as_default: Whether to set as default prompt.

        Returns:
            Response dict or None.
        """
        url = f"{self.base_url}/templates/prompts"

        payload = {
            "name": name,
            "prompt_text": prompt_text,
            "set_as_default": set_as_default,
        }

        try:
            response = httpx.post(url, headers=self.headers, json=payload, timeout=10.0)
            response.raise_for_status()
            data = response.json()
            logger.info(f"Prompt saved successfully: {data.get('id')}")
            return data
        except httpx.HTTPStatusError as e:
            error_detail = e.response.text
            try:
                error_json = e.response.json()
                error_detail = error_json.get("detail", error_detail)
            except:
                pass
            logger.error(f"Save prompt failed: {e.response.status_code} - {error_detail}")
            st.error(f"Save prompt failed ({e.response.status_code}): {error_detail}")
        except Exception as e:
            logger.error(f"Save prompt error: {e}", exc_info=True)
            st.error(f"Save prompt error: {e}")

        return None


# =============================================================================
# UI Components - Document Ingestion
# =============================================================================


def render_ingestion_upload(client: APIClient) -> None:
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


def render_ingestion_status(client: APIClient) -> None:
    st.subheader("📊 Document Status")

    status_data = client.get_document_status(page=1, page_size=50)

    if not status_data.get("documents"):
        st.info("No documents found. Upload a document to get started.")
        return

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

    def status_badge(status: str) -> str:
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

    st.dataframe(table_data, use_container_width=True, hide_index=True)


# =============================================================================
# UI Components - Template Engine
# =============================================================================


def init_template_session_state() -> None:
    """Initialize template engine session state."""
    if "template_uploaded_file" not in st.session_state:
        st.session_state.template_uploaded_file = None
    if "template_analysis_result" not in st.session_state:
        st.session_state.template_analysis_result = None
    if "template_variables" not in st.session_state:
        st.session_state.template_variables = None
    if "template_client_data" not in st.session_state:
        st.session_state.template_client_data = {}
    if "finalized_template_id" not in st.session_state:
        st.session_state.finalized_template_id = None
    if "show_template_download" not in st.session_state:
        st.session_state.show_template_download = False
    if "saved_template_id" not in st.session_state:
        st.session_state.saved_template_id = None
    # Stored template selection state
    if "selected_template" not in st.session_state:
        st.session_state.selected_template = None
    # Legacy inject engine session state
    if "selected_doc_for_inject" not in st.session_state:
        st.session_state.selected_doc_for_inject = None
    if "inject_analysis_result" not in st.session_state:
        st.session_state.inject_analysis_result = None
    if "inject_uploaded_file" not in st.session_state:
        st.session_state.inject_uploaded_file = None
    if "current_batch_id" not in st.session_state:
        st.session_state.current_batch_id = None


def render_template_upload(client: TemplateAPIClient) -> None:
    st.subheader("📤 Step 1: Upload Templates")

    # Check if there's an active batch
    current_batch_id = st.session_state.get("current_batch_id")
    batch_status = None
    if current_batch_id:
        batch_status = client.get_batch_status(current_batch_id)

    # Show status message about current/next batch
    if current_batch_id and batch_status:
        in_progress = batch_status.get("in_progress", 0)
        queued = batch_status.get("queued", 0)
        next_batch_id = batch_status.get("next_batch_id")

        if in_progress > 0:
            st.info(f"📦 **Current batch processing**: `{current_batch_id}`")
            st.caption("New uploads will be queued for the next batch")
        elif next_batch_id:
            st.info(f"📋 **Next batch queued**: `{next_batch_id}`")
            st.caption(f"{queued} file(s) waiting to process")

    # File uploader with multi-file support
    st.caption("**💡 Tip:** Use Cmd+Click (Mac) or Ctrl+Click (Windows) to select multiple files at once.")
    uploaded_files = st.file_uploader(
        "Choose Word templates",
        type=["docx"],
        accept_multiple_files=True,
        help="Select one or more .docx files. Hold Cmd/Ctrl to select multiple files.",
    )

    # Show count of selected files
    if uploaded_files:
        st.info(f"📎 **{len(uploaded_files)}** file(s) selected: {', '.join([f.name for f in uploaded_files])}")

    # Upload button
    if uploaded_files and st.button("🚀 Upload & Analyze", type="primary"):
        # Determine if we should queue as next batch or create new batch
        if current_batch_id and batch_status and batch_status.get("in_progress", 0) > 0:
            # Queue as next batch (auto-starts when current completes)
            next_batch_id = client.queue_next_batch(uploaded_files, current_batch_id)
            if next_batch_id:
                st.success(f"✅ Files queued for next batch: `{next_batch_id}`")
                st.rerun()
        else:
            # Create new batch
            batch_id = client.analyze_batch(uploaded_files)
            if batch_id:
                st.session_state.current_batch_id = batch_id
                st.success(f"✅ Files uploaded! Batch ID: `{batch_id}`")
                st.rerun()

    # Display processing status if batch is active
    if current_batch_id and batch_status:
        st.divider()
        st.subheader("🔄 Processing Status")

        total = batch_status.get("total_templates", 0)
        completed = batch_status.get("completed", 0)
        failed = batch_status.get("failed", 0)
        in_progress = batch_status.get("in_progress", 0)
        queued = total - completed - failed - in_progress

        if total > 0:
            progress = (completed + failed) / total
            
            # Auto-refresh if still processing
            if (completed + failed) < total:
                # Add animated CSS for progress indicator
                st.markdown(f"""
                <style>
                @keyframes pulse {{
                    0% {{ opacity: 0.6; }}
                    50% {{ opacity: 1; }}
                    100% {{ opacity: 0.6; }}
                }}
                .processing-indicator {{
                    animation: pulse 1.5s infinite;
                    background: linear-gradient(90deg, #4CAF50, #8BC34A);
                    padding: 10px 20px;
                    border-radius: 8px;
                    color: white;
                    font-weight: bold;
                    display: inline-block;
                    margin-bottom: 10px;
                }}
                @keyframes progress-animation {{
                    0% {{ background-position: 0% 50%; }}
                    100% {{ background-position: 100% 50%; }}
                }}
                .animated-bar {{
                    background: linear-gradient(90deg, #667eea, #764ba2, #667eea);
                    background-size: 200% 100%;
                    animation: progress-animation 2s linear infinite;
                    height: 8px;
                    border-radius: 4px;
                    margin: 5px 0;
                }}
                </style>
                <div class="processing-indicator">🔄 Processing {completed}/{total} templates...</div>
                """, unsafe_allow_html=True)
                
                # Show status breakdown
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("✅ Completed", completed)
                with col2:
                    st.metric("⚙️ In Progress", in_progress)
                with col3:
                    st.metric("⏳ Queued", queued)
                with col4:
                    st.metric("❌ Failed", failed)
                
                # Animated progress bar
                st.markdown('<div class="animated-bar"></div>', unsafe_allow_html=True)
                st.progress(progress)
                
                # Auto-refresh every 3 seconds
                import time
                time.sleep(3)
                st.rerun()
            else:
                st.success("✨ Batch processing complete!")
                st.progress(1.0, text=f"{total}/{total} processed")
                st.caption(f"✅ {completed} completed | ❌ {failed} failed")
                if st.button("Clear", key="clear_batch"):
                    st.session_state.current_batch_id = None
                    st.rerun()


def render_template_variables(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Render detected variables and allow editing."""
    st.subheader("🔍 Step 2: Review & Edit Variables")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Variables Detected", len(result.get("detected_variables", [])))
    with col2:
        st.metric("Total Paragraphs", result.get("total_paragraphs", 0))
    with col3:
        st.metric("Template ID", result.get("template_id", "")[:8] + "...")

    st.divider()

    variables = result.get("detected_variables", [])
    if not variables:
        st.warning("No variables detected in the template.")
        return []

    st.info(
        "💡 **Tip**: Review the detected variables below. "
        "You can edit variable names or remove false positives."
    )

    edited_vars = []
    for i, var in enumerate(variables):
        with st.expander(
            f"Variable {i + 1}: `{var.get('original_text', '')[:40]}...`",
            expanded=i < 3,
        ):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.write("**Original Text:**")
                st.code(var.get("original_text", ""), language="text")

                new_name = st.text_input(
                    "Variable Name",
                    value=var.get("suggested_variable_name", ""),
                    key=f"template_name_{i}",
                    help="Snake_case variable name for Jinja2",
                )

            with col2:
                keep = st.checkbox(
                    "Keep in Template",
                    value=True,
                    key=f"template_keep_{i}",
                )
                st.write("**Paragraph:**", var.get("paragraph_index"))
                st.write("**Rationale:**", var.get("rationale", "")[:50] + "...")

            if keep:
                edited_vars.append({
                    "original_text": var.get("original_text", ""),
                    "suggested_variable_name": new_name,
                    "rationale": var.get("rationale", ""),
                    "paragraph_index": var.get("paragraph_index", 0),
                })

    st.divider()
    st.info(f"✅ **{len(edited_vars)} variables** will be injected into the template.")

    return edited_vars


def render_client_data_input(variables: list[dict[str, Any]]) -> dict[str, Any]:
    """Render client data input form for each variable with context snippets."""
    st.subheader("👤 Step 3: Provide Client Data")

    st.info(
        "📝 Enter the client data that will be used to render the final document. "
        "Click on each variable to see the context snippet from your document."
    )

    st.divider()

    client_data = {}

    # Display each variable in an expander with context
    for i, var in enumerate(variables):
        var_name = var.get("suggested_variable_name", "")
        original_text = var.get("original_text", "")
        paragraph_index = var.get("paragraph_index", 0)
        default = st.session_state.template_client_data.get(var_name, "")

        with st.expander(
            f"📝 {var_name.replace('_', ' ').title()} - {original_text[:40]}...",
            expanded=i < 3,  # Expand first 3 by default
        ):
            col1, col2 = st.columns([2, 1])

            with col1:
                # Input field
                client_data[var_name] = st.text_input(
                    "Replacement Value",
                    value=default,
                    key=f"client_data_{var_name}",
                    help="Enter the value to replace this variable",
                )

            with col2:
                st.write("**Detected:**")
                st.code(original_text, language="text")
                st.caption(f"Paragraph {paragraph_index}")

            # Context snippet - helps user understand where this appears
            st.info(f"📍 **Context:** \"...{original_text}...\"")
            st.caption(
                "_The context snippet shows where this value appears in your document. "
                "Use this to identify which value needs to be replaced._"
            )

    # Store client data in session state
    st.session_state.template_client_data = client_data

    return client_data


def render_template_finalize(
    client: TemplateAPIClient,
    result: dict[str, Any],
    variables: list[dict[str, Any]],
) -> bool:
    """Render finalize and download section."""
    st.subheader("✨ Step 4: Generate & Download Template")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Variables to Inject", len(variables))
    with col2:
        st.metric("Output Format", "Jinja2 .docx")

    st.divider()

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("🎯 Generate Tagged Template", type="primary", use_container_width=True):
            if not variables:
                st.error("❌ No variables to inject.")
                return False

            finalize_result = client.finalize_template(
                template_id=result.get("template_id", ""),
                variables=variables,
                original_filename=result.get("filename", ""),
            )

            if finalize_result:
                st.success(f"✅ Template generated successfully!")
                st.info(f"📥 Template ID: `{finalize_result['template_id']}`")
                st.session_state.finalized_template_id = finalize_result["template_id"]
                st.session_state.finalized_filename = result.get("filename", "").replace(
                    ".docx", "_tagged.docx"
                )
                st.session_state.show_template_download = True
                st.rerun()

    with col2:
        if st.session_state.get("show_template_download", False):
            if st.button("📥 Download Template", type="secondary", use_container_width=True):
                template_id = st.session_state.get("finalized_template_id")
                if template_id:
                    content = client.download_template(template_id)
                    if content:
                        filename = st.session_state.get("finalized_filename", "tagged_template.docx")
                        st.download_button(
                            label="💾 Save File",
                            data=content,
                            file_name=filename,
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            type="primary",
                        )

    # Show usage guide
    if st.session_state.get("show_template_download", False):
        with st.expander("📖 How to Use Your Template", expanded=False):
            st.markdown("""
            ### Option 1: Python with docxtpl

            ```python
            from docxtpl import DocxTemplate

            # Load the tagged template
            doc = DocxTemplate("tagged_template.docx")

            # Prepare context with your data
            context = {
            """)
            for var in variables[:5]:
                st.code(f"                '{var.get('suggested_variable_name')}': '',", language="python")
            if len(variables) > 5:
                st.code(f"                # ... and {len(variables) - 5} more", language="python")

            st.markdown("""
            }

            # Render template
            doc.render(context)
            doc.save("output.docx")
            ```
            """)

    return st.session_state.get("show_template_download", False)


def render_document_selector(client: TemplateAPIClient) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    """Render document selection UI for inject phase.

    Shows available DOCX documents. When clicked, displays detected variables
    with context snippets for easy replacement.

    Returns:
        Tuple of (selected_document, edited_variables) or None.
    """
    st.subheader("📂 Select Document for Variable Injection")

    # Fetch available DOCX documents
    with st.spinner("Loading documents..."):
        documents = client.get_docx_documents()

    # Debug info
    if documents:
        st.info(f"Debug: Found {len(documents)} documents")
        for doc in documents:
            st.caption(f"  - {doc.get('filename')}: id={str(doc.get('id'))}")

    if not documents:
        st.info("No DOCX documents available. Upload a document first.")
        return None

    # Display document grid
    st.info(f"Found {len(documents)} DOCX documents. Click to analyze variables.")

    # Create grid of document cards
    num_cols = min(3, len(documents))
    cols = st.columns(num_cols)

    for i, doc in enumerate(documents):
        with cols[i % num_cols]:
            filename = doc.get("filename", "unknown")
            doc_id = str(doc.get("id", ""))  # Convert UUID to string
            status = doc.get("status", "unknown")

            # Status badge
            status_emoji = {
                "completed": "✅",
                "failed": "❌",
                "processing": "🔄",
            }.get(status.lower(), "📄")

            # Document card
            if st.button(
                f"{status_emoji} {filename[:30]}...",
                key=f"select_doc_{doc_id}_{i}",  # Add index to ensure uniqueness
                use_container_width=True,
            ):
                # Store selected document
                st.session_state.selected_doc_for_inject = doc
                st.rerun()

    # If document selected, show inject engine
    selected_doc = st.session_state.get("selected_doc_for_inject")
    if selected_doc:
        return render_inject_engine(client, selected_doc)

    return None


def render_inject_engine(client: TemplateAPIClient, doc_data: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Render the inject engine popup with variables and context.

    Args:
        client: API client instance.
        doc_data: Selected document data.

    Returns:
        Tuple of (document, variables_with_context).
    """
    # Header with back button
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"### 📄 {doc_data.get('filename', 'Document')}")
        st.markdown(f"*Status: {doc_data.get('status', 'unknown')}*")
    with col2:
        if st.button("← Back", use_container_width=True):
            st.session_state.selected_doc_for_inject = None
            st.rerun()

    st.divider()

    st.info("""
    **Variable Injection Workflow**

    This document has already been ingested. To inject variables:

    1. **Upload** the DOCX file to analyze for variables
    2. **Review** detected variables with context snippets
    3. **Provide** replacement values
    4. **Generate** the output document

    💡 *Note: For existing documents, you can re-upload the file to create a dynamic template.*
    """)

    st.divider()

    # File upload for analysis
    col1, col2 = st.columns([3, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload the DOCX file to analyze for variables",
            type=["docx"],
            key="inject_engine_upload",
        )

    with col2:
        st.write("")
        if uploaded_file and st.button("🔍 Analyze", type="primary", use_container_width=True):
            result = client.analyze_template(uploaded_file)
            if result:
                st.session_state.inject_analysis_result = result
                st.session_state.inject_uploaded_file = uploaded_file
                st.rerun()

    # Show analysis results if available
    if st.session_state.get("inject_analysis_result"):
        result = st.session_state.inject_analysis_result

        st.success(f"✅ **Analysis complete!** Found {len(result.get('detected_variables', []))} variables.")

        # Show variables with context
        variables = result.get("detected_variables", [])
        edited_vars = []

        for i, var in enumerate(variables):
            var_name = var.get("suggested_variable_name", "")
            original_text = var.get("original_text", "")
            paragraph_index = var.get("paragraph_index", 0)

            with st.expander(f"📝 {var_name.replace('_', ' ').title()}", expanded=i < 3):
                col1, col2 = st.columns([2, 1])

                with col1:
                    # Input field for replacement value
                    new_value = st.text_input(
                        "Replacement Value",
                        key=f"inject_value_{i}",
                        help="Enter the value to replace this variable",
                    )

                    # Keep checkbox
                    keep = st.checkbox("Include in injection", value=True, key=f"inject_keep_{i}")

                with col2:
                    st.write("**Detected:**")
                    st.code(original_text, language="text")
                    st.caption(f"Paragraph {paragraph_index}")

                # Context snippet
                st.info(f"📍 **Context:** \"...{original_text}...\"")
                st.caption("_The context shows where this value appears in the document._")

                if keep and new_value:
                    edited_vars.append({
                        "original_text": original_text,
                        "suggested_variable_name": var_name,
                        "replacement_value": new_value,
                        "paragraph_index": paragraph_index,
                    })

        # Generate output button
        if edited_vars:
            st.divider()
            col1, col2 = st.columns(2)

            with col1:
                if st.button("🎯 Generate Output Document", type="primary", use_container_width=True):
                    st.success(f"✅ Would generate document with {len(edited_vars)} replacements")
                    st.info("💾 Document generation feature coming soon!")

            with col2:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state.inject_analysis_result = None
                    st.session_state.inject_uploaded_file = None
                    st.rerun()

    return doc_data, []


def render_stored_template_selector(client: TemplateAPIClient) -> None:
    """Render stored template selection UI for inject phase.

    Shows stored templates. When clicked, displays detected variables
    with context snippets for easy replacement.

    Args:
        client: The API client instance.
    """
    st.subheader("📂 Select Stored Template for Variable Injection")

    # Fetch stored templates
    with st.spinner("Loading stored templates..."):
        templates = client.get_stored_templates()

    if not templates:
        st.info("""
        **No stored templates available.**

        To create a stored template:
        1. Switch to "Upload New Template" mode
        2. Upload and analyze a Word document
        3. Click "Save Template" after reviewing variables
        """)
        return

    # Display template grid
    st.info(f"Found {len(templates)} stored templates. Click to inject variables.")

    # Create grid of template cards
    num_cols = min(3, len(templates))
    cols = st.columns(num_cols)

    for i, tmpl in enumerate(templates):
        with cols[i % num_cols]:
            name = tmpl.get("name", "unknown")
            filename = tmpl.get("original_filename", "")
            desc = tmpl.get("description", "")
            # Handle None case for detected_variables
            detected_vars = tmpl.get("detected_variables") or {}
            var_count = detected_vars.get("count", 0)
            is_tagged = tmpl.get("is_tagged", False)
            created_at = tmpl.get("created_at", "")

            # Status badge
            status_emoji = "✅" if is_tagged else "📝"

            # Template card
            if st.button(
                f"{status_emoji} {name[:25]}",
                key=f"select_tmpl_{tmpl.get('id')}_{i}",
                use_container_width=True,
                help=f"{filename}\n{var_count} variables\nCreated: {created_at[:10]}",
            ):
                st.session_state.selected_template = tmpl
                st.rerun()

    # If template selected, show inject engine
    selected_template = st.session_state.get("selected_template")
    if selected_template:
        render_stored_template_inject(client, selected_template)


def generate_random_values(variables: list[dict]) -> dict[str, str]:
    """Generate random UK-specific values based on variable name patterns."""
    import random
    import string
    from datetime import datetime, timedelta

    # Personal Data
    first_names = ["John", "Jane", "James", "Sarah", "Michael", "Emma", "David", "Emily", "Robert", "Olivia"]
    last_names = ["Smith", "Jones", "Williams", "Brown", "Taylor", "Davies", "Wilson", "Evans", "Thomas", "Johnson"]

    # Address Data
    streets = ["High Street", "Church Road", "Queen's Road", "Park Avenue", "King Street", "Market Street", "London Road", "Victoria Road"]
    cities = ["London", "Manchester", "Birmingham", "Leeds", "Glasgow", "Bristol", "Liverpool", "Sheffield", "Edinburgh", "Cardiff"]
    uk_postcodes = ["SW1A 1AA", "M1 1AA", "B1 1AA", "LS1 1AA", "G1 1AA", "BS1 1AA", "L1 1AA", "S1 1AA", "EH1 1AA", "CF1 1AA"]

    # Financial Data (UK)
    tax_codes = ["1257L", "1256L", "1250L", "1185L", "BR", "D0", "D1", "0T", "NT"]

    # Business Data (UK)
    company_suffixes = ["Ltd", "Limited", "PLC", "LLP"]
    company_prefixes = ["British", "United", "National", "Capital", "Premier", "Atlantic", "Pacific", "Global", "First", "Prime"]
    company_industries = ["Financial Services", "Investments", "Holdings", "Solutions", "Consulting", "Partners", "Group", "Associates", "Securities", "Wealth Management"]

    random_values = {}
    for var in variables:
        var_name = var.get("suggested_variable_name", "")
        var_lower = var_name.lower()

        # Sort Code (XX-XX-XX format)
        if "sort_code" in var_lower or "sortcode" in var_lower:
            random_values[var_name] = f"{random.randint(10, 99):02d}-{random.randint(10, 99):02d}-{random.randint(10, 99):02d}"

        # Bank Account Number (8 digits)
        elif "account_number" in var_lower or "accountnumber" in var_lower:
            random_values[var_name] = f"{random.randint(10000000, 99999999)}"

        # Tax Code
        elif "tax_code" in var_lower or "taxcode" in var_lower:
            random_values[var_name] = random.choice(tax_codes)

        # National Insurance Number (XX 00 00 00 X)
        elif "ni_number" in var_lower or "nino" in var_lower or "national_insurance" in var_lower:
            prefix = random.choice(["AB", "AC", "AD", "AE", "AH", "AJ", "AK", "AL", "AM", "AN"])
            suffix = random.choice(["A", "B", "C", "D"])
            random_values[var_name] = f"{prefix} {random.randint(10, 99):02d} {random.randint(10, 99):02d} {random.randint(10, 99):02d} {suffix}"

        # VAT Number (GBXXX XXXXXX)
        elif "vat" in var_lower:
            random_values[var_name] = f"GB{random.randint(100, 999):03d} {random.randint(100000, 999999):06d}"

        # Company Registration Number (8 digits)
        elif "crn" in var_lower or "registration_number" in var_lower or "companies_house" in var_lower:
            random_values[var_name] = f"{random.randint(10000000, 99999999)}"

        # Company Name
        elif "company_name" in var_lower or "company" in var_lower:
            prefix = random.choice(company_prefixes)
            industry = random.choice(company_industries)
            suffix = random.choice(company_suffixes)
            random_values[var_name] = f"{prefix} {industry} {suffix}"

        # Reference Number (alphanumeric)
        elif "reference" in var_lower or "ref" in var_lower:
            chars = string.ascii_uppercase + string.digits
            ref = ''.join(random.choices(chars, k=8))
            random_values[var_name] = f"REF-{ref}"

        # Pension Value
        elif "pension" in var_lower:
            amount = random.randint(50000, 2000000)
            random_values[var_name] = f"£{amount:,.2f}"

        # Names (must come after more specific patterns to avoid false matches)
        elif "name" in var_lower or "client" in var_lower:
            if "first" in var_lower or "forename" in var_lower:
                random_values[var_name] = random.choice(first_names)
            elif "last" in var_lower or "surname" in var_lower:
                random_values[var_name] = random.choice(last_names)
            else:
                random_values[var_name] = f"{random.choice(first_names)} {random.choice(last_names)}"

        # Date
        elif "date" in var_lower:
            rand_date = datetime.now() + timedelta(days=random.randint(-365, 365))
            random_values[var_name] = rand_date.strftime("%d/%m/%Y")

        # Monetary Amount / Balance
        elif "amount" in var_lower or "monetary" in var_lower or "balance" in var_lower:
            amount = random.randint(1000, 500000)
            random_values[var_name] = f"£{amount:,.2f}"

        # Percentage / Rate (must come after pension_rate, tax_rate checks above)
        elif "percentage" in var_lower or "percent" in var_lower:
            random_values[var_name] = f"{random.randint(1, 100)}%"

        # Address
        elif "address" in var_lower:
            house_num = random.randint(1, 999)
            street = random.choice(streets)
            city = random.choice(cities)
            postcode = random.choice(uk_postcodes)
            random_values[var_name] = f"{house_num} {street}, {city}, {postcode}"

        # Email
        elif "email" in var_lower:
            name = random.choice(first_names).lower()
            domain = random.choice(["gmail.com", "yahoo.com", "outlook.com", "hotmail.com"])
            random_values[var_name] = f"{name}.{random.choice(last_names).lower()}@{domain}"

        # Phone / Telephone
        elif "phone" in var_lower or "telephone" in var_lower:
            random_values[var_name] = f"+44 {random.randint(1000, 9999)} {random.randint(100000, 999999)}"

        # Passport Number (9 digits)
        elif "passport" in var_lower:
            random_values[var_name] = f"{random.randint(100000000, 999999999)}"

        # Driving License Number (UK format)
        elif "driving_license" in var_lower or "drivinglicence" in var_lower:
            # UK driving license format: 5 chars + number + 2 chars + number
            surname = random.choice(last_names)[:5].upper().ljust(5, '9')
            random_values[var_name] = f"{surname}{random.choice(['M', 'F'])}{random.randint(100, 999):03d}{random.randint(10, 99):02d}{random.choice(['A', 'B', 'C'])}{random.randint(1, 9)}"

        # Fallback - use variable name as placeholder
        else:
            random_values[var_name] = f"[{var_name}]"

    return random_values


def render_stored_template_inject(client: TemplateAPIClient, tmpl: dict[str, Any]) -> None:
    """Render the inject engine for a stored template.

    Args:
        client: API client instance.
        tmpl: Selected template data.
    """
    # Header with back button
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"### 📝 {tmpl.get('name', 'Template')}")
        st.markdown(f"*{tmpl.get('description', 'No description')}*")
        # Handle None case for detected_variables
        detected_vars = tmpl.get('detected_variables') or {}
        st.caption(f"Original: `{tmpl.get('original_filename')}` | {detected_vars.get('count', 0)} variables")
    with col2:
        if st.button("← Back", use_container_width=True):
            st.session_state.selected_template = None
            st.rerun()

    st.divider()

    # Fetch current template status from API for real-time updates
    template_id = tmpl.get("id")
    if template_id:
        current_status = client.get_template_status(template_id)
        if current_status:
            # Update the display with fresh status
            template_status = current_status.get("status", "")
            injection_status = current_status.get("injection_status", "")

            if template_status == "FINALIZING" or injection_status in ["queued", "processing"]:
                # Show simple progress with spinner
                if injection_status == "queued":
                    st.info("⏳ Queued for processing...")
                    st.progress(0.2)
                elif injection_status == "processing":
                    st.info("⚙️ Processing template injection...")
                    st.progress(0.6)
                elif template_status == "FINALIZING":
                    st.info("🔄 Finalizing content...")
                    st.progress(0.4)
                else:
                    st.info("🔄 Initializing...")
                    st.progress(0.1)

                # Auto-refresh with spinner
                with st.spinner("Waiting for update..."):
                    import time
                    time.sleep(3)
                st.rerun()
            elif template_status == "COMPLETED" and current_status.get("download_ready"):
                st.success("✅ Processing Complete!")
                st.progress(1.0)
                st.caption("Template is ready for download!")

                # Show download button
                download_url = current_status.get("download_url")
                if download_url:
                    # Create a proper download link with org_id query parameter (browser downloads can't send custom headers)
                    full_url = f"{client.base_url}{download_url}?org_id={client.org_id}"
                    st.markdown(
                        f"### 📥 Download Ready\n"
                        f"Click the link below to download your filled document:\n\n"
                        f"[📥 Download {tmpl.get('original_filename')}]({full_url})"
                    )

                # Show clear button
                if st.button("← Back to Templates", use_container_width=True):
                    st.session_state.selected_template = None
                    st.rerun()
                return  # Exit early to show only the download option

    st.divider()

    # Get variables from stored template (handle None case)
    variables_data = tmpl.get("detected_variables") or {}
    variables = variables_data.get("variables", [])

    if not variables:
        st.warning("No variables found in this template.")
        return

    st.success(f"✅ **Template loaded!** Found {len(variables)} variables.")

    st.info("""
    **Variable Injection Workflow**

    Enter replacement values for each variable below. When ready,
    click "Generate Document" to create the output.
    """)

    st.divider()

    # Random Fill Logic - Use template-specific storage to avoid conflicts
    random_values_key = f"random_values_{tmpl.get('id')}"

    # Use a generation counter to force widget recreation when values change
    gen_count_key = f"gen_count_{tmpl.get('id')}"
    if gen_count_key not in st.session_state:
        st.session_state[gen_count_key] = 0

    # Auto-generate random values on first load (if not already generated)
    if random_values_key not in st.session_state or not st.session_state[random_values_key]:
        st.session_state[random_values_key] = generate_random_values(variables)
        st.session_state[gen_count_key] = 1  # First generation
        # Debug: Show what was generated
        with st.expander("🔍 Debug: Generated Random Values", expanded=False):
            st.json(st.session_state[random_values_key])

    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    with col_btn1:
        if st.button("🔄 Regenerate", use_container_width=True, key=f"random_fill_btn_{tmpl.get('id')}"):
            # Generate and store random values for this template
            st.session_state[random_values_key] = generate_random_values(variables)
            # Increment generation counter to force widget recreation
            st.session_state[gen_count_key] += 1
            st.rerun()

    with col_btn2:
        # Clear button to remove random values
        if st.button("🗑️ Clear", use_container_width=True, key=f"clear_btn_{tmpl.get('id')}"):
            st.session_state[random_values_key] = {}
            # Increment generation counter to force widget recreation
            st.session_state[gen_count_key] += 1
            st.rerun()

    st.divider()

    # Show variables with context - OUTSIDE of form for dynamic updates
    # Track which variables to include in injection
    include_vars_key = f"include_vars_{tmpl.get('id')}"
    if include_vars_key not in st.session_state:
        # Initialize all as included
        st.session_state[include_vars_key] = {var.get("suggested_variable_name", ""): True for var in variables}

    # Get current generation count for widget keys
    gen_count = st.session_state.get(gen_count_key, 0)

    # Display all variables
    for i, var in enumerate(variables):
        var_name = var.get("suggested_variable_name", "")
        original_text = var.get("original_text", "")
        paragraph_index = var.get("paragraph_index", 0)

        with st.expander(f"📝 {var_name.replace('_', ' ').title()}", expanded=i < 3):
            col1, col2 = st.columns([2, 1])

            with col1:
                # Input field for replacement value - get from random values
                random_values = st.session_state.get(random_values_key, {})
                default_val = random_values.get(var_name, "")

                # Debug for first 3 variables
                if i < 3:
                    st.caption(f"🔍 Debug: var_name='{var_name}', default_val='{default_val}'")

                # Widget key includes generation count to force recreation when regenerated
                widget_key = f"stored_inject_value_{i}_{tmpl.get('id')}_gen{gen_count}"

                new_value = st.text_input(
                    "Replacement Value",
                    value=default_val,
                    key=widget_key,
                    help="Enter the value to replace this variable",
                )

                # Keep checkbox - also use generation count for consistency
                keep_key = f"stored_inject_keep_{i}_{tmpl.get('id')}_gen{gen_count}"
                keep = st.checkbox("Include in injection", value=st.session_state[include_vars_key].get(var_name, True), key=keep_key)
                # Update include state
                st.session_state[include_vars_key][var_name] = keep

            with col2:
                st.write("**Detected:**")
                st.code(original_text, language="text")
                st.write("**Paragraph:**", paragraph_index)

    st.divider()

    # Form for submit button only
    with st.form(key=f"submit_form_{tmpl.get('id')}"):
        submit = st.form_submit_button("🚀 Inject & Process", type="primary", use_container_width=True)

    if submit:
        # Collect values from widgets using current generation count
        edited_vars = []
        for i, var in enumerate(variables):
            var_name = var.get("suggested_variable_name", "")
            original_text = var.get("original_text", "")
            paragraph_index = var.get("paragraph_index", 0)

            # Check if this variable is included
            if st.session_state[include_vars_key].get(var_name, True):
                widget_key = f"stored_inject_value_{i}_{tmpl.get('id')}_gen{gen_count}"
                value = st.session_state.get(widget_key, "")
                edited_vars.append({
                    "variable_name": var_name,
                    "replacement_value": value,
                    "original_text": original_text,
                    "paragraph_index": paragraph_index
                })

        if not edited_vars:
            st.warning("No variables selected for injection.")
        else:
            task_id = client.finalize_template_async(tmpl.get("id"), edited_vars)
            if task_id:
                st.success(f"✅ Injection queued! Task ID: `{task_id}`")
                st.info("Check the 'processing' status tab or Download Center for results.")
                # Clear random values and reset state
                st.session_state[random_values_key] = {}
                st.session_state.selected_template = None
                st.rerun()


def render_injection_queue_dashboard(client: TemplateAPIClient) -> None:
    """Render the injection queue status dashboard."""
    st.subheader("📊 Injection Queue Status")

    # Poll status
    status_data = client.get_injection_queue()
    
    if not status_data:
        st.warning("Could not fetch queue status.")
        return

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Queued", status_data.get("queued", 0))
    with col2:
        st.metric("Processing", status_data.get("processing", 0))
    with col3:
        st.metric("Completed", status_data.get("completed", 0))
    with col4:
        st.metric("Failed", status_data.get("failed", 0))

    st.divider()

    # Recent Jobs Table
    st.markdown("### Recent Injection Jobs")
    jobs = status_data.get("jobs", [])
    
    if not jobs:
        st.info("No recent injection jobs found.")
    else:
        # Create a dataframe-like display or just container list
        for job in jobs:
            with st.container():
                cols = st.columns([3, 1, 2, 2])
                with cols[0]:
                    st.write(f"**{job.get('filename')}**")
                    st.caption(f"ID: `{job.get('template_id')}`")
                with cols[1]:
                    status = job.get("status", "unknown")
                    icon = "⏳"
                    if status == "processing": icon = "🔄"
                    elif status == "completed": icon = "✅"
                    elif status == "failed": icon = "❌"
                    st.write(f"{icon} {status}")
                with cols[2]:
                    started = job.get("started_at")
                    if started:
                        st.caption(f"Started: {started[:19]}")
                with cols[3]:
                    completed = job.get("completed_at")
                    if completed:
                        st.caption(f"Completed: {completed[:19]}")
                st.divider()


                st.divider()


def render_download_center(client: TemplateAPIClient) -> None:
    """Render the download center for completed templates."""
    st.subheader("📥 Download Center")
    st.markdown("Access all your processed and injected documents here.")

    # Fetch ready files
    with st.spinner("Loading available downloads..."):
        ready_files = client.get_ready_for_download()

    if not ready_files:
        st.info("No files ready for download yet.")
        st.caption("Once you finalize a template injection, it will appear here.")
        return

    st.success(f"✅ Found {len(ready_files)} files ready for download.")
    st.divider()

    # Display files
    for tmpl in ready_files:
        with st.container():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**📄 {tmpl.get('original_filename', 'Document')}**")
                # Show injection task details if available
                if tmpl.get('injection_completed_at'):
                    completed_at = tmpl.get('injection_completed_at', '')[:19]
                    st.caption(f"Completed: {completed_at}")
                
            with col2:
                url = tmpl.get('download_url')
                if url:
                    st.link_button(
                        "⬇️ Download", 
                        url,
                        type="primary",
                        use_container_width=True
                    )
            st.divider()


def render_prompt_settings(client: TemplateAPIClient) -> None:
    """Render the prompt management settings section."""
    st.subheader("📝 Analysis Prompt Settings")
    st.markdown("Customize the prompt used for template analysis.")

    # Get the current default prompt
    default_prompt_data = client.get_default_prompt()

    # Show current default info
    current_source = default_prompt_data.get("source", "system")
    current_name = default_prompt_data.get("name", "System Default")

    st.info(f"**Current Default:** {current_name} ({current_source})")

    st.divider()

    # Saved prompts dropdown
    saved_prompts = client.get_prompts()

    if saved_prompts:
        st.subheader("Saved Prompts")
        prompt_options = {p["name"]: p for p in saved_prompts}
        selected_name = st.selectbox(
            "Select a saved prompt to view/edit",
            options=list(prompt_options.keys()),
            key="saved_prompt_selector",
        )

        if selected_name:
            selected_prompt = prompt_options[selected_name]
            st.text_area(
                "Prompt Text (read-only)",
                value=selected_prompt.get("prompt_text", ""),
                height=200,
                disabled=True,
                key="saved_prompt_preview",
            )

            col1, col2 = st.columns(2)
            with col1:
                if selected_prompt.get("is_default"):
                    st.success("✅ This is the default prompt")
                else:
                    st.caption("Not set as default")
            with col2:
                if st.button("🗑️ Delete", key=f"delete_{selected_prompt['id']}"):
                    if client.delete_prompt(selected_prompt["id"]):
                        st.success("Prompt deleted!")
                        st.rerun()

        st.divider()

    # Create new prompt section
    st.subheader("Create New Prompt")

    # Show system default for reference
    with st.expander("🔍 View System Default Prompt (reference)"):
        system_default = """You are an expert Document Intelligence Engineer. Analyze the text segment.

DEFINITIONS:
1. Dynamic: Text that changes per client (Names, Dates, Risk Profiles, Amounts).
2. Static: Legal headers, boilerplate, firm branding.

FEW-SHOT EXAMPLES:
Input: "prepared for Mr. James Arlington on 12th March"
Output: { "is_dynamic": true, "extraction": [{ "original": "Mr. James Arlington", "var": "client_name" }, { "original": "12th March", "var": "report_date" }] }

Input: "The value of investments can go down as well as up."
Output: { "is_dynamic": false, "extraction": [] }

TASK:
Analyze the user input. Return valid JSON only."""
        st.code(system_default, language="text")

    # New prompt form
    new_prompt_name = st.text_input(
        "Prompt Name",
        placeholder="e.g., Financial Documents Analyzer",
        key="new_prompt_name",
    )

    new_prompt_text = st.text_area(
        "Prompt Text",
        height=300,
        placeholder="Enter your custom analysis prompt here...",
        key="new_prompt_text",
        help="This prompt will be used for LLM-based template analysis. Include examples and clear instructions.",
    )

    set_as_default = st.checkbox(
        "Set as default prompt",
        key="new_prompt_default",
        help="If checked, this prompt will be used for all future template analyses.",
    )

    if st.button("💾 Save Prompt", type="primary", key="save_new_prompt"):
        if not new_prompt_name:
            st.error("Please provide a name for the prompt.")
        elif not new_prompt_text:
            st.error("Please provide the prompt text.")
        elif len(new_prompt_text) > 8000:
            st.error("Prompt text is too long. Maximum 8000 characters.")
        else:
            result = client.save_prompt(new_prompt_name, new_prompt_text, set_as_default)
            if result:
                st.success(f"✅ Prompt '{new_prompt_name}' saved successfully!")
                st.rerun()


def render_template_engine(client: TemplateAPIClient) -> None:
    """Render the complete template engine workflow."""
    init_template_session_state()

    st.title("Template Engine")
    st.markdown("Transform Word documents into dynamic Jinja2 templates")
    st.divider()

    # Mode selection
    mode = st.radio(
        "Select Mode",
        ["📤 Batch Upload", "💉 Variable Injection", "📊 Status Dashboard", "📥 Download Center", "⚙️ Prompt Settings"],
        horizontal=True,
    )

    if mode == "📤 Batch Upload":
        render_template_upload(client)

    elif mode == "💉 Variable Injection":
        # This will be Phase 3 implementation
        render_stored_template_selector(client)

    elif mode == "📊 Status Dashboard":
        render_injection_queue_dashboard(client)

    elif mode == "📥 Download Center":
        render_download_center(client)

    elif mode == "⚙️ Prompt Settings":
        render_prompt_settings(client)


# =============================================================================
# Main App
# =============================================================================


def render_sidebar(ingestion_client: APIClient, template_client: TemplateAPIClient) -> None:
    """Render the sidebar with connection status."""
    with st.sidebar:
        st.title("📄 Doc Intelligence")

        st.divider()

        # Connection status
        is_healthy = ingestion_client.health_check()
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
            value=ORG_ID,
            disabled=True,
            help="Your organization ID for multi-tenancy isolation",
        )

        st.divider()

        # Instructions
        st.subheader("Features")
        st.markdown("""
        ### 📥 Document Ingestion
        Upload documents for processing and vector storage.

        ### 📝 Template Engine
        Convert Word templates to dynamic Jinja2 templates.

        **Workflow:**
        1. Upload Word (.docx) template
        2. Review detected variables
        3. Provide client data
        4. Generate & download tagged template
        """)

        st.divider()
        st.caption(f"API: `{API_BASE_URL}`")


def main() -> None:
    """Main application entry point."""

    # Initialize API clients
    ingestion_client = APIClient(API_BASE_URL, ORG_ID)
    template_client = TemplateAPIClient(API_BASE_URL, ORG_ID)

    # Render sidebar
    render_sidebar(ingestion_client, template_client)

    # Main content area - Only Template Engine is active
    # COMMENTED_OUT: Document Ingestion and Settings tabs
    # tab1, tab2, tab3 = st.tabs(["📥 Document Ingestion", "📝 Template Engine", "⚙️ Settings"])
    #
    # with tab1:
    #     st.title("Document Ingestion Dashboard")
    #     st.markdown("Upload and track document processing")
    #     st.divider()
    #     st.info("Ingestion module temporarily disabled.")
    #
    # with tab3:
    #     st.subheader("⚙️ Settings")
    #     st.write("**API Configuration**")
    #     st.text_input("API Base URL", value=API_BASE_URL, disabled=True)
    #     st.write("**Organization**")
    #     st.text_input("Organization ID", value=ORG_ID, disabled=True)
    #     st.write("**Strategy Configuration**")
    #     st.info("Strategies are configured server-side via environment variables.")
    #     st.write("**Current Strategies**")
    #     st.json({
    #         "parser": "llama_parse",
    #         "chunker": "markdown",
    #         "embedder": "openai",
    #         "vector_store": "qdrant",
    #     })

    # Only Template Engine tab is active
    render_template_engine(template_client)


if __name__ == "__main__":
    main()
