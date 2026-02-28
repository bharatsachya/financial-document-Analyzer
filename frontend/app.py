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
import streamlit.components.v1 as components
from streamlit.runtime.uploaded_file_manager import UploadedFile

# Custom components
d3_carousel_component = components.declare_component("d3_carousel", path="frontend/carousel_component")

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

    def get_stored_template_preview(self, template_id: str) -> dict[str, Any] | None:
        """Get the parsed text preview of a specific stored template.

        Args:
            template_id: The template ID.

        Returns:
            Template preview data or None if not found/failed.
        """
        url = f"{self.base_url}/templates/stored/{template_id}/preview"

        try:
            response = httpx.get(url, headers=self.headers, timeout=10.0)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            import logging
            logging.getLogger(__name__).error(f"Fetch template preview failed: {e.response.status_code}")
            return None
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Fetch template preview error: {e}")
            return None

    def get_template_pdf(self, template_id: str) -> bytes | None:
        """Get the PDF version of a stored template.

        Args:
            template_id: The template ID.

        Returns:
            PDF bytes or None if conversion failed.
        """
        url = f"{self.base_url}/templates/stored/{template_id}/pdf"

        try:
            response = httpx.get(url, headers=self.headers, timeout=60.0)
            response.raise_for_status()
            return response.content
        except httpx.HTTPStatusError as e:
            import logging
            logging.getLogger(__name__).error(f"Fetch template PDF failed: {e.response.status_code} — {e.response.text}")
            return None
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Fetch template PDF error: {e}")
            return None

    def get_draft_versions(self, template_id: str) -> list[dict[str, Any]]:
        """Get all draft versions for a template."""
        url = f"{self.base_url}/templates/stored/{template_id}/versions"
        try:
            response = httpx.get(url, headers=self.headers, timeout=10.0)
            response.raise_for_status()
            return response.json().get("versions", [])
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Fetch draft versions failed: {e}")
            return []
            
    def get_version_pdf(self, version_id: str) -> bytes | None:
        """Get the PDF version of a specific draft version."""
        url = f"{self.base_url}/templates/versions/{version_id}/pdf"
        try:
            response = httpx.get(url, headers=self.headers, timeout=10.0)
            response.raise_for_status()
            return response.content
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Fetch version PDF failed: {e}")
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
        """Save a custom extraction prompt."""
        url = f"{self.base_url}/templates/prompts"
        payload = {
            "name": name,
            "prompt_text": prompt_text,
            "set_as_default": set_as_default,
        }
        try:
            response = httpx.post(url, headers=self.headers, json=payload, timeout=10.0)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            error_detail = e.response.text
            try:
                error_json = e.response.json()
                error_detail = error_json.get("detail", error_detail)
            except Exception:
                pass
            logger.error(f"Save prompt failed: {e.response.status_code} - {error_detail}")
            st.error(f"Save prompt failed ({e.response.status_code}): {error_detail}")
        except Exception as e:
            logger.error(f"Save prompt error: {e}", exc_info=True)
            st.error(f"Save prompt error: {e}")
        return None

    # =========================================================================
    # Report Learning API Methods
    # =========================================================================

    def capture_feedback(
        self,
        adviser_id: str,
        original_text: str,
        edited_text: str,
        report_type: str | None = None,
    ) -> dict[str, Any] | None:
        """Send feedback (original vs edited text) to learn preferences."""
        url = f"{self.base_url}/templates/capture-feedback"
        payload = {
            "adviser_id": adviser_id,
            "original_text": original_text,
            "edited_text": edited_text,
        }
        if report_type:
            payload["report_type"] = report_type
        try:
            response = httpx.post(url, headers=self.headers, json=payload, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Capture feedback failed: {e.response.status_code}")
            st.error(f"Feedback capture failed: {e.response.status_code}")
            return None
        except Exception as e:
            logger.error(f"Capture feedback error: {e}")
            st.error(f"Feedback capture error: {e}")
            return None

    def get_preferences(self, adviser_id: str) -> dict[str, Any]:
        """Fetch learned style preferences for Memory Insights."""
        url = f"{self.base_url}/templates/adviser-preferences/{adviser_id}"
        try:
            response = httpx.get(url, headers=self.headers, timeout=10.0)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Get preferences error: {e}")
            return {"rules": [], "total": 0}

    def generate_report(
        self,
        adviser_id: str,
        prompt: str,
    ) -> dict[str, Any] | None:
        """Queue personalized report generation."""
        url = f"{self.base_url}/templates/generate-personalized-report"
        payload = {"adviser_id": adviser_id, "prompt": prompt}
        try:
            response = httpx.post(url, headers=self.headers, json=payload, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Generate report error: {e}")
            st.error(f"Report generation failed: {e}")
            return None

    def get_task_result(self, task_id: str) -> dict[str, Any] | None:
        """Poll Celery task result via Flower or direct."""
        # For hackathon we poll the backend status endpoint
        url = f"{self.base_url}/templates/report-status/{task_id}"
        try:
            response = httpx.get(url, headers=self.headers, timeout=10.0)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None

    def generate_draft(self, adviser_id: str, client_id: str, topic: str, template_text: str = None, template_id: str = None) -> dict[str, Any] | None:
        """Generate a real draft report via the LLM combining factual and procedural data."""
        url = f"{self.base_url}/templates/generate-draft"
        payload = {
            "adviser_id": adviser_id,
            "client_id": client_id,
            "topic": topic,
        }
        if template_text:
            payload["template_text"] = template_text
        if template_id:
            payload["template_id"] = str(template_id)
            
        try:
            response = httpx.post(url, headers=self.headers, json=payload, timeout=60.0)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Generate draft failed: {e.response.status_code} - {e.response.text}")
            st.error(f"Generate draft failed: {e.response.status_code}")
            return None
        except Exception as e:
            logger.error(f"Generate draft error: {e}")
            st.error(f"Generate draft error: {e}")
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


# =============================================================================
# Report Review & Learn UI
# =============================================================================


SAMPLE_REPORT = """Dear Mr. and Mrs. Henderson,

Following our recent meeting on 15th January 2025, I am pleased to present your Annual Portfolio Review for the period ending 31st December 2024.

Portfolio Performance Summary:
Your portfolio has achieved a total return of 8.2% over the review period, compared to the benchmark return of 7.1%. The portfolio value currently stands at £485,000, representing a net increase of £36,770.

Asset Allocation:
- UK Equities: 35% (£169,750) — Overweight vs. target of 30%
- Global Equities: 25% (£121,250) — In line with target
- Fixed Income: 20% (£97,000) — Underweight vs. target of 25%
- Property: 10% (£48,500) — In line with target
- Cash: 10% (£48,500) — Overweight vs. target of 5%

Risk Assessment:
Based on your completed risk questionnaire (score: 6/10), we classify your risk profile as "Balanced Growth". This remains appropriate given your stated investment horizon of 15+ years and your objective of funding retirement at age 65.

Recommendations:
1. Rebalance UK equities to target allocation, taking profits of approximately £24,250
2. Increase fixed income allocation by £24,250 to provide greater portfolio stability
3. Reduce cash holdings to 5%, deploying £24,250 into global equity markets
4. Consider adding emerging market exposure (5%) for diversification benefits

These recommendations are subject to market conditions and your ongoing agreement. Past performance is not a guarantee of future returns. The value of investments can go down as well as up.

Kind regards,
Financial Advisory Team"""


def render_review_and_learn(client: TemplateAPIClient) -> None:
    """Render the Report Review & Learn interface.

    This is where the RLHF training signal is captured:
    1. Display AI-generated report text
    2. User edits text to match their preferred style
    3. Diff is captured and sent to learn_preference_task
    4. Memory Insights shows what rules have been learned
    """
    st.subheader("🧠 Report Review & Style Learning")
    st.markdown(
        "Edit the sample report below to match your preferred style. "
        "Your edits teach the system your formatting and tone preferences."
    )

    # Initialize session state for review
    if "review_original" not in st.session_state:
        st.session_state.review_original = SAMPLE_REPORT
    if "review_submitted" not in st.session_state:
        st.session_state.review_submitted = False
    if "review_adviser_id" not in st.session_state:
        st.session_state.review_adviser_id = "adv_001"

    # Adviser selector
    col_adviser, col_reset = st.columns([3, 1])
    with col_adviser:
        adviser = st.selectbox(
            "Acting as Adviser",
            ["adv_001 — Sarah Mitchell", "adv_002 — James Crawford"],
            key="adviser_selector",
        )
        adviser_id = adviser.split(" — ")[0]
        st.session_state.review_adviser_id = adviser_id
    with col_reset:
        st.write("")
        st.write("")
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.review_original = SAMPLE_REPORT
            st.session_state.review_submitted = False
            st.rerun()

    st.divider()

    # Two-tab layout: Edit | Diff + Learn
    tab_edit, tab_learn = st.tabs(["✏️ Edit Report", "📊 Diff & Learn"])

    with tab_edit:
        st.markdown("**Edit the report text below** — change tone, formatting, terminology, structure:")

        edited_text = st.text_area(
            "Report Text",
            value=st.session_state.review_original,
            height=500,
            label_visibility="collapsed",
            key="report_editor",
        )

        col_submit, col_info = st.columns([1, 3])
        with col_submit:
            if st.button("📝 Submit Edits", type="primary", use_container_width=True):
                if edited_text.strip() != st.session_state.review_original.strip():
                    st.session_state.review_edited = edited_text
                    st.session_state.review_submitted = True
                    st.success("✅ Edits captured! Switch to **Diff & Learn** tab to review and approve.")
                else:
                    st.warning("No changes detected. Please edit the text first.")
        with col_info:
            st.caption(
                "💡 Try changing tone (formal ↔ casual), replacing jargon, "
                "restructuring paragraphs, or adjusting number formatting."
            )

    with tab_learn:
        if not st.session_state.review_submitted:
            st.info("✏️ Edit the report in the **Edit Report** tab first, then submit your changes.")
        else:
            original = st.session_state.review_original
            edited = st.session_state.get("review_edited", original)

            # Show diff
            st.markdown("### Changes Detected")

            # Simple line-by-line diff visualization
            import difflib

            original_lines = original.splitlines()
            edited_lines = edited.splitlines()

            diff = list(difflib.unified_diff(
                original_lines, edited_lines,
                fromfile="Original (AI)", tofile="Your Edits",
                lineterm="",
            ))

            if diff:
                diff_text = "\n".join(diff)
                st.code(diff_text, language="diff")

                changes_count = sum(1 for line in diff if line.startswith("+") and not line.startswith("+++"))
                st.metric("Lines Changed", changes_count)
            else:
                st.success("No differences found.")

            st.divider()

            # Approve & Learn button
            col_approve, col_info2 = st.columns([1, 2])
            with col_approve:
                if st.button(
                    "🧠 Approve & Learn",
                    type="primary",
                    use_container_width=True,
                    help="Send your edits to the AI to learn your style preferences",
                ):
                    with st.spinner("Sending feedback to learning pipeline..."):
                        result = client.capture_feedback(
                            adviser_id=st.session_state.review_adviser_id,
                            original_text=original,
                            edited_text=edited,
                            report_type="portfolio_review",
                        )
                        if result:
                            st.success(
                                f"✅ **Feedback captured!** Task `{result.get('task_id', 'N/A')}` queued.\n\n"
                                f"The system is now extracting your style preferences and storing them. "
                                f"Future template analyses for your organization will reflect these preferences."
                            )
                            st.balloons()
                        else:
                            st.error("Failed to send feedback. Check API connection.")
            with col_info2:
                st.caption(
                    "This sends your original and edited text to the learning pipeline. "
                    "An LLM will extract the stylistic rules you applied, embed them, "
                    "and store them in Qdrant for future template analyses."
                )

    # Memory Insights Panel
    st.divider()
    with st.expander("🧠 Memory Insights — Learned Style Preferences", expanded=False):
        prefs = client.get_preferences(st.session_state.review_adviser_id)
        rules = prefs.get("rules", [])
        total = prefs.get("total", 0)

        if total > 0:
            st.metric("Total Learned Rules", total)

            for i, rule in enumerate(rules):
                rule_text = rule.get("rule_text", "Unknown rule")
                created = rule.get("created_at", "")[:19]
                adviser = rule.get("adviser_id", "")

                st.markdown(
                    f"**Rule {i+1}:** {rule_text}\n\n"
                    f"*Learned from: {adviser} • {created}*"
                )
                if i < len(rules) - 1:
                    st.divider()
        else:
            st.info(
                "No style preferences learned yet for this organization. "
                "Use the **Edit Report** tab to make edits and click **Approve & Learn** "
                "to teach the system your preferred style."
            )


def render_director_typist_ui(client: TemplateAPIClient) -> None:
    """Render the Report Review & Learn interface with Director-Typist UI."""
    st.subheader("🧠 Report Review & Style Learning (Director-Typist)")
    
    # Session state init
    if "dt_report_text" not in st.session_state:
        st.session_state.dt_report_text = ""
    if "dt_ai_variables" not in st.session_state:
        st.session_state.dt_ai_variables = {}
    if "dt_user_variables" not in st.session_state:
        st.session_state.dt_user_variables = {}
    if "dt_stylistic_feedback" not in st.session_state:
        st.session_state.dt_stylistic_feedback = ""
    if "dt_draft_generated" not in st.session_state:
        st.session_state.dt_draft_generated = False

    # Adviser selector
    col_adviser, col_client, col_topic = st.columns(3)
    with col_adviser:
        adviser = st.selectbox(
            "Acting as Adviser",
            ["adv_001 — Sarah Mitchell", "adv_002 — James Crawford"],
            key="dt_adviser_selector",
        )
        adviser_id = adviser.split(" — ")[0]
    with col_client:
        client_id_input = st.text_input("Client ID", value="client_001", key="dt_client_id")
    with col_topic:
        topic_input = st.text_input("Topic", value="Annual Portfolio Review", key="dt_topic")

    st.markdown("### 📝 Select Template")
    
    # Fetch actual DOCX records present in PostgreSQL via our TemplateStorage queries
    templates = client.get_stored_templates()
    template_id = None
    
    if templates:
        # Create a mapping of template name -> ID
        template_options = {t["name"]: t["id"] for t in templates}
        selected_template_name = st.selectbox(
            "Select an uploaded template to review",
            list(template_options.keys()),
            key="dt_template_selector",
        )
        template_id = template_options[selected_template_name]
    else:
        st.warning("No templates found in the database. Please upload one first (Batch Upload).")
        return

    st.divider()

    # ── Full Document Preview + Feedback ──────────────────────────────────────
    
    col_doc, col_feedback = st.columns([7, 3])
    
    with col_doc:
        st.markdown("### 📄 Document Preview")
        
        # Check for versions
        versions = client.get_draft_versions(template_id)
        active_version_id = st.session_state.get(f"active_version_{template_id}")
        
        if versions:
            # Sort descending by version number
            versions = sorted(versions, key=lambda x: x["version_number"], reverse=True)
            if not active_version_id:
                active_version_id = versions[0]["id"]
                st.session_state[f"active_version_{template_id}"] = active_version_id
                
            selected_id = d3_carousel_component(
                versions=versions,
                selected_id=active_version_id,
                api_base_url=client.base_url,
                key=f"d3_carousel_{template_id}_{len(versions)}"
            )
            
            if selected_id and selected_id != active_version_id:
                st.session_state[f"active_version_{template_id}"] = selected_id
                st.rerun()
                
            pdf_bytes = client.get_version_pdf(st.session_state[f"active_version_{template_id}"])
        else:
            # Fallback to the original template PDF if no generated drafts exist yet
            pdf_bytes = client.get_template_pdf(template_id)
        
        if pdf_bytes:
            import base64
            b64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
            pdf_display = (
                f'<iframe src="data:application/pdf;base64,{b64_pdf}" '
                f'width="100%" height="650" type="application/pdf" '
                f'style="border: 1px solid #333; border-radius: 10px; margin-top: 15px;">'
                f'</iframe>'
            )
            st.markdown(pdf_display, unsafe_allow_html=True)
        else:
            # Fallback to the rich HTML preview if PDF conversion isn't available
            if versions and active_version_id:
                active_version = next((v for v in versions if v["id"] == active_version_id), None)
                if active_version and "generated_text" in active_version:
                    st.markdown(
                        f'<div style="background:#1a1a2e; padding:24px 28px; border-radius:12px; '
                        f'border:1px solid #333; font-size:14px; line-height:1.8; '
                        f'color:#e0e0e0; font-family: Georgia, serif; '
                        f'max-height:600px; overflow-y:auto; '
                        f'box-shadow: 0 4px 12px rgba(0,0,0,0.3); white-space: pre-wrap;">'
                        f'{active_version["generated_text"]}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.warning("⚠️ Could not load document preview for this version.")
            else:
                preview_data = client.get_stored_template_preview(template_id)
                if preview_data and "template_text" in preview_data:
                    st.markdown(
                        f'<div style="background:#1a1a2e; padding:24px 28px; border-radius:12px; '
                        f'border:1px solid #333; font-size:14px; line-height:1.8; '
                        f'color:#e0e0e0; font-family: Georgia, serif; '
                        f'max-height:600px; overflow-y:auto; '
                        f'box-shadow: 0 4px 12px rgba(0,0,0,0.3); white-space: pre-wrap;">'
                        f'{preview_data["template_text"]}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.warning("⚠️ Could not load document preview. Ensure MS Word or LibreOffice is installed for PDF conversion.")
    
    with col_feedback:
        st.markdown("### 🗣️ Feedback")
        st.caption("Review the document and provide your feedback before generating.")
        
        # Stylistic Feedback
        stylistic_fb = st.text_area(
            "Style & Tone Feedback",
            value=st.session_state.dt_stylistic_feedback,
            height=150,
            placeholder="e.g. Make the tone more formal, use shorter paragraphs, avoid jargon...",
            key="dt_preview_style_feedback",
        )
        if stylistic_fb != st.session_state.dt_stylistic_feedback:
            st.session_state.dt_stylistic_feedback = stylistic_fb
        
        st.divider()
        
        # Generate Draft button
        if st.button("🚀 Generate Draft", type="primary", use_container_width=True):
            with st.spinner("Atlas is recalling knowledge and writing the draft..."):
                result = client.generate_draft(
                    adviser_id=adviser_id,
                    client_id=client_id_input,
                    topic=topic_input,
                    template_id=template_id,
                )
                if result:
                    # Save version state
                    if "version_id" in result:
                        st.session_state[f"active_version_{template_id}"] = result["version_id"]
                    
                    # Update variables and result state
                    extracted = result.get("extracted_variables", {})
                    st.session_state.dt_ai_variables = extracted
                    st.session_state.dt_user_variables = extracted.copy()
                    
                    st.session_state.dt_report_text = result.get("generated_text", "")
                    st.session_state.dt_generation_result = result
                    st.session_state.dt_draft_generated = True
                    st.success("Draft generated successfully!")
                    st.rerun()

    st.divider()

    if not st.session_state.dt_draft_generated:
        st.info("👆 Review the document above and click **Generate Draft** to continue.")
        return

    col_left, col_right = st.columns([6, 4])

    with col_left:
        st.markdown("### 📄 The Report")
        st.markdown(
            f'<div style="background:#1e1e2e; padding:20px; border-radius:10px; '
            f'border:1px solid #444; font-size:14px; line-height:1.7; '
            f'white-space:pre-wrap; color:#e0e0e0; margin-bottom:20px;">'
            f'{st.session_state.dt_report_text}'
            f'</div>',
            unsafe_allow_html=True,
        )
        
    with col_right:
        st.markdown("### 🔍 Data & Feedback Inspector")
        st.caption("Review extracted variables and provide stylistic feedback")
        
        # Stylistic Feedback
        st.markdown("#### 🗣️ Stylistic Instructions")
        feedback = st.text_input("Adjust the style/tone of this draft...", value=st.session_state.dt_stylistic_feedback, key="dt_style_input")
        if feedback != st.session_state.dt_stylistic_feedback:
            st.session_state.dt_stylistic_feedback = feedback
        
        st.divider()
        
        # Procedural Variables
        st.markdown("#### ⚙️ Extracted Variables")
        procedural_corrections = []
        
        if not st.session_state.dt_ai_variables:
            st.info("No variables extracted.")
        
        for key, ai_val in st.session_state.dt_ai_variables.items():
            # Convert values to strings for the text input
            ai_val_str = str(ai_val) if ai_val is not None else ""
            current_user_val = str(st.session_state.dt_user_variables.get(key, ""))
            
            user_val = st.text_input(
                key, 
                value=current_user_val, 
                key=f"inspector_{key}"
            )
            if user_val != current_user_val:
                st.session_state.dt_user_variables[key] = user_val
            
            if st.session_state.dt_user_variables[key] != ai_val_str:
                procedural_corrections.append({
                    "variable_name": key, 
                    "correction_rule": f"Change {ai_val_str} to {st.session_state.dt_user_variables[key]}"
                })
                st.warning(f"Modified: {ai_val_str} ➡️ {st.session_state.dt_user_variables[key]}")
        
        st.divider()
        
        if st.button("🧠 Approve & Learn", type="primary", use_container_width=True):
            payload_data = {
                "adviser_id": adviser_id,
                "client_id": client_id_input,
                "topic": topic_input,
                "stylistic_feedback": st.session_state.dt_stylistic_feedback,
                "procedural_corrections": procedural_corrections
            }
            
            try:
                # Use the feedback capture endpoint
                url = f"{client.base_url}/feedback/capture"
                with st.spinner("Updating Atlas Memory..."):
                    response = httpx.post(
                        url,
                        json=payload_data,
                        timeout=10.0,
                        headers=client.headers
                    )
                    response.raise_for_status()
                st.success("Atlas has updated its memory.")
                st.toast("Atlas has updated its memory.")
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(f"Feedback capture error: {e}")
                st.error(f"Engine offline. Could not connect to Atlas: {e}")

def render_template_engine(client: TemplateAPIClient) -> None:
    """Render the complete template engine workflow."""
    init_template_session_state()

    st.title("Template Engine")
    st.markdown("Transform Word documents into dynamic Jinja2 templates")
    st.divider()

    # Mode selection
    mode = st.radio(
        "Select Mode",
        ["📤 Batch Upload", "📊 Status Dashboard", "🧠 Review & Learn", "⚙️ Prompt Settings"],
        horizontal=True,
    )

    if mode == "📤 Batch Upload":
        render_template_upload(client)

    elif mode == "📊 Status Dashboard":
        render_injection_queue_dashboard(client)

    elif mode == "🧠 Review & Learn":
        # from report_ui import render_director_typist
        render_director_typist_ui(client)

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
            st.warning(
                "💡 **API is asleep** - Click the link below to wake it up, then refresh:\n\n"
                f"[🔗 Open {API_BASE_URL}]({API_BASE_URL})"
            )

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
