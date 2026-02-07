"""Streamlit frontend for Template Intelligence Engine.

Provides a UI for uploading Word templates, detecting variables,
extracting conditional logic, and generating tagged templates.
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
from typing_extensions import TypedDict

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
ORG_ID = os.getenv("ORG_ID", "bf0d03fb-d8ea-4377-a991-b3b5818e71ec")

# Page config
st.set_page_config(
    page_title="Template Intelligence Engine",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Type Definitions
# =============================================================================


class DetectedVariable(TypedDict):
    """A variable detected in the template."""
    original_text: str
    suggested_variable_name: str
    rationale: str
    paragraph_index: int


class LogicTag(TypedDict):
    """A conditional logic tag extracted from the template."""
    original_text: str
    paragraph_index: int
    is_conditional: bool
    condition_variable: str
    operator: str
    condition_value: str
    jinja_wrapper: str
    confidence: float
    reasoning: str


class TemplateAnalysisResult(TypedDict):
    """Result from template analysis."""
    template_id: str
    filename: str
    detected_variables: list[DetectedVariable]
    total_paragraphs: int
    analyzed_at: str


class LogicExtractionResult(TypedDict):
    """Result from logic extraction."""
    logic_tags: list[LogicTag]
    total_paragraphs: int
    candidates_analyzed: int
    extracted_at: str
    extraction_method: str


# =============================================================================
# API Client
# =============================================================================


class TemplateAPIClient:
    """API client for template management endpoints."""

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

    def analyze_template(
        self,
        file: UploadedFile,
        normalize: bool = True,
    ) -> TemplateAnalysisResult | None:
        """Analyze a Word template to detect dynamic variables.

        Args:
            file: The uploaded file from Streamlit.
            normalize: Whether to normalize the document before analysis.

        Returns:
            TemplateAnalysisResult with detected variables, or None if failed.
        """
        url = f"{self.base_url}/templates/analyze?normalize={normalize}"

        files = {"file": (file.name, file.getvalue(), file.type)}

        try:
            with st.spinner("Analyzing template..."):
                response = httpx.post(url, headers=self.headers, files=files, timeout=60.0)
                response.raise_for_status()
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

    def extract_logic(
        self,
        file: UploadedFile,
        available_variables: list[str],
    ) -> LogicExtractionResult | None:
        """Extract conditional logic from a Word template.

        Args:
            file: The uploaded file from Streamlit.
            available_variables: List of variable names detected in the document.

        Returns:
            LogicExtractionResult with extracted logic tags, or None if failed.
        """
        # Build query parameters
        params = {"available_variables": available_variables}
        url = f"{self.base_url}/templates/extract-logic"

        files = {"file": (file.name, file.getvalue(), file.type)}

        try:
            with st.spinner("Extracting conditional logic..."):
                response = httpx.post(
                    url,
                    headers=self.headers,
                    files=files,
                    params=params,
                    timeout=60.0,
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Logic extraction failed: {e.response.status_code} - {e.response.text}")
            st.error(f"Logic extraction failed: {e.response.status_code}")
            st.error(e.response.text)
            return None
        except Exception as e:
            logger.error(f"Logic extraction error: {e}")
            st.error(f"Logic extraction error: {e}")
            return None

    def finalize_template(
        self,
        template_id: str,
        variables: list[DetectedVariable],
        original_filename: str,
    ) -> dict[str, Any] | None:
        """Finalize template by injecting Jinja2 tags.

        Args:
            template_id: The template ID from analysis.
            variables: Reviewed list of variables to inject.
            original_filename: Original filename to locate the temp file.

        Returns:
            Dict with template status and download URL, or None if failed.
        """
        url = f"{self.base_url}/templates/finalize"

        payload = {
            "template_id": template_id,
            "variables": variables,
            "original_filename": original_filename,
        }

        try:
            with st.spinner("Finalizing template..."):
                response = httpx.post(url, headers=self.headers, json=payload, timeout=30.0)
                response.raise_for_status()
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
        """Download the finalized template.

        Args:
            template_id: The template ID to download.

        Returns:
            Template file content as bytes, or None if failed.
        """
        url = f"{self.base_url}/templates/download/{template_id}"

        try:
            response = httpx.get(url, headers=self.headers, timeout=30.0)
            response.raise_for_status()
            return response.content
        except httpx.HTTPStatusError as e:
            logger.error(f"Download failed: {e.response.status_code}")
            st.error(f"Download failed: {e.response.status_code}")
            return None
        except Exception as e:
            logger.error(f"Download error: {e}")
            st.error(f"Download error: {e}")
            return None

    # =============================================================================
    # Batch Processing Methods
    # =============================================================================

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

    def get_template_status(self, template_id: str) -> dict | None:
        """Get processing status for a specific template.

        Args:
            template_id: The template ID to check.

        Returns:
            Status dict with progress and state, or None if failed.
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
        url = f"{self.base_url}/templates/list-ready?status=completed&download_ready=false&page=1&page_size=100"

        try:
            response = httpx.get(url, headers=self.headers, timeout=10.0)
            response.raise_for_status()
            result = response.json()
            return result.get("templates", [])
        except httpx.HTTPStatusError as e:
            logger.error(f"List ready templates failed: {e.response.status_code}")
            return None
        except Exception as e:
            logger.error(f"List ready templates error: {e}")
            return None

    def get_ready_for_download(self) -> list[dict] | None:
        """Get templates ready for download.

        Returns:
            List of templates with download_ready=True.
        """
        url = f"{self.base_url}/templates/list-ready?download_ready=true&page=1&page_size=100"

        try:
            response = httpx.get(url, headers=self.headers, timeout=10.0)
            response.raise_for_status()
            result = response.json()
            return result.get("templates", [])
        except httpx.HTTPStatusError as e:
            logger.error(f"List download templates failed: {e.response.status_code}")
            return None
        except Exception as e:
            logger.error(f"List download templates error: {e}")
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
    ) -> bool:
        """Queue template finalization with variable injection (async).

        Args:
            template_id: The template ID to finalize.
            variables: Variables with values to inject.

        Returns:
            True if queued successfully, False otherwise.
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
                return True
        except httpx.HTTPStatusError as e:
            logger.error(f"Async finalization failed: {e.response.status_code} - {e.response.text}")
            st.error(f"Finalization failed: {e.response.status_code}")
            return False
        except Exception as e:
            logger.error(f"Async finalization error: {e}")
            st.error(f"Finalization error: {e}")
            return False


# =============================================================================
# UI Components
# =============================================================================


def generate_random_values(variables: list[dict]) -> dict[str, str]:
    """Generate random values based on variable name patterns.

    Args:
        variables: List of detected variables.

    Returns:
        Dict mapping variable names to random values.
    """
    import random
    from datetime import datetime, timedelta

    first_names = ["John", "Jane", "Bob", "Alice", "Charlie", "Diana", "Edward", "Fiona"]
    last_names = ["Smith", "Doe", "Johnson", "Williams", "Brown", "Jones", "Davis", "Miller"]
    streets = ["Main Street", "High Street", "Church Road", "Queen's Road", "Park Avenue", "King Street"]
    cities = ["London", "Manchester", "Birmingham", "Leeds", "Glasgow", "Bristol", "Liverpool"]
    uk_postcodes = ["SW1A 1AA", "M1 1AA", "B1 1AA", "LS1 1AA", "G1 1AA", "BS1 1AA", "L1 1AA"]

    random_values = {}
    for var in variables:
        var_name = var.get("suggested_variable_name", "")
        var_lower = var_name.lower()

        # Generate random values based on variable type
        if "name" in var_lower or "client" in var_lower:
            if "first" in var_lower or "forename" in var_lower:
                random_values[var_name] = random.choice(first_names)
            elif "last" in var_lower or "surname" in var_lower:
                random_values[var_name] = random.choice(last_names)
            else:
                random_values[var_name] = f"{random.choice(first_names)} {random.choice(last_names)}"
        elif "date" in var_lower:
            rand_date = datetime.now() + timedelta(days=random.randint(-365, 365))
            random_values[var_name] = rand_date.strftime("%d/%m/%Y")
        elif "amount" in var_lower or "monetary" in var_lower or "balance" in var_lower or "value" in var_lower:
            amount = random.randint(1000, 500000)
            random_values[var_name] = f"£{amount:,.2f}"
        elif "percentage" in var_lower or "percent" in var_lower or "rate" in var_lower:
            if "rate" in var_lower:
                random_values[var_name] = f"{random.randint(1, 100)}.{random.randint(0, 99):02d}%"
            else:
                random_values[var_name] = f"{random.randint(1, 100)}%"
        elif "address" in var_lower:
            house_num = random.randint(1, 999)
            street = random.choice(streets)
            city = random.choice(cities)
            postcode = random.choice(uk_postcodes)
            random_values[var_name] = f"{house_num} {street}, {city}, {postcode}"
        elif "email" in var_lower:
            name = random.choice(first_names).lower()
            domain = random.choice(["gmail.com", "yahoo.com", "outlook.com", "hotmail.com"])
            random_values[var_name] = f"{name}.{random.choice(last_names).lower()}@{domain}"
        elif "phone" in var_lower or "telephone" in var_lower:
            random_values[var_name] = f"+44 {random.randint(1000, 9999)} {random.randint(100000, 999999)}"
        elif "reference" in var_lower or "ref" in var_lower or "number" in var_lower:
            random_values[var_name] = f"REF-{random.randint(100000, 999999)}"
        elif "age" in var_lower:
            random_values[var_name] = str(random.randint(18, 85))
        elif "gender" in var_lower or "sex" in var_lower:
            random_values[var_name] = random.choice(["Male", "Female"])
        elif "national" in var_lower or "ni" in var_lower:
            random_values[var_name] = f"{random.randint(100000, 999999)} {random.choice(['A', 'B', 'C'])}"
        else:
            # Fallback to placeholder based on variable name
            random_values[var_name] = f"[{var_name}]"

    return random_values


def render_sidebar(client: TemplateAPIClient) -> None:
    """Render the sidebar with connection status and help.

    Args:
        client: The API client instance.
    """
    with st.sidebar:
        st.title("📝 Template Intelligence")

        st.divider()

        # Connection status
        is_healthy = client.health_check()
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
            value=client.org_id,
            disabled=True,
            help="Your organization ID for multi-tenancy isolation",
        )

        st.divider()

        # Workflow info
        st.subheader("Workflow")
        st.markdown("""
        1. **Upload** - Upload one or more Word templates (.docx)
        2. **Processing Status** - Monitor analysis progress in real-time
        3. **Inject Variables** - Fill variables manually or with random values
        4. **Download** - Get your tagged Jinja2 templates

        Templates are processed asynchronously in batches.
        """)

        st.divider()

        # Settings link
        st.caption(f"API: `{API_BASE_URL}`")


def render_upload_section(client: TemplateAPIClient) -> None:
    """Render the batch file upload section.

    Args:
        client: The API client instance.
    """
    st.subheader("📤 Upload Templates")

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

    # Normalize checkbox (applies to all files)
    normalize = st.checkbox(
        "Normalize documents",
        value=True,
        help="Fix broken runs caused by spell-checker and revision tracking",
    )

    # Upload button
    if uploaded_files and st.button("🚀 Upload & Analyze", type="primary"):
        # Determine if we should queue as next batch or create new batch
        if current_batch_id and batch_status and batch_status.get("in_progress", 0) > 0:
            # Queue as next batch (auto-starts when current completes)
            next_batch_id = client.queue_next_batch(uploaded_files, current_batch_id)
            if next_batch_id:
                st.success(f"✅ Files queued for next batch: `{next_batch_id}`")
                st.session_state.active_tab = "Processing Status"
                st.rerun()
        else:
            # Create new batch
            batch_id = client.analyze_batch(uploaded_files)
            if batch_id:
                st.session_state.current_batch_id = batch_id
                st.success(f"✅ Files uploaded! Batch ID: `{batch_id}`")
                st.session_state.active_tab = "Processing Status"
                st.rerun()


def render_analysis_results(result: TemplateAnalysisResult) -> list[DetectedVariable]:
    """Render and allow editing of analysis results.

    Args:
        result: The template analysis result.

    Returns:
        The edited list of variables.
    """
    st.subheader("🔍 Step 2: Review Detected Variables")

    # Summary stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Variables Detected", len(result["detected_variables"]))
    with col2:
        st.metric("Total Paragraphs", result["total_paragraphs"])
    with col3:
        st.metric("Template ID", result["template_id"][:8] + "...")

    st.divider()

    if not result["detected_variables"]:
        st.warning("No variables detected in the template.")
        return []

    # Instructions
    st.info(
        "💡 **Tip**: Review the detected variables below. "
        "You can edit variable names, remove false positives, or add new variables."
    )

    # Editable table
    edited_vars = []

    for i, var in enumerate(result["detected_variables"]):
        with st.expander(
            f"Paragraph {var['paragraph_index']}: `{var['original_text'][:50]}...`",
            expanded=i < 3,  # Expand first 3 by default
        ):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.write("**Original Text:**")
                st.code(var["original_text"], language="text")

                new_name = st.text_input(
                    "Variable Name",
                    value=var["suggested_variable_name"],
                    key=f"name_{i}",
                    help="Snake_case variable name for Jinja2",
                )

                new_rationale = st.text_area(
                    "Rationale",
                    value=var["rationale"],
                    key=f"rationale_{i}",
                    help="Why this was identified as dynamic",
                )

            with col2:
                keep = st.checkbox(
                    "Keep in Template",
                    value=True,
                    key=f"keep_{i}",
                )

                st.write("**Paragraph Index:**")
                st.write(var["paragraph_index"])

            if keep:
                edited_vars.append({
                    "original_text": var["original_text"],
                    "suggested_variable_name": new_name,
                    "rationale": new_rationale,
                    "paragraph_index": var["paragraph_index"],
                })

    # Summary
    st.divider()
    st.info(f"✅ **{len(edited_vars)} variables** will be injected into the template.")

    return edited_vars


def render_logic_extraction(
    client: TemplateAPIClient,
    uploaded_file: UploadedFile,
    available_variables: list[str],
) -> list[LogicTag] | None:
    """Render logic extraction section.

    Args:
        client: The API client instance.
        uploaded_file: The uploaded template file.
        available_variables: List of detected variable names.

    Returns:
        List of extracted logic tags, or None if extraction was skipped.
    """
    st.subheader("🧠 Step 3: Extract Conditional Logic (Optional)")

    st.info(
        "ℹ️ This step detects conditional clauses (e.g., 'if savings exceed £1,073,100') "
        "and converts them to Jinja2 tags (e.g., `{% if savings > 1073100 %}`)."
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        extract_logic = st.checkbox(
            "Extract conditional logic",
            value=False,
            help="Use DSPy to detect conditional clauses",
        )

    if not extract_logic:
        st.info("⏭️ Skipping logic extraction. Click 'Continue' to proceed.")
        return None

    with col2:
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Minimum confidence for including extracted logic",
        )

    if st.button("🚀 Extract Logic", type="primary"):
        if not available_variables:
            st.error("❌ No variables available. Please complete Step 2 first.")
            return None

        # Extract logic
        result = client.extract_logic(uploaded_file, available_variables)

        if result:
            st.success(f"✅ Extracted {len(result['logic_tags'])} logic tags")

            # Display results
            if result["logic_tags"]:
                for tag in result["logic_tags"]:
                    # Filter by confidence
                    if tag["confidence"] < confidence_threshold:
                        continue

                    with st.expander(
                        f"📍 Paragraph {tag['paragraph_index']}: {tag['jinja_wrapper']}"
                    ):
                        col1, col2 = st.columns([2, 1])

                        with col1:
                            st.write("**Original Text:**")
                            st.write(tag["original_text"])

                            st.write("**Jinja2 Tag:**")
                            st.code(tag["jinja_wrapper"], language="jinja2")

                        with col2:
                            # Confidence meter
                            conf_color = "🟢" if tag["confidence"] > 0.8 else "🟡" if tag["confidence"] > 0.6 else "🔴"
                            st.metric(
                                "Confidence",
                                f"{tag['confidence']:.0%}",
                                help=tag["reasoning"],
                            )

                            st.write("**Variable:**")
                            st.write(f"`{tag['condition_variable']}`")

                            st.write("**Operator:**")
                            st.write(f"`{tag['operator']}`")

                            st.write("**Value:**")
                            st.write(f"`{tag['condition_value']}`")

                # Filter logic tags by confidence
                filtered_tags = [
                    tag for tag in result["logic_tags"]
                    if tag["confidence"] >= confidence_threshold
                ]

                st.info(
                    f"📊 Summary: {len(result['logic_tags'])} total tags, "
                    f"{len(filtered_tags)} above threshold ({confidence_threshold:.0%})"
                )

                return filtered_tags
            else:
                st.warning("⚠️ No conditional logic detected in this template.")
                return []

    return None


def render_finalize_section(
    client: TemplateAPIClient,
    result: TemplateAnalysisResult,
    variables: list[DetectedVariable],
    logic_tags: list[LogicTag] | None = None,
) -> bool:
    """Render finalize and download section.

    Args:
        client: The API client instance.
        result: The template analysis result.
        variables: The edited list of variables.
        logic_tags: Optional list of extracted logic tags.

    Returns:
        True if template was finalized successfully.
    """
    st.subheader("✨ Step 4: Finalize & Download")

    st.divider()

    # Summary
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Variables", len(variables))

    with col2:
        if logic_tags is not None:
            st.metric("Logic Tags", len(logic_tags))
        else:
            st.metric("Logic Tags", "N/A", "Skipped")

    with col3:
        st.metric("Output", "Tagged .docx")

    st.divider()

    # Finalize button
    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("🎯 Generate Template", type="primary", use_container_width=True):
            if not variables:
                st.error("❌ No variables to inject. Please review Step 2.")
                return False

            # Call finalize endpoint
            finalize_result = client.finalize_template(
                template_id=result["template_id"],
                variables=variables,
                original_filename=result["filename"],
            )

            if finalize_result:
                st.success(f"✅ Template generated successfully!")
                st.info(f"📥 Template ID: `{finalize_result['template_id']}`")

                # Store in session state for download
                st.session_state.finalized_template_id = finalize_result["template_id"]
                st.session_state.finalized_filename = result["filename"].replace(
                    ".docx", "_tagged.docx"
                )
                st.session_state.show_download = True

                st.rerun()

    with col2:
        # Download button (shown after finalization)
        if st.session_state.get("show_download", False):
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

    return st.session_state.get("show_download", False)


def render_usage_guide(variables: list[DetectedVariable]) -> None:
    """Render usage guide for the tagged template.

    Args:
        variables: The list of injected variables.
    """
    with st.expander("📖 How to Use Your Template", expanded=False):
        st.markdown("""
        ### Python with docxtpl

        ```python
        from docxtpl import DocxTemplate

        # Load the tagged template
        doc = DocxTemplate("tagged_template.docx")

        # Prepare context
        context = {
        """)
        for var in variables[:5]:  # Show first 5
            st.code(f"            '{var['suggested_variable_name']}': '',", language="python")
        if len(variables) > 5:
            st.code(f"            # ... and {len(variables) - 5} more", language="python")

        st.markdown("""
        }

        # Render template
        doc.render(context)
        doc.save("output.docx")
        ```
        """)


def render_processing_status_tab(client: TemplateAPIClient) -> None:
    """Render the Processing Status tab.

    Args:
        client: The API client instance.
    """
    st.header("📊 Processing Status")

    # Get current batch
    batch_id = st.session_state.get("current_batch_id")
    if not batch_id:
        st.info("ℹ️ No uploads in progress. Go to the **Upload** tab to get started.")
        return

    # Fetch batch status
    batch_status = client.get_batch_status(batch_id)

    if not batch_status:
        st.error(f"❌ Could not fetch status for batch `{batch_id}`")
        st.button("🔄 Clear Batch", on_click=lambda: st.session_state.update({"current_batch_id": None}))
        return

    # Show aggregate progress
    st.subheader(f"Batch: `{batch_id}`")

    total = batch_status.get("total_templates", 0)
    completed = batch_status.get("completed", 0)
    failed = batch_status.get("failed", 0)
    in_progress = batch_status.get("in_progress", 0)

    # Progress bar
    if total > 0:
        progress_ratio = completed / total
        st.progress(progress_ratio)
        st.caption(f"**{completed}** of **{total}** completed")

    # Stats row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total", total)
    with col2:
        st.metric("Completed", completed, delta_color="normal" if completed > 0 else "off")
    with col3:
        st.metric("In Progress", in_progress, delta_color="normal" if in_progress > 0 else "off")
    with col4:
        st.metric("Failed", failed, delta_color="inverse" if failed > 0 else "normal")

    st.divider()

    # Show next batch info if exists
    next_batch_id = batch_status.get("next_batch_id")
    if next_batch_id:
        st.info(f"📋 **Next batch queued**: `{next_batch_id}`")
        st.caption(f"{batch_status.get('queued', 0)} file(s) waiting to process")
        st.divider()

    # Show individual file status
    st.subheader("Template Status")

    templates = batch_status.get("templates", [])
    if not templates:
        st.warning("⚠️ No templates found in this batch.")
        return

    for template in templates:
        status = template.get("status", "unknown").upper()
        filename = template.get("filename", "Unknown")
        template_id = template.get("template_id")

        # Status icon
        status_icons = {
            "QUEUED": "⏳",
            "ANALYZING": "🔍",
            "COMPLETED": "✅",
            "FAILED": "❌",
            "FINALIZING": "⚙️",
        }
        icon = status_icons.get(status, "📄")

        with st.expander(f"{icon} {filename} - {status}", expanded=False):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.write(f"**Status:** {status}")
                st.write(f"**ID:** `{str(template_id)[:8]}...`")

            with col2:
                progress = template.get("progress")
                if progress is not None:
                    st.progress(progress / 100)
                    st.caption(f"Progress: {progress}%")
                else:
                    st.caption("No progress data")

                error_msg = template.get("error_message")
                if error_msg:
                    st.error(f"Error: {error_msg}")

            with col3:
                if template.get("download_ready"):
                    if st.button("⬇️ Download", key=f"dl_{template_id}"):
                        content = client.download_template(str(template_id))
                        if content:
                            dl_filename = filename.replace(".docx", "_tagged.docx")
                            st.download_button(
                                label="💾 Save File",
                                data=content,
                                file_name=dl_filename,
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            )

    # Auto-refresh
    time.sleep(3)
    st.rerun()


def render_inject_variables_tab(client: TemplateAPIClient) -> None:
    """Render the Inject Variables tab.

    Args:
        client: The API client instance.
    """
    st.header("💉 Inject Variables")

    # Fetch templates ready for injection
    ready_templates = client.get_ready_for_injection()

    if not ready_templates:
        st.info("ℹ️ No templates ready for injection.")
        st.caption("Templates that have completed analysis will appear here.")
        return

    st.write(f"**{len(ready_templates)}** template(s) ready for variable injection")

    # Template selector
    template_options = {t["filename"]: t for t in ready_templates}
    selected_filename = st.selectbox(
        "Select template to inject",
        options=list(template_options.keys()),
    )

    if not selected_filename:
        return

    selected = template_options[selected_filename]
    template_id = str(selected.get("template_id"))

    # Get template status to retrieve detected variables
    template_status = client.get_template_status(template_id)
    if not template_status:
        st.error("❌ Could not fetch template details")
        return

    detected_vars = template_status.get("detected_variables")

    if not detected_vars:
        st.warning("⚠️ No variables detected in this template.")
        return

    st.divider()
    st.subheader(f"Variables: {selected_filename}")

    # Variables input form
    edited_variables = {}

    for i, var in enumerate(detected_vars):
        var_name = var.get("suggested_variable_name", "")
        rationale = var.get("rationale", "")

        col1, col2 = st.columns([3, 1])

        with col1:
            value = st.text_input(
                var_name,
                key=f"var_{template_id}_{i}",
                label_visibility="visible",
                placeholder=f"Enter value for {var_name}",
            )
            edited_variables[var_name] = value

        with col2:
            st.caption(rationale)

    st.divider()

    # Action buttons
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        # Fill Random Values button
        if st.button("🎲 Fill Random Values", use_container_width=True):
            random_values = generate_random_values(detected_vars)
            for i, var_name in enumerate(random_values):
                key = f"var_{template_id}_{i}"
                st.session_state[key] = random_values[var_name]
            st.success("✅ Random values filled! Review and edit as needed, then click Inject.")
            st.rerun()

    with col2:
        # Inject button
        has_values = any(v for v in edited_variables.values())
        if st.button("💉 Inject Variables", type="primary", use_container_width=True, disabled=not has_values):
            # Prepare variables list for API
            variables_payload = []
            for var in detected_vars:
                var_name = var.get("suggested_variable_name")
                value = edited_variables.get(var_name, "")
                variables_payload.append({
                    "suggested_variable_name": var_name,
                    "original_text": var.get("original_text", ""),
                    "value": value,
                })

            # Queue finalization
            success = client.finalize_template_async(template_id, variables_payload)
            if success:
                st.success("✅ Injection queued! Check **Processing Status** for progress.")
                st.session_state.active_tab = "Processing Status"
                st.rerun()

    with col3:
        # Quick random inject button
        if st.button("🚀 Random & Inject", use_container_width=True):
            success = client.inject_random_values(template_id)
            if success:
                st.success("✅ Random values injected! Check **Processing Status** for progress.")
                st.session_state.active_tab = "Processing Status"
                st.rerun()


def render_download_tab(client: TemplateAPIClient) -> None:
    """Render the Download tab.

    Args:
        client: The API client instance.
    """
    st.header("⬇️ Download")

    # Fetch templates ready for download
    ready_templates = client.get_ready_for_download()

    if not ready_templates:
        st.info("ℹ️ No templates ready for download.")
        st.caption("Completed templates will appear here after injection.")
        return

    st.write(f"**{len(ready_templates)}** template(s) ready for download")

    st.divider()

    # Show as table with download buttons
    for template in ready_templates:
        filename = template.get("filename", "Unknown")
        template_id = str(template.get("template_id"))
        completed_at = template.get("processing_completed_at", "Unknown")

        with st.container():
            col1, col2, col3 = st.columns([4, 2, 1])

            with col1:
                st.write(f"**{filename}**")
                st.caption(f"Completed: {completed_at}")

            with col2:
                # Variable count if available
                detected_vars = template.get("detected_variables")
                if detected_vars:
                    if isinstance(detected_vars, dict):
                        var_count = len(detected_vars)
                    elif isinstance(detected_vars, list):
                        var_count = len(detected_vars)
                    else:
                        var_count = "?"
                    st.write(f"Variables: **{var_count}**")

            with col3:
                if st.button("⬇️", key=f"dl_tab_{template_id}", help="Download"):
                    content = client.download_template(template_id)
                    if content:
                        dl_filename = filename.replace(".docx", "_tagged.docx")
                        st.download_button(
                            label="💾",
                            data=content,
                            file_name=dl_filename,
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            help="Save file",
                        )

        st.divider()


# =============================================================================
# Main App
# =============================================================================


def init_session_state() -> None:
    """Initialize session state variables."""
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
    if "edited_variables" not in st.session_state:
        st.session_state.edited_variables = None
    if "logic_tags" not in st.session_state:
        st.session_state.logic_tags = None
    if "show_download" not in st.session_state:
        st.session_state.show_download = False
    # New session state for batch processing
    if "current_batch_id" not in st.session_state:
        st.session_state.current_batch_id = None
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Upload"


def main() -> None:
    """Main application entry point."""

    # Initialize session state
    init_session_state()

    # Initialize API client
    client = TemplateAPIClient(API_BASE_URL, ORG_ID)

    # Render sidebar
    render_sidebar(client)

    # Main content area
    st.title("Template Intelligence Engine")
    st.markdown("Transform Word documents into dynamic Jinja2 templates")

    st.divider()

    # Tabbed interface
    tabs = st.tabs([
        "📤 Upload",
        "📊 Processing Status",
        "💉 Inject Variables",
        "⬇️ Download",
    ])

    # Track active tab in session state
    current_tab = st.session_state.get("active_tab", "Upload")
    tab_index = {
        "Upload": 0,
        "Processing Status": 1,
        "Inject Variables": 2,
        "Download": 3,
    }.get(current_tab, 0)

    with tabs[0]:
        st.session_state.active_tab = "Upload"
        render_upload_section(client)

    with tabs[1]:
        if st.session_state.active_tab == "Processing Status" or tab_index == 1:
            st.session_state.active_tab = "Processing Status"
            render_processing_status_tab(client)

    with tabs[2]:
        if st.session_state.active_tab == "Inject Variables" or tab_index == 2:
            st.session_state.active_tab = "Inject Variables"
            render_inject_variables_tab(client)

    with tabs[3]:
        if st.session_state.active_tab == "Download" or tab_index == 3:
            st.session_state.active_tab = "Download"
            render_download_tab(client)


if __name__ == "__main__":
    main()
