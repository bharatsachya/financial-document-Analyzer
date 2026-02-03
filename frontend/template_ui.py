"""Streamlit frontend for Template Intelligence Engine.

Provides a UI for uploading Word templates, detecting variables,
extracting conditional logic, and generating tagged templates.
"""

import io
import logging
import os
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
# UI Components
# =============================================================================


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
        1. **Upload** a Word template (.docx)
        2. **Review** detected variables
        3. **Extract** conditional logic (optional)
        4. **Finalize** to inject Jinja2 tags
        5. **Download** the tagged template

        The template is ready for use with docxtpl or Jinja2.
        """)

        st.divider()

        # Settings link
        st.caption(f"API: `{API_BASE_URL}`")


def render_upload_section(client: TemplateAPIClient) -> tuple[UploadedFile | None, bool]:
    """Render the file upload section.

    Args:
        client: The API client instance.

    Returns:
        Tuple of (uploaded_file, normalize_flag).
    """
    st.subheader("📤 Step 1: Upload Template")

    col1, col2 = st.columns([3, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Choose a Word template",
            type=["docx"],
            label_visibility="collapsed",
            help="Upload a .docx file for analysis",
        )

    with col2:
        st.write("")
        st.write("")
        normalize = st.checkbox(
            "Normalize",
            value=True,
            help="Fix broken runs caused by spell-checker and revision tracking",
        )

    if uploaded_file:
        st.info(f"📄 Selected: `{uploaded_file.name}`")

    return uploaded_file, normalize


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

    # Step 1: Upload
    uploaded_file, normalize = render_upload_section(client)

    # Store file in session if changed
    if uploaded_file and uploaded_file != st.session_state.get("uploaded_file"):
        st.session_state.uploaded_file = uploaded_file
        # Reset other state
        st.session_state.analysis_result = None
        st.session_state.edited_variables = None
        st.session_state.logic_tags = None
        st.session_state.show_download = False

    # Analyze button
    if uploaded_file and st.button("🔍 Analyze Template", type="primary"):
        result = client.analyze_template(uploaded_file, normalize)
        if result:
            st.session_state.analysis_result = result
            st.rerun()

    # Step 2: Review variables (after analysis)
    if st.session_state.analysis_result:
        edited_vars = render_analysis_results(st.session_state.analysis_result)

        if edited_vars:
            st.session_state.edited_variables = edited_vars

            # Step 3: Extract logic
            logic_tags = render_logic_extraction(
                client,
                uploaded_file,
                [v["suggested_variable_name"] for v in edited_vars],
            )

            if logic_tags is not None:
                st.session_state.logic_tags = logic_tags

            # Step 4: Finalize
            render_finalize_section(
                client,
                st.session_state.analysis_result,
                edited_vars,
                st.session_state.logic_tags,
            )

            # Usage guide
            render_usage_guide(edited_vars)


if __name__ == "__main__":
    main()
