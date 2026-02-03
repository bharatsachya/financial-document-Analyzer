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
        except Exception as e:
            logger.error(f"Download error: {e}")
            st.error(f"Download error: {e}")
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


def render_template_upload(client: TemplateAPIClient) -> tuple[UploadedFile | None, bool]:
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
        analyze_button = st.button("Analyze", type="primary", use_container_width=True)

    if uploaded_file:
        st.info(f"📄 Selected: `{uploaded_file.name}`")

        if analyze_button:
            result = client.analyze_template(uploaded_file)
            if result:
                st.session_state.template_analysis_result = result
                st.session_state.template_uploaded_file = uploaded_file
                st.session_state.template_variables = None
                st.session_state.template_client_data = {}
                st.session_state.show_template_download = False
                st.rerun()

    return uploaded_file, analyze_button


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
    """Render client data input form for each variable."""
    st.subheader("👤 Step 3: Provide Client Data")

    st.info(
        "📝 Enter the client data that will be used to render the final document. "
        "These values will replace the variables in the template."
    )

    st.divider()

    client_data = {}

    # Group variables by type for better organization
    name_vars = [v for v in variables if "name" in v.get("suggested_variable_name", "").lower()]
    date_vars = [v for v in variables if "date" in v.get("suggested_variable_name", "").lower()]
    amount_vars = [v for v in variables if any(x in v.get("suggested_variable_name", "").lower() for x in ["amount", "value", "balance"])]
    other_vars = [v for v in variables if v not in name_vars + date_vars + amount_vars]

    # Display each group
    if name_vars:
        st.markdown("### 👤 Names")
        for var in name_vars:
            var_name = var.get("suggested_variable_name", "")
            default = st.session_state.template_client_data.get(var_name, "")
            client_data[var_name] = st.text_input(
                var_name.replace("_", " ").title(),
                value=default,
                key=f"client_data_{var_name}",
                help=f"Original: {var.get('original_text', '')}",
            )

    if date_vars:
        st.markdown("### 📅 Dates")
        for var in date_vars:
            var_name = var.get("suggested_variable_name", "")
            default = st.session_state.template_client_data.get(var_name, "")
            client_data[var_name] = st.text_input(
                var_name.replace("_", " ").title(),
                value=default,
                key=f"client_data_{var_name}",
                help=f"Original: {var.get('original_text', '')}",
            )

    if amount_vars:
        st.markdown("### 💰 Amounts & Values")
        for var in amount_vars:
            var_name = var.get("suggested_variable_name", "")
            default = st.session_state.template_client_data.get(var_name, "")
            client_data[var_name] = st.text_input(
                var_name.replace("_", " ").title(),
                value=default,
                key=f"client_data_{var_name}",
                help=f"Original: {var.get('original_text', '')}",
            )

    if other_vars:
        st.markdown("### 📋 Other Variables")
        for var in other_vars:
            var_name = var.get("suggested_variable_name", "")
            default = st.session_state.template_client_data.get(var_name, "")
            client_data[var_name] = st.text_input(
                var_name.replace("_", " ").title(),
                value=default,
                key=f"client_data_{var_name}",
                help=f"Original: {var.get('original_text', '')}",
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


def render_template_engine(client: TemplateAPIClient) -> None:
    """Render the complete template engine workflow."""
    init_template_session_state()

    st.title("Template Engine")
    st.markdown("Transform Word documents into dynamic Jinja2 templates")
    st.divider()

    # Step 1: Upload and Analyze
    uploaded_file, _ = render_template_upload(client)

    # Step 2-4: Show after analysis
    if st.session_state.template_analysis_result:
        result = st.session_state.template_analysis_result

        # Step 2: Review Variables
        edited_vars = render_template_variables(result)

        if edited_vars:
            st.session_state.template_variables = edited_vars

            # Step 3: Client Data Input
            client_data = render_client_data_input(edited_vars)

            # Step 4: Finalize and Download
            render_template_finalize(client, result, edited_vars)


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

    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["📥 Document Ingestion", "📝 Template Engine", "⚙️ Settings"])

    with tab1:
        st.title("Document Ingestion Dashboard")
        st.markdown("Upload and track document processing")

        st.divider()
        render_ingestion_upload(ingestion_client)
        st.divider()
        render_ingestion_status(ingestion_client)

    with tab2:
        render_template_engine(template_client)

    with tab3:
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
