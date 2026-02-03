"""Template analyzer strategy.

Analyzes Word templates to detect dynamic variables using regex patterns
or LLM-based analysis with Few-Shot Chain-of-Thought prompting.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

from app.interfaces.template import BaseTemplateAnalyzer
from app.strategies.template_engine.models import DetectedVariable

logger = logging.getLogger(__name__)


# =============================================================================
# Few-Shot Chain-of-Thought Prompt for LLM Analysis
# =============================================================================

TEMPLATE_ANALYSIS_PROMPT = """
You are an expert Document Intelligence Engineer. Analyze the text segment.

DEFINITIONS:
1. Dynamic: Text that changes per client (Names, Dates, Risk Profiles, Amounts).
2. Static: Legal headers, boilerplate, firm branding.

FEW-SHOT EXAMPLES:
Input: "prepared for Mr. James Arlington on 12th March"
Output: { "is_dynamic": true, "extraction": [{ "original": "Mr. James Arlington", "var": "client_name" }, { "original": "12th March", "var": "report_date" }] }

Input: "The value of investments can go down as well as up."
Output: { "is_dynamic": false, "extraction": [] }

TASK:
Analyze the user input. Return valid JSON only.
"""


class TemplateAnalyzer(BaseTemplateAnalyzer):
    """Analyzes Word templates to detect dynamic injection points.

    Uses regex-based pattern matching for common financial document variables.
    Can be extended with LLM-based analysis for more sophisticated detection.
    """

    def __init__(
        self,
        use_llm: bool = False,
        openai_api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = "openai/gpt-4o",
    ) -> None:
        """Initialize the analyzer.

        Args:
            use_llm: Whether to use LLM for analysis (else mock/regex).
            openai_api_key: OpenAI/OpenRouter API key if use_llm=True.
            base_url: API base URL (default: OpenRouter).
            model: Model name to use for LLM analysis.
        """
        self._use_llm = use_llm
        self._openai_api_key = openai_api_key
        self._base_url = base_url
        self._model = model

        # Common patterns for UK financial documents
        self._patterns = {
            # Name patterns
            r"\b(Mr|Mrs|Ms|Dr|Prof|Sir|Dame)\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b": "client_full_name",
            r"\b(Mr|Mrs|Ms|Dr|Prof)\s+[A-Z][a-z]+\b": "client_name",
            # Date patterns
            r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b": "date",
            r"\b\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b": "formatted_date",
            r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}\b": "short_date",
            # Monetary patterns
            r"\b£[$€]?\s?[\d,]+\.?\d{0,2}\b": "monetary_amount",
            r"\b[\d,]+\.?\d{0,2}\s?(GBP|USD|EUR)\b": "currency_amount",
            # Number patterns
            r"\b\d+%\b": "percentage",
            r"\b\d+(?:,\d{3})*(?:\.\d+)?\b": "numeric_value",
            # Address patterns
            r"\d+\s+[A-Z][a-z]+\s+(Street|Road|Lane|Drive|Avenue|Close|Way)\b": "address_line",
        }

        logger.info(
            f"TemplateAnalyzer initialized: use_llm={use_llm}, "
            f"patterns_registered={len(self._patterns)}"
        )

    async def analyze(self, file_path: str) -> list[DetectedVariable]:
        """Analyze document to detect dynamic variables.

        Args:
            file_path: Path to the Word template file.

        Returns:
            List of DetectedVariable objects with paragraph indices.

        Raises:
            FileNotFoundError: If template file doesn't exist.
        """
        logger.info(f"Starting template analysis: {file_path}")

        try:
            # Validate file exists
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"Template file not found: {file_path}")

            # Load document using python-docx
            try:
                from docx import Document
            except ImportError:
                logger.error("python-docx not installed. Run: poetry add python-docx")
                raise ImportError(
                    "python-docx is required for template analysis. "
                    "Install it with: poetry add python-docx"
                )

            doc = Document(file_path)
            detected_vars: list[DetectedVariable] = []

            logger.info(f"Document loaded: {len(doc.paragraphs)} paragraphs")

            # Analyze each paragraph
            for idx, paragraph in enumerate(doc.paragraphs):
                if not paragraph.text.strip():
                    continue

                # Analyze paragraph for dynamic content
                analysis_result = await self._analyze_paragraph(
                    paragraph.text, idx
                )

                if analysis_result:
                    detected_vars.extend(analysis_result)

            logger.info(
                f"Analysis complete: {len(detected_vars)} variables detected "
                f"from {len(doc.paragraphs)} paragraphs"
            )
            return detected_vars

        except FileNotFoundError:
            raise
        except ImportError:
            raise
        except Exception as e:
            logger.error(f"Template analysis failed: {e}", exc_info=True)
            raise RuntimeError(f"Template analysis failed: {e}") from e

    async def _analyze_paragraph(
        self, text: str, paragraph_index: int
    ) -> list[DetectedVariable]:
        """Analyze a single paragraph for dynamic content.

        Why preserve paragraph_index: Handles duplicate text occurrences.
        E.g., "client_name" may appear in headers and body - we need precise location.

        Args:
            text: The paragraph text to analyze.
            paragraph_index: The paragraph's position in the document.

        Returns:
            List of DetectedVariable objects found in this paragraph.
        """
        if self._use_llm:
            return await self._call_llm(text, paragraph_index)
        else:
            return self._regex_analysis(text, paragraph_index)
# ... inside TemplateAnalyzer class ...

    async def _call_llm(
        self, text: str, paragraph_index: int
    ) -> list[DetectedVariable]:
        """
        Executes the Few-Shot Chain-of-Thought analysis using OpenAI.
        """
        if not self._openai_api_key:
            logger.warning("No OpenAI API key provided. Falling back to regex.")
            return self._regex_analysis(text, paragraph_index)

        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(
                api_key=self._openai_api_key,
                base_url=self._base_url,
            )

            # Combine the Few-Shot Prompt with the User Input
            full_prompt = f"{TEMPLATE_ANALYSIS_PROMPT}\n\nInput: \"{text}\"\nOutput:"

            response = await client.chat.completions.create(
                model=self._model,
                messages=[{"role": "system", "content": full_prompt}],
                response_format={"type": "json_object"}, # CRITICAL: Enforce JSON
                temperature=0.1 # Low temp for deterministic code/json
            )

            content = response.choices[0].message.content
            if not content:
                return []

            # Parse JSON: { "is_dynamic": bool, "extraction": [...] }
            data = json.loads(content)
            
            if not data.get("is_dynamic", False):
                return []

            results = []
            for item in data.get("extraction", []):
                results.append(
                    DetectedVariable(
                        original_text=item.get("original"),
                        suggested_variable_name=item.get("var"),
                        rationale="LLM Few-Shot Analysis",
                        paragraph_index=paragraph_index
                    )
                )
            return results

        except ImportError:
            logger.error("openai library not installed. Run: poetry add openai")
            return self._regex_analysis(text, paragraph_index)
        except Exception as e:
            logger.error(f"LLM Analysis failed for paragraph {paragraph_index}: {e}")
            # Graceful degradation: If LLM fails, try regex
            return self._regex_analysis(text, paragraph_index)
        
    def _regex_analysis(
        self, text: str, paragraph_index: int
    ) -> list[DetectedVariable]:
        """Analyze text using regex patterns (fallback method).

        Detects common patterns in UK financial documents:
        - Client names (Mr/Mrs/Ms + First + Last)
        - Dates (various formats)
        - Monetary amounts (with £, €, $ symbols)
        - Percentages
        - Address patterns

        Args:
            text: The paragraph text to analyze.
            paragraph_index: The paragraph's position.

        Returns:
            List of DetectedVariable objects detected via regex.
        """
        variables = []

        for pattern, var_name in self._patterns.items():
            try:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    # Handle tuple matches from regex groups
                    match_str = (
                        " ".join(match) if isinstance(match, tuple) else str(match)
                    )

                    # Avoid duplicates within same paragraph
                    if any(
                        v.original_text == match_str and v.paragraph_index == paragraph_index
                        for v in variables
                    ):
                        continue

                    variables.append(
                        DetectedVariable(
                            original_text=match_str,
                            suggested_variable_name=var_name,
                            rationale=f"Detected pattern matching {var_name}",
                            paragraph_index=paragraph_index,
                        )
                    )
            except re.error as e:
                logger.warning(f"Regex pattern error for {var_name}: {e}")

        return variables

    @property
    def supported_extensions(self) -> set[str]:
        """Return supported file extensions."""
        return {".docx"}
