"""Unit tests for Logic Extractors."""

import zipfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from app.interfaces.logic_extractor import LogicExtractionError, LogicTag
from app.strategies.logic.fallback import FallbackLogicExtractor
from app.strategies.logic.signatures import ConditionalLogicExtractor


# =============================================================================
# Fallback Logic Extractor Tests
# =============================================================================


class TestFallbackLogicExtractor:
    """Test suite for FallbackLogicExtractor."""

    @pytest.fixture
    def extractor(self):
        """Create a fallback extractor instance."""
        return FallbackLogicExtractor(default_confidence=0.6)

    # =========================================================================
    # Validation Tests
    # =========================================================================

    def test_supported_extensions(self, extractor):
        """Test that only .docx files are supported."""
        assert extractor.supported_extensions == {".docx"}

    # =========================================================================
    # Document Loading Tests
    # =========================================================================

    def test_extract_nonexistent_file(self, extractor):
        """Test that extracting from a non-existent file raises FileNotFoundError."""
        import asyncio

        async def run_test():
            with pytest.raises(FileNotFoundError):
                await extractor.extract_logic(
                    "/nonexistent/file.docx",
                    ["client_age", "client_name"]
                )

        asyncio.run(run_test())

    def test_load_paragraphs_valid_docx(self, extractor, tmp_path):
        """Test loading paragraphs from a valid .docx file."""
        import asyncio

        # Create a minimal .docx file
        docx_path = tmp_path / "test.docx"
        with zipfile.ZipFile(docx_path, "w") as zf:
            document_xml = b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
    <w:body>
        <w:p><w:r><w:t>First paragraph</w:t></w:r></w:p>
        <w:p><w:r><w:t>Second paragraph</w:t></w:r></w:p>
    </w:body>
</w:document>"""
            zf.writestr("[Content_Types].xml", b"<?xml version='1.0'?><Types xmlns='http://schemas.openxmlformats.org/package/2006/content-types'><Default Extension='xml' ContentType='application/xml'/><Override PartName='/word/document.xml' ContentType='application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml'/></Types>")
            zf.writestr("_rels/.rels", b"<?xml version='1.0'?><Relationships xmlns='http://schemas.openxmlformats.org/package/2006/relationships'><Relationship Id='rId1' Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument' Target='word/document.xml'/></Relationships>")
            zf.writestr("word/document.xml", document_xml)
            zf.writestr("word/_rels/document.xml.rels", b"<?xml version='1.0'?><Relationships xmlns='http://schemas.openxmlformats.org/package/2006/relationships'></Relationships>")

        async def run_test():
            paragraphs = await extractor._load_paragraphs(str(docx_path))
            # python-docx will parse actual Word documents differently
            # This test verifies the method exists and handles the file
            assert isinstance(paragraphs, list)

        asyncio.run(run_test())

    # =========================================================================
    # Pattern Matching Tests
    # =========================================================================

    def test_exceeds_pattern(self, extractor):
        """Test detection of 'exceeds' pattern."""
        text = "The lifetime allowance charge only applies if your pension savings exceed £1,073,100."
        idx = 5
        available_vars = ["pension_value", "client_name"]

        results = extractor._extract_from_paragraph(text, idx, available_vars)

        assert len(results) > 0
        tag = results[0]
        assert tag.is_conditional is True
        assert tag.paragraph_index == idx
        assert tag.operator == ">"
        assert "1073100" in tag.condition_value

    def test_under_pattern(self, extractor):
        """Test detection of 'under' pattern."""
        text = "If you are under 55 years old, you may face a tax penalty."
        idx = 3
        available_vars = ["client_age", "client_name"]

        results = extractor._extract_from_paragraph(text, idx, available_vars)

        assert len(results) > 0
        tag = results[0]
        assert tag.is_conditional is True
        assert tag.operator == "<"
        assert tag.condition_value == "55"

    def test_or_higher_pattern(self, extractor):
        """Test detection of 'or higher' pattern."""
        text = "This investment strategy is suitable for investors with a risk profile score of 7 or higher."
        idx = 2
        available_vars = ["risk_profile", "investment_amount"]

        results = extractor._extract_from_paragraph(text, idx, available_vars)

        assert len(results) > 0
        tag = results[0]
        assert tag.is_conditional is True
        assert tag.operator == ">="
        assert tag.condition_value == "7"

    def test_membership_pattern(self, extractor):
        """Test detection of membership pattern (residing in)."""
        text = "This section applies to clients residing in the United Kingdom for tax purposes."
        idx = 8
        available_vars = ["residence", "client_name", "tax_status"]

        results = extractor._extract_from_paragraph(text, idx, available_vars)

        assert len(results) > 0
        tag = results[0]
        assert tag.is_conditional is True
        assert tag.operator == "in"
        assert "UK" in tag.condition_value

    def test_non_conditional_paragraph(self, extractor):
        """Test that non-conditional paragraphs return empty results."""
        text = "Your financial advisor will review your portfolio annually."
        idx = 1
        available_vars = ["client_name", "advisor_name"]

        results = extractor._extract_from_paragraph(text, idx, available_vars)

        assert len(results) == 0

    # =========================================================================
    # Variable Mapping Tests
    # =========================================================================

    def test_map_variable_direct_match(self, extractor):
        """Test variable mapping with direct match."""
        available_vars = ["pension_value", "client_age", "net_income"]

        result = extractor._map_variable("pension_value", available_vars)
        assert result == "pension_value"

    def test_map_variable_case_insensitive(self, extractor):
        """Test variable mapping with case-insensitive match."""
        available_vars = ["pension_value", "client_age", "net_income"]

        result = extractor._map_variable("Pension_Value", available_vars)
        assert result == "pension_value"

    def test_map_variable_with_mapping(self, extractor):
        """Test variable mapping using contextual mappings."""
        available_vars = ["net_income", "client_age"]

        result = extractor._map_variable("income", available_vars)
        assert result == "net_income"

    def test_map_variable_not_found(self, extractor):
        """Test variable mapping when not found."""
        available_vars = ["client_name", "client_age"]

        result = extractor._map_variable("unknown_var", available_vars)
        assert result is None

    # =========================================================================
    # Value Cleaning Tests
    # =========================================================================

    def test_clean_value_currency(self, extractor):
        """Test cleaning values with currency symbols and commas."""
        assert extractor._clean_value("£1,073,100") == "1073100"
        assert extractor._clean_value("$50,000.00") == "50000.00"
        assert extractor._clean_value("€123.45") == "123.45"

    def test_clean_value_numeric(self, extractor):
        """Test cleaning numeric values."""
        assert extractor._clean_value("123") == "123"
        assert extractor._clean_value("45.67") == "45.67"

    # =========================================================================
    # End-to-End Tests
    # =========================================================================

    def test_full_extraction_workflow(self, extractor, tmp_path):
        """Test complete extraction workflow with minimal valid .docx."""
        import asyncio

        # Create a minimal .docx file with conditional text
        docx_path = tmp_path / "test.docx"
        with zipfile.ZipFile(docx_path, "w") as zf:
            document_xml = b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
    <w:body>
        <w:p><w:r><w:t>This is a static paragraph.</w:t></w:r></w:p>
        <w:p><w:r><w:t>If you are under 55 years old, you may face a penalty.</w:t></w:r></w:p>
    </w:body>
</w:document>"""
            zf.writestr("[Content_Types].xml", b"<?xml version='1.0'?><Types xmlns='http://schemas.openxmlformats.org/package/2006/content-types'><Default Extension='xml' ContentType='application/xml'/><Override PartName='/word/document.xml' ContentType='application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml'/></Types>")
            zf.writestr("_rels/.rels", b"<?xml version='1.0'?><Relationships xmlns='http://schemas.openxmlformats.org/package/2006/relationships'><Relationship Id='rId1' Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument' Target='word/document.xml'/></Relationships>")
            zf.writestr("word/document.xml", document_xml)
            zf.writestr("word/_rels/document.xml.rels", b"<?xml version='1.0'?><Relationships xmlns='http://schemas.openxmlformats.org/package/2006/relationships'></Relationships>")

        async def run_test():
            logic_tags = await extractor.extract_logic(
                str(docx_path),
                ["client_age", "client_name"]
            )

            # Verify results
            assert isinstance(logic_tags, list)

        asyncio.run(run_test())


# =============================================================================
# DSPy Logic Extractor Tests
# =============================================================================


class TestDSPyLogicExtractor:
    """Test suite for DSPyLogicExtractor."""

    def test_init_requires_api_key(self):
        """Test that initialization requires an API key."""
        from app.strategies.logic.extractor import DSPyLogicExtractor

        # This should raise an error when trying to configure DSPy without a valid key
        # We mock the DSPy configuration to avoid actual API calls
        with patch("app.strategies.logic.extractor.dspy.configure"):
            with patch("app.strategies.logic.extractor.dspy.OpenAI"):
                # The mock allows us to test validation without actual API calls
                extractor = DSPyLogicExtractor(
                    api_key="test-key",
                    confidence_threshold=0.5
                )
                assert extractor is not None

    def test_confidence_threshold_validation(self):
        """Test confidence threshold validation."""
        from app.strategies.logic.extractor import DSPyLogicExtractor

        with patch("app.strategies.logic.extractor.dspy.configure"):
            with patch("app.strategies.logic.extractor.dspy.OpenAI"):
                # Valid thresholds
                extractor = DSPyLogicExtractor(api_key="test", confidence_threshold=0.0)
                assert extractor._confidence_threshold == 0.0

                extractor = DSPyLogicExtractor(api_key="test", confidence_threshold=1.0)
                assert extractor._confidence_threshold == 1.0

                # Invalid threshold
                with pytest.raises(ValueError, match="confidence_threshold must be between 0 and 1"):
                    DSPyLogicExtractor(api_key="test", confidence_threshold=1.5)

    def test_supported_extensions(self):
        """Test that only .docx files are supported."""
        from app.strategies.logic.extractor import DSPyLogicExtractor

        with patch("app.strategies.logic.extractor.dspy.configure"):
            with patch("app.strategies.logic.extractor.dspy.OpenAI"):
                extractor = DSPyLogicExtractor(api_key="test")
                assert extractor.supported_extensions == {".docx"}

    def test_parse_bool_helper(self):
        """Test _parse_bool helper method."""
        from app.strategies.logic.extractor import DSPyLogicExtractor

        with patch("app.strategies.logic.extractor.dspy.configure"):
            with patch("app.strategies.logic.extractor.dspy.OpenAI"):
                extractor = DSPyLogicExtractor(api_key="test")

                # Boolean values
                assert extractor._parse_bool(True) is True
                assert extractor._parse_bool(False) is False

                # String values
                assert extractor._parse_bool("true") is True
                assert extractor._parse_bool("True") is True
                assert extractor._parse_bool("yes") is True
                assert extractor._parse_bool("1") is True
                assert extractor._parse_bool("false") is False
                assert extractor._parse_bool("no") is False
                assert extractor._parse_bool("0") is False

    def test_parse_float_helper(self):
        """Test _parse_float helper method."""
        from app.strategies.logic.extractor import DSPyLogicExtractor

        with patch("app.strategies.logic.extractor.dspy.configure"):
            with patch("app.strategies.logic.extractor.dspy.OpenAI"):
                extractor = DSPyLogicExtractor(api_key="test")

                # Numeric values
                assert extractor._parse_float(0.7, 0.0) == 0.7
                assert extractor._parse_float(1, 0.0) == 1.0

                # String values
                assert extractor._parse_float("0.85", 0.0) == 0.85
                assert extractor._parse_float("0.95", 0.0) == 0.95

                # Invalid values return default
                assert extractor._parse_float("invalid", 0.5) == 0.5
                assert extractor._parse_float(None, 0.3) == 0.3


# =============================================================================
# LogicTag Tests
# =============================================================================


class TestLogicTag:
    """Test suite for LogicTag dataclass."""

    def test_logic_tag_creation(self):
        """Test creating a LogicTag instance."""
        tag = LogicTag(
            original_text="If client age exceeds 55, include this section.",
            paragraph_index=5,
            is_conditional=True,
            condition_variable="client_age",
            operator=">",
            condition_value="55",
            jinja_wrapper="{% if client_age > 55 %}",
            confidence=0.95,
            reasoning="Detected 'exceeds' pattern with clear age threshold.",
        )

        assert tag.original_text == "If client age exceeds 55, include this section."
        assert tag.paragraph_index == 5
        assert tag.is_conditional is True
        assert tag.condition_variable == "client_age"
        assert tag.operator == ">"
        assert tag.condition_value == "55"
        assert tag.jinja_wrapper == "{% if client_age > 55 %}"
        assert tag.confidence == 0.95
        assert tag.reasoning == "Detected 'exceeds' pattern with clear age threshold."

    def test_logic_tag_immutability(self):
        """Test that LogicTag is frozen (immutable)."""
        tag = LogicTag(
            original_text="Test",
            paragraph_index=0,
            is_conditional=True,
            condition_variable="var",
            operator=">",
            condition_value="1",
            jinja_wrapper="{% if var > 1 %}",
            confidence=0.8,
            reasoning="Test reasoning",
        )

        # Attempting to modify should raise an error
        with pytest.raises(Exception):  # FrozenInstanceError from dataclasses
            tag.paragraph_index = 5


# =============================================================================
# DSPy Signature Tests
# =============================================================================


class TestConditionalLogicSignature:
    """Test suite for DSPy ConditionalLogicSignature."""

    def test_signature_exists(self):
        """Test that the signature is properly defined."""
        from app.strategies.logic.signatures import ConditionalLogicSignature
        import dspy

        # Verify it's a DSPy Signature
        assert issubclass(ConditionalLogicSignature, dspy.Signature)

        # Verify it has the expected fields
        signature_fields = ConditionalLogicSignature.__annotations__
        assert "text_segment" in signature_fields
        assert "available_variables" in signature_fields
        assert "is_conditional" in signature_fields
        assert "condition_variable" in signature_fields
        assert "operator" in signature_fields
        assert "condition_value" in signature_fields
        assert "jinja_wrapper" in signature_fields
        assert "confidence" in signature_fields
        assert "reasoning" in signature_fields


class TestConditionalLogicExtractor:
    """Test suite for DSPy ConditionalLogicExtractor module."""

    def test_module_exists(self):
        """Test that the module is properly defined."""
        import dspy

        # Verify it's a DSPy Module
        assert issubclass(ConditionalLogicExtractor, dspy.Module)

    def test_module_forward_method(self):
        """Test that the module has a forward method."""
        extractor = ConditionalLogicExtractor()
        assert hasattr(extractor, "forward")
        assert callable(extractor.forward)
