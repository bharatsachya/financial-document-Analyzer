"""Intelligence Engine - Report Generation with Multi-Tier Memory.

This module orchestrates the retrieval from the multi-tier memory system
and generates final reports using LLM inference.

Memory Tiers:
- Factual Data (Neo4j): Client fact find data
- Procedural Memory (PostgreSQL): Learned logic and corrections
- Semantic Memory (Qdrant): Tone and style preferences
- Episodic Memory (PostgreSQL): Event logging and audit trail
"""

import datetime
import json
import logging
import re
import uuid
from typing import Any

from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import Settings, get_settings
from app.db.database import EventType
from app.db.graph import GraphRepository
from app.db.session import get_session_maker
from app.services.episodic_memory import EpisodicMemoryManager, get_episodic_memory_manager
from app.services.procedural_memory import (
    ProceduralMemoryManager,
    get_procedural_memory_manager,
)
from app.services.semantic_memory import (
    SemanticMemoryManager,
    get_semantic_memory_manager,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Prompt Templates
# =============================================================================

_SYSTEM_PROMPT_TEMPLATE = """You are an expert UK financial advisory report writer.

Your task is to generate a professional, compliant financial report based on
the provided template and factual client data.

{memory_sections}

CRITICAL INSTRUCTIONS:
1. Use only the factual data provided in the CLIENT FACTS section.
2. Apply all PROCEDURAL MAPPING RULES strictly - these are learned corrections.
3. Follow all SEMANTIC STYLE RULES - these reflect the adviser's preferences.
4. Maintain FCA compliance throughout the report.
5. Be accurate and professional at all times.
6. If factual data is missing or unavailable, indicate this clearly using [DATA NOT AVAILABLE].

OUTPUT FORMAT:
Generate the complete report text, filling in the template placeholders
with appropriate content derived from the factual data.
"""


_CLIENT_FACTS_TEMPLATE = """
### CLIENT FACTS

The following factual data has been retrieved from the client record:

```json
{client_facts}
```

Use this data as the primary source of truth for the report content.
"""


_PROCEDURAL_RULES_TEMPLATE = """
### PROCEDURAL MAPPING RULES (CRITICAL)

Apply these learned correction rules when processing variables:

{procedural_rules}

These rules have been learned from the adviser's previous corrections.
Apply them without exception to ensure consistency with the adviser's
established patterns.
"""


_SEMANTIC_RULES_TEMPLATE = """
### SEMANTIC STYLE RULES

Apply these stylistic preferences learned from the adviser's feedback:

{semantic_rules}

These rules reflect the tone, voice, and writing style preferences
of the adviser. Incorporate them naturally throughout the report.
"""


# =============================================================================
# Variable Extraction
# =============================================================================


def extract_template_variables(template_text: str) -> dict[str, Any]:
    """Extract variable placeholders from template text.

    Supports both {{ variable }} and {variable} patterns.

    Args:
        template_text: The template text to parse.

    Returns:
        Dict mapping variable names to None (for filling).
    """
    # Match {{ variable }} style (Jinja2)
    jinja2_pattern = r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}"
    jinja2_vars = re.findall(jinja2_pattern, template_text)

    # Match {variable} style (Python format)
    python_pattern = r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}"
    python_vars = re.findall(python_pattern, template_text)

    # Combine and deduplicate
    all_vars = set(jinja2_vars + python_vars)

    return {var: None for var in all_vars}


def map_facts_to_variables(
    client_facts: dict[str, Any],
    template_variables: dict[str, Any],
) -> dict[str, Any]:
    """Map client facts to template variables.

    Performs intelligent mapping between client data structure
    and template variable names.

    Args:
        client_facts: Factual data from GraphRepository.
        template_variables: Variables extracted from template.

    Returns:
        Dict mapping variable names to their values.
    """
    mapped = {}

    # Direct name matches
    for var_name in template_variables.keys():
        if var_name in client_facts:
            mapped[var_name] = client_facts[var_name]
            continue

        # Try nested access
        if var_name.startswith("client_"):
            # Try client.<field>
            field = var_name[7:]  # Remove "client_" prefix
            if f"client_{field}" in client_facts:
                mapped[var_name] = client_facts[f"client_{field}"]
            elif "client_name" in client_facts and field == "name":
                mapped[var_name] = client_facts["client_name"]

    # Risk profile mapping
    if "risk_profile" in client_facts and client_facts["risk_profile"]:
        rp = client_facts["risk_profile"]
        for var_name in template_variables.keys():
            if var_name in ["risk_tolerance", "risk_level", "investment_horizon", "profile_type"]:
                if var_name in rp:
                    mapped[var_name] = rp[var_name]

    # Goals mapping
    if "goals" in client_facts and client_facts["goals"]:
        goals = client_facts["goals"]
        if "goals" in template_variables:
            mapped["goals"] = goals
        if "primary_goal" in template_variables and goals:
            # Get highest priority goal
            primary = max(goals, key=lambda g: g.get("priority", 0), default=None)
            if primary:
                mapped["primary_goal"] = primary

    # Income mapping
    if "income_sources" in client_facts and client_facts["income_sources"]:
        incomes = client_facts["income_sources"]
        if "income" in template_variables:
            mapped["income"] = incomes
        if "total_income" in template_variables:
            mapped["total_income"] = sum(
                inc.get("amount", 0) for inc in incomes
            )

    # Assets mapping
    if "accounts" in client_facts and client_facts["accounts"]:
        accounts = client_facts["accounts"]
        if "accounts" in template_variables:
            mapped["accounts"] = accounts
        if "total_assets" in template_variables:
            mapped["total_assets"] = sum(
                acc.get("balance", 0) for acc in accounts
            )

    return mapped


# =============================================================================
# LLM Inference with Retry Logic
# =============================================================================


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((TimeoutError, ConnectionError)),
)
async def call_llm_with_retry(
    client: AsyncOpenAI,
    system_prompt: str,
    user_message: str,
    model: str = "openai/gpt-4o-mini",
) -> str:
    """Call LLM with exponential backoff retry logic.

    Args:
        client: AsyncOpenAI client.
        system_prompt: System prompt for the LLM.
        user_message: User message containing the template and context.
        model: Model to use for generation.

    Returns:
        The generated text.

    Raises:
        Exception: If all retries are exhausted.
    """
    logger.info(f"Calling LLM (model={model})")

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.7,
        max_tokens=4000,
    )

    content = response.choices[0].message.content or ""
    logger.info(f"LLM generated {len(content)} characters")

    return content


# =============================================================================
# Main Generator Function
# =============================================================================


async def generate_report(
    adviser_id: str,
    client_id: str,
    topic: str,
    template_text: str,
    org_id: uuid.UUID | None = None,
) -> dict[str, Any]:
    """Generate a report using the Intelligence Engine.

    Orchestrates retrieval from multi-tier memory and generates
    the final report with LLM inference.

    Args:
        adviser_id: The adviser generating the report.
        client_id: The client to generate the report for.
        topic: The topic/subject of the report.
        template_text: The template text to fill in.
        org_id: Optional organization ID.

    Returns:
        Dict containing:
            - report_id: UUID of the report
            - generated_text: The generated report content
            - extracted_variables: Variables extracted from template
            - sources: Source information for each memory tier
            - session_id: Session UUID for tracking
            - timestamp: Generation timestamp

    Raises:
        Exception: If generation fails.
    """
    settings = get_settings()
    session_id = uuid.uuid4()
    sources: dict[str, list[str]] = {
        "factual": [],
        "procedural": [],
        "semantic": [],
    }

    logger.info(
        f"Generating report for adviser {adviser_id}, "
        f"client {client_id}, session {session_id}"
    )

    # ── Step 1: Fetch Factual Data from GraphRepository ────────────────────────
    logger.info("Fetching factual data from GraphRepository...")
    graph_repo = GraphRepository(settings=settings)
    client_facts = await graph_repo.get_client_fact_find(client_id)

    if client_facts.get("_fallback"):
        logger.warning(f"Using fallback data for client {client_id}")
        sources["factual"].append("fallback_data")
    else:
        sources["factual"].append(f"neo4j_client_{client_id}")

    # ── Step 2: Extract Template Variables ────────────────────────────────────
    logger.info("Extracting template variables...")
    template_variables = extract_template_variables(template_text)
    logger.info(f"Extracted {len(template_variables)} template variables")

    # ── Step 3: Map Facts to Variables ────────────────────────────────────────
    logger.info("Mapping client facts to template variables...")
    variable_values = map_facts_to_variables(client_facts, template_variables)

    # ── Step 4: Fetch Procedural Memory (Logic) from PostgreSQL ───────────────
    logger.info("Fetching procedural memory...")
    session_maker = get_session_maker(settings)
    async with session_maker() as session:
        procedural_manager = get_procedural_memory_manager()
        procedural_rules = await procedural_manager.get_adviser_rules(
            session=session,
            adviser_id=adviser_id,
            org_id=org_id,
        )

        # Apply corrections if applicable
        if template_variables:
            corrected_values = await procedural_manager.apply_corrections(
                session=session,
                adviser_id=adviser_id,
                variables=variable_values,
            )
            variable_values.update(corrected_values)

        sources["procedural"] = [f"adviser_{adviser_id}_rules"]

    # ── Step 5: Fetch Semantic Memory (Tone) using Qdrant ──────────────────────
    logger.info("Fetching semantic memory...")
    semantic_manager = get_semantic_memory_manager()

    # Get embedding for the topic to search semantic preferences
    embedding_client = AsyncOpenAI(
        api_key=settings.openrouter_api_key,
        base_url=settings.openai_base_url,
    )

    try:
        embed_resp = await embedding_client.embeddings.create(
            model=settings.llm_embedding_model,
            input=topic,
            encoding_format="float",
        )
        topic_embedding = embed_resp.data[0].embedding

        semantic_rules = semantic_manager.search_preferences(
            adviser_id=adviser_id,
            query_embedding=topic_embedding,
            top_k=5,
        )
        sources["semantic"] = [f"qdrant_{len(semantic_rules)}_rules"]

    except Exception as e:
        logger.error(f"Error fetching semantic preferences: {e}", exc_info=True)
        semantic_rules = []
        sources["semantic"] = ["fetch_error"]

    # ── Step 6: Construct System Prompt with Memory Sections ──────────────────
    logger.info("Constructing system prompt...")

    # Build client facts section
    client_facts_section = _CLIENT_FACTS_TEMPLATE.format(
        client_facts=_format_json_for_prompt(client_facts)
    )

    # Build procedural rules section
    if procedural_rules:
        procedural_lines = [
            f"- {rule['key']}: {rule.get('value', {}).get('description', '')}"
            f" (confidence: {rule['confidence']:.2f})"
            for rule in procedural_rules
        ]
        procedural_section = _PROCEDURAL_RULES_TEMPLATE.format(
            procedural_rules="\n".join(procedural_lines)
        )
    else:
        procedural_section = (
            "\n### PROCEDURAL MAPPING RULES (CRITICAL)\n"
            "No procedural rules found for this adviser.\n"
        )

    # Build semantic rules section
    if semantic_rules:
        semantic_lines = [
            f"- {rule['rule_text']}"
            for rule in semantic_rules
        ]
        semantic_section = _SEMANTIC_RULES_TEMPLATE.format(
            semantic_rules="\n".join(semantic_lines)
        )
    else:
        semantic_section = (
            "\n### SEMANTIC STYLE RULES\n"
            "No semantic style rules found for this adviser. "
            "Use standard professional tone.\n"
        )

    # Combine all memory sections
    memory_sections = (
        client_facts_section + "\n" + procedural_section + "\n" + semantic_section
    )

    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(memory_sections=memory_sections)

    # Build user message with template and variable context
    user_message = f"""Generate a {topic} report based on the following template.

TEMPLATE:
{template_text}

AVAILABLE VARIABLES:
{json.dumps(variable_values, indent=2, default=str)}

Please fill in the template placeholders with appropriate content derived
from the client facts, following all the procedural and semantic rules."""

    # ── Step 7: LLM Inference with Retry Logic ───────────────────────────────
    logger.info("Calling LLM with retry logic...")

    llm_client = AsyncOpenAI(
        api_key=settings.openrouter_api_key,
        base_url=settings.openai_base_url,
    )

    generated_text = await call_llm_with_retry(
        client=llm_client,
        system_prompt=system_prompt,
        user_message=user_message,
        model="openai/gpt-4o-mini",
    )

    # ── Step 8: Log Successful Generation to EpisodicMemory ──────────────────
    logger.info("Logging report generation to episodic memory...")
    session_maker = get_session_maker(settings)
    async with session_maker() as session:
        episodic_manager = get_episodic_memory_manager()

        try:
            await episodic_manager.log_report_generation(
                session=session,
                adviser_id=adviser_id,
                org_id=org_id if org_id else uuid.uuid4(),
                client_id=client_id,
                topic=topic,
                generated_text=generated_text,
                extracted_variables=variable_values,
                sources=sources,
                session_id=session_id,
            )
        except Exception as e:
            logger.error(f"Error logging to episodic memory: {e}", exc_info=True)
            # Don't fail the whole generation if logging fails

    # ── Step 9: Return Results ────────────────────────────────────────────────
    result = {
        "report_id": uuid.uuid4(),
        "generated_text": generated_text,
        "extracted_variables": variable_values,
        "sources": sources,
        "session_id": session_id,
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }

    logger.info(
        f"Report generation completed for session {session_id} "
        f"({len(generated_text)} chars)"
    )

    return result


# =============================================================================
# Utilities
# =============================================================================


def _format_json_for_prompt(data: Any) -> str:
    """Format JSON data for inclusion in prompt.

    Truncates long values and structures for better LLM consumption.

    Args:
        data: Data to format.

    Returns:
        Formatted JSON string.
    """
    import json

    # Convert to JSON with string representation of UUIDs
    def json_serializer(obj):
        if isinstance(obj, uuid.UUID):
            return str(obj)
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
        raise TypeError(f"Type {type(obj)} not serializable")

    json_str = json.dumps(data, default=json_serializer, indent=2)

    # Truncate if too long
    if len(json_str) > 10000:
        logger.warning("Client facts JSON truncated for prompt")
        return json_str[:10000] + "\n... (truncated)"

    return json_str
