"""Intelligence Engine Services.

This package provides services for the multi-tier memory architecture:

- semantic_memory.py: Qdrant-based semantic memory for tone/style
- procedural_memory.py: PostgreSQL-based procedural memory for logic/corrections
- episodic_memory.py: PostgreSQL-based episodic memory for events
- generator.py: Report generation orchestration with multi-tier memory
"""

from app.services.episodic_memory import EpisodicMemoryManager, get_episodic_memory_manager
from app.services.generator import (
    call_llm_with_retry,
    extract_template_variables,
    generate_report,
    map_facts_to_variables,
)
from app.services.procedural_memory import (
    ProceduralMemoryManager,
    get_procedural_memory_manager,
)
from app.services.semantic_memory import (
    SemanticMemoryManager,
    get_semantic_memory_manager,
)

__all__ = [
    "EpisodicMemoryManager",
    "get_episodic_memory_manager",
    "ProceduralMemoryManager",
    "get_procedural_memory_manager",
    "SemanticMemoryManager",
    "get_semantic_memory_manager",
    "call_llm_with_retry",
    "extract_template_variables",
    "generate_report",
    "map_facts_to_variables",
]
