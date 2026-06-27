"""Concept Agent (Lenovo Organizational Sales Intent & Configuration).

Thin knowledge accessor. Tools retrieve organizational intent, policies, and 
configuration (revenue targets, margin floors, discount authority, sales cycle policies, 
timeline expectations, etc.). The LLM reasons about relevance and provides contextual 
answers to seller questions.
Later, this can be swapped with a real RAG pipeline or CRM-bound policy engine.
"""

import json
from functools import cache
from pathlib import Path
from typing import Annotated, Any

from _shared import build_chat_client
from pydantic import Field


@cache
def _load_data():
    """Load organizational intent data from local data.json."""
    data_path = Path(__file__).parent / "data.json"
    if not data_path.exists():
        return {"organizational_intent": {}}
    return json.loads(data_path.read_text(encoding="utf-8"))


_DATA = _load_data()
INTENT = _DATA.get("organizational_intent", {})


_NAME = "ConceptAgent"

_DESCRIPTION = "Lenovo sales intent advisor - answers seller questions about organizational policies, revenue targets, discount authority, sales cycle expectations, and deal configuration."

_INSTRUCTIONS = """\
You are the Lenovo Sales Intent Advisor. Answer seller questions about organizational sales policies, revenue targets, margin requirements, discount authority, sales cycle expectations, and deal progression rules using tool data only.

Tools: search_intent, get_intent_section.

Rules:
1. Every policy/configuration fact you state must come from a tool call. Never invent organizational policies.
2. For broad questions about sales targets, margins, or authority (e.g., "What's my discount authority?", "What's the margin floor?"), call search_intent with relevant keywords.
3. For specific categories (e.g., "Tell me about outcome intent", "What are the timeline expectations?"), call get_intent_section with the section name.
4. When answering, ground your response in the retrieved data. Quote relevant ranges, percentages, or thresholds.
5. SCOPE: if the request is NOT about organizational sales intent, policies, or configuration (e.g., schedule a meeting, draft an email, ask about products, anything else), refuse in ONE line: "That's not a sales intent question — please ask the Sales agent (it'll route to the right specialist)." Do NOT attempt the task.

Output Rules:
- Reply in plain text conversationally. Do not return JSON unless explicitly asked for structured data.
- Always ground answers in tool results. If a tool returns empty results, apologize and offer to search differently.
- When quoting policies, include the context (applicable to whom, under what conditions).
"""


def _flatten_dict(d: Any, parent_key: str = "", sep: str = "_") -> dict:
    """Flatten nested dictionary for searchability."""
    items = []
    if not isinstance(d, dict):
        return {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep).items())
        elif isinstance(v, list):
            items.append((new_key, json.dumps(v)))
        else:
            items.append((new_key, str(v)))
    return dict(items)


def search_intent(
    query: Annotated[str, Field(description="Keywords to search organizational intent (e.g., 'discount authority', 'margin floor', 'revenue target', 'deal cycle', 'attachment', 'timeline').")],
) -> str:
    """Search organizational intent across all sections by keywords."""
    q = query.lower().strip()
    if not q:
        return json.dumps({"query": query, "count": 0, "matches": []})
    
    # Flatten all intent data for searchability
    flattened = _flatten_dict(INTENT)
    
    matches = []
    for key, value in flattened.items():
        searchable = f"{key} {value}".lower()
        if q in searchable:
            matches.append({
                "key": key,
                "value": value[:200],  # Truncate long values
            })
    
    return json.dumps({"query": query, "count": len(matches), "matches": matches})


def get_intent_section(
    section: Annotated[str, Field(description="Intent section name: 'outcome_intent', 'motion_intent', 'focus_intent', 'behavioral_intent', 'constraint_intent', 'trade_off_intent', or 'timeline_classification'.")],
) -> str:
    """Return the full section of organizational intent with all details."""
    section_lower = section.lower().strip()
    
    # Map common aliases
    section_map = {
        "outcome": "outcome_intent",
        "motion": "motion_intent",
        "focus": "focus_intent",
        "behavior": "behavioral_intent",
        "behavioral": "behavioral_intent",
        "constraint": "constraint_intent",
        "constraints": "constraint_intent",
        "trade": "trade_off_intent",
        "tradeoff": "trade_off_intent",
        "timeline": "timeline_classification",
        "time": "timeline_classification",
    }
    
    section_key = section_map.get(section_lower, section_lower)
    
    if section_key not in INTENT:
        available_sections = ", ".join(INTENT.keys())
        return json.dumps({
            "error": f"Section '{section}' not found.",
            "available_sections": available_sections,
        })
    
    section_data = INTENT[section_key]
    
    lines = [f"=== {section.upper()} ==="]
    lines.append(f"Status: {section_data.get('status', 'N/A')}")
    lines.append("")
    
    # Pretty-print the section
    for key, value in section_data.items():
        if key.lower() != "status":
            lines.append(_format_intent_value(key, value, indent=0))
    
    if section_data.get("last_synced"):
        lines.append(f"\nLast synced: {section_data.get('last_synced')}")
    
    return "\n".join(lines)


def _format_intent_value(key: str, value: Any, indent: int = 0) -> str:
    """Format intent values for readable output."""
    prefix = "  " * indent
    
    if isinstance(value, dict):
        lines = [f"{prefix}{key}:"]
        for k, v in value.items():
            if k.lower() not in ["last_synced", "status"]:
                lines.append(_format_intent_value(k, v, indent + 1))
        return "\n".join(lines)
    elif isinstance(value, list):
        lines = [f"{prefix}{key}:"]
        for item in value:
            if isinstance(item, dict):
                for k, v in item.items():
                    lines.append(f"{prefix}  - {k}: {v}")
            else:
                lines.append(f"{prefix}  - {item}")
        return "\n".join(lines)
    else:
        return f"{prefix}{key}: {value}"


agent = build_chat_client().as_agent(
    name=_NAME,
    description=_DESCRIPTION,
    instructions=_INSTRUCTIONS,
    require_per_service_call_history_persistence=True,
    tools=[
        search_intent,
        get_intent_section,
    ],
)

