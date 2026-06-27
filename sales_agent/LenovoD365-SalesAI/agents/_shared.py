"""Shared helpers for the DevUI discovery shims.

DevUI scans `agents/` and registers every direct subdirectory as an entity.
Top-level files starting with `_` are skipped, so this module is safe to
import from sibling agents without polluting the entity list.

`load_data()` returns the project-wide CRM stub data (accounts, opportunities,
interactions, email threads, meeting transcripts, seller activities) from
`agents/crm_data.json`. Each agent's name/description/instructions are
defined inline in its own `agent.py` file.
"""

import json
import os
from functools import cache
from pathlib import Path

from agent_framework.openai import OpenAIChatClient
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = Path(__file__).resolve().parent / "crm_data.json"


@cache
def load_data():
    """Return the parsed `agents/crm_data.json`. Cached - read disk once per process."""
    with DATA_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def build_chat_client():
    """Build an OpenAIChatClient from `.env` or `.github/.env` credentials.

    Prefers Azure OpenAI (`AZURE_OPENAI_*`); falls back to public OpenAI.
    """
    load_dotenv(ROOT / ".env", override=False)
    load_dotenv(ROOT / ".github" / ".env", override=False)

    azure_endpoint = (
        os.getenv("AZURE_OPENAI_ENDPOINT")
        or os.getenv("AZURE_COGNITIVE_SERVICES_ENDPOINT")
        or ""
    ).strip()
    azure_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    azure_model = (
        os.getenv("AZURE_OPENAI_MODEL")
        or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        or ""
    ).strip()

    if azure_endpoint and azure_key and azure_model:
        return OpenAIChatClient(
            model=azure_model,
            azure_endpoint=azure_endpoint,
            api_key=azure_key,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION") or None,
        )

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key:
        return OpenAIChatClient(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip(),
            api_key=openai_key,
        )

    raise RuntimeError(
        "Missing model credentials. Set AZURE_OPENAI_ENDPOINT + "
        "AZURE_OPENAI_API_KEY + AZURE_OPENAI_MODEL, or OPENAI_API_KEY."
    )
