"""LLM layer: turn a prior email body into outreach insight.

Given the raw ``body`` of the previous email, produce two grounded fields:

  * ``why_now``         — one short line explaining why we are drafting a new email.
  * ``latest_activity`` — a small summary of the previous email.

Mirrors the Azure OpenAI client + JSON-contract pattern used in
``app/generate_ai_data/email_draft.py``.
"""

import os
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
from pydantic import BaseModel, Field

_ROOT = Path(__file__).resolve().parent
_PROJECT_ROOT = _ROOT.parent

load_dotenv(_PROJECT_ROOT / ".env", override=False)
load_dotenv(_PROJECT_ROOT / ".github" / ".env", override=False)  # fallback for CI/CD envs

_openai = AsyncAzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
)
_MODEL = os.getenv("AZURE_OPENAI_MODEL") or os.environ["AZURE_OPENAI_DEPLOYMENT"]


_EMAIL_INSIGHT_PROMPT = """\
You are a sales outreach assistant. You are given the body of the PREVIOUS \
email exchanged with a customer. Produce two grounded fields that explain the \
next outreach.

Return ONLY valid JSON with this exact schema:
{
  "classification": "outreach | action | document",
  "why_now": "one short single-line reason why we are drafting a new email now",
  "latest_activity": "a small summary of the previous email (1-2 sentences) if numerical value is present in the ingested data."
}

Rules:
- classification MUST be exactly one of: outreach, action, document.
  * outreach  — relationship / follow-up / check-in / nurturing emails.
  * action    — a task, request, or decision the seller must act on.
  * document  — sharing or requesting a document, proposal, quote, contract.
- why_now MUST be a single concise line.
- latest_activity MUST be a short summary of the previous email only.
- ONLY use facts explicitly present in the email body. Do NOT invent names, \
dates, prices, or commitments.
- If the body is empty or too thin to summarise, return: \
{"error": "Insufficient email content to generate insight."}
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_json_response(raw_text):
    clean = raw_text.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
        clean = clean.rsplit("```", 1)[0]
    return json.loads(clean)


async def _call_llm(system_prompt, user_text):
    completion = await _openai.chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
    )
    return completion.choices[0].message.content or ""


class EmailInsightRequest(BaseModel):
    body: str = Field(default="", description="Previous email body to summarise")


def _build_user_input(req: EmailInsightRequest) -> str:
    payload = {"email_body": req.body.strip()}
    return json.dumps(payload, ensure_ascii=False, indent=2)


async def generate_email_insight(req: EmailInsightRequest) -> dict:
    """Generate ``why_now`` + ``latest_activity`` from a prior email body."""
    user_input = _build_user_input(req)

    started = datetime.now(timezone.utc)
    raw_text = await _call_llm(_EMAIL_INSIGHT_PROMPT, user_input)
    latency_ms = int((datetime.now(timezone.utc) - started).total_seconds() * 1000)

    why_now = ""
    latest_activity = ""
    classification = ""
    error = None
    try:
        parsed = _parse_json_response(raw_text)
        if "error" in parsed:
            error = parsed["error"]
        else:
            classification = (parsed.get("classification") or "").strip()
            why_now = (parsed.get("why_now") or "").strip()
            latest_activity = (parsed.get("latest_activity") or "").strip()
    except (ValueError, AttributeError):
        error = "Failed to parse LLM response."

    return {
        "trace_id": f"tr_{uuid.uuid4().hex}",
        "latency_ms": latency_ms,
        "classification": classification,
        "why_now": why_now,
        "latest_activity": latest_activity,
        "error": error,
        "timestamp": _utc_now(),
    }
