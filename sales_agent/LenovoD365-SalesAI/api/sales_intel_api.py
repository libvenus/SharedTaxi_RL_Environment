import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
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

app = FastAPI(title="Sales Intel API", version="1.0.0")

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_MEETING_PREP_PROMPT = """\
You are a sales meeting preparation assistant. Given deal context (account info, \
recent signals, opportunity details, stakeholders, competitor intel), produce a \
structured JSON meeting prep brief.

Return ONLY valid JSON with this exact schema:

{
  "recent_signals": [
    {"signal": "...", "source": "email|calendar|crm|linkedin|document", "timestamp": "...", "relevance": "high|medium|low"}
  ],
  "prep_tasks": [
    {"task": "...", "priority": "high|medium|low"}
  ],
  "talking_points": [
    {"point": "...", "rationale": "..."}
  ],
  "watch_outs": [
    {"risk": "...", "mitigation": "..."}
  ]
}

Rules:
- 3-6 recent signals, ordered by recency
- 4-6 prep tasks, ordered by priority
- 4-6 talking points, ordered for conversation flow
- 2-4 watch-outs with mitigations
- Be specific to the deal data provided — no generic filler
- Keep each item concise (1 sentence max)
- ONLY use facts explicitly present in the input. Do NOT invent, assume, or fabricate any data.
- If the input lacks sufficient deal context, return: {"error": "Insufficient input data to generate meeting prep."}
"""

_ACCOUNT_SUMMARY_PROMPT = """\
You are a sales intelligence assistant. Given raw CRM data about an account, \
write a concise paragraph (3-5 sentences) covering: relationship tier, size, \
key contacts, revenue potential, executive relationships, prior purchases.

Return ONLY valid JSON: {"text": "your paragraph here"}

Rules:
- ONLY use facts explicitly stated in the input. Do NOT invent or assume any data.
- If the input lacks real account data, return: {"error": "Insufficient input data to generate account summary."}
"""

_DEAL_SUMMARY_PROMPT = """\
You are a sales intelligence assistant. Given raw CRM/deal data, write a concise \
paragraph (3-5 sentences) covering: product, quantity, stage, budget status, \
timeline, pricing position, DaaS interest, competitor landscape.

Return ONLY valid JSON: {"text": "your paragraph here"}

Rules:
- ONLY use facts explicitly stated in the input. Do NOT invent or assume any data.
- If the input lacks real deal data, return: {"error": "Insufficient input data to generate deal summary."}
"""

# ---------------------------------------------------------------------------
# Request models (drives Swagger UI input fields)
# ---------------------------------------------------------------------------


class MeetingPrepRequest(BaseModel):
    deal_context: str = Field(..., description="Deal context: account, opportunity, signals, stakeholders, competitor intel")
    meeting_id: str = Field(default="default", description="Meeting identifier")


class AccountSummaryRequest(BaseModel):
    context: str = Field(..., description="Raw CRM/account data to summarize")
    account_id: str = Field(default="default", description="Account identifier")


class DealSummaryRequest(BaseModel):
    context: str = Field(..., description="Raw CRM/deal data to summarize")
    deal_id: str = Field(default="default", description="Deal identifier")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_json_response(raw_text):
    """Strip markdown fences and parse JSON."""
    clean = raw_text.strip()
    if clean.startswith("```"):  # LLMs sometimes wrap JSON in code fences despite instructions
        clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
        clean = clean.rsplit("```", 1)[0]
    return json.loads(clean)


async def _call_llm(system_prompt, user_text):
    """Single LLM call, returns raw content string."""
    completion = await _openai.chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
    )
    return completion.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/v2/meeting-prep")
async def meeting_prep(req: MeetingPrepRequest):
    deal_context = req.deal_context.strip()
    if not deal_context:
        raise HTTPException(status_code=400, detail="deal_context is required")

    started = datetime.now(timezone.utc)
    raw_text = await _call_llm(_MEETING_PREP_PROMPT, deal_context)
    latency_ms = int((datetime.now(timezone.utc) - started).total_seconds() * 1000)

    try:
        parsed = _parse_json_response(raw_text)
    except (json.JSONDecodeError, ValueError):
        parsed = None

    return {
        "trace_id": f"tr_{uuid.uuid4().hex}",
        "meeting_id": req.meeting_id,
        "latency_ms": latency_ms,
        "valid_json": parsed is not None,
        "prep": parsed,
        "raw_text": raw_text if parsed is None else None,  # only return raw when JSON parse failed
    }


@app.post("/v2/account-summary")
async def account_summary(req: AccountSummaryRequest):
    context = req.context.strip()
    if not context:
        raise HTTPException(status_code=400, detail="context is required")

    started = datetime.now(timezone.utc)
    raw_text = await _call_llm(_ACCOUNT_SUMMARY_PROMPT, context)
    latency_ms = int((datetime.now(timezone.utc) - started).total_seconds() * 1000)

    try:
        parsed = _parse_json_response(raw_text)
        text = parsed.get("text", raw_text.strip())
    except (json.JSONDecodeError, ValueError):
        text = raw_text.strip()

    return {
        "trace_id": f"tr_{uuid.uuid4().hex}",
        "account_id": req.account_id,
        "latency_ms": latency_ms,
        "text": text,
    }


@app.post("/v2/deal-summary")
async def deal_summary(req: DealSummaryRequest):
    context = req.context.strip()
    if not context:
        raise HTTPException(status_code=400, detail="context is required")

    started = datetime.now(timezone.utc)
    raw_text = await _call_llm(_DEAL_SUMMARY_PROMPT, context)
    latency_ms = int((datetime.now(timezone.utc) - started).total_seconds() * 1000)

    try:
        parsed = _parse_json_response(raw_text)
        text = parsed.get("text", raw_text.strip())
    except (json.JSONDecodeError, ValueError):
        text = raw_text.strip()

    return {
        "trace_id": f"tr_{uuid.uuid4().hex}",
        "deal_id": req.deal_id,
        "latency_ms": latency_ms,
        "text": text,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("sales_intel_api:app", host="127.0.0.1", port=8090, reload=False)
