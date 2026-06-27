import os
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import AsyncAzureOpenAI
from pydantic import BaseModel, Field

_ROOT = Path(__file__).resolve().parent
_PROJECT_ROOT = _ROOT.parent

# Make api/ importable so logger.py can be found
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

load_dotenv(_PROJECT_ROOT / ".env", override=False)
load_dotenv(_PROJECT_ROOT / ".github" / ".env", override=False)  # fallback for CI/CD envs

from logger import logger  # api/logger.py

_openai = AsyncAzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
)
_MODEL = os.getenv("AZURE_OPENAI_MODEL") or os.environ["AZURE_OPENAI_DEPLOYMENT"]

app = FastAPI(title="Email API", version="1.0.0")

logger.info("Email API starting up | model=%s", _MODEL)


_EMAIL_DRAFT_PROMPT = """\
You are a professional sales email writer. Given context about a recipient, \
deal, or situation, draft a clear, concise, professional email body.

You receive a single JSON object as input with these fields:
- "context": free-text background describing the recipient, deal, or situation.
- "written_context": extra instructions written by the user. Treat as highest priority.
- "data": a structured object of facts (names, deal, dates, amounts, etc.).
    Use these values as the source of truth for any concrete details.
- "template": an optional email body template. If present, follow its
    structure, tone, and wording.
- "placeholders": an optional map of placeholder names to values used to fill
    the template (example: {"deal_name": "Acme Refresh"}).

Return ONLY valid JSON: {"body": "your email body here"}

Rules:
- Write only the email body — no subject line, no "To:"/"From:" headers.
- Keep it professional, warm, and concise (3-6 short paragraphs max).
- Use a greeting and a sign-off.
- Treat written_context as highest-priority user instructions.
- If an email template is provided, follow its structure and tone.
- When resolving placeholders, use the "placeholders" map first, then fall back
    to "data", then to "context".
- Keep unresolved placeholders untouched (example: {{deal_name}}) rather than
    inventing values.
- ONLY use facts explicitly stated in context, written_context, or data. Do NOT \
invent names, dates, prices, or commitments not present in the input.
- If the context is too thin to write a meaningful email, return: \
{"error": "Insufficient context to draft an email."}

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


class EmailDraftRequest(BaseModel):
    context: str = Field(default="", description="Free-text context to draft from")
    written_context: str = Field(default="", description="Additional user-written context from frontend")
    data: dict = Field(default_factory=dict, description="Structured frontend payload for facts and values")
    template: str | None = Field(default=None, description="Optional email body template with placeholders like {{recipient_name}}")
    placeholders: dict = Field(default_factory=dict, description="Optional placeholder map from frontend")


def _build_user_input(req: EmailDraftRequest) -> str:
    context = req.context.strip()
    written_context = req.written_context.strip()
    template = (req.template or "").strip()
    payload = {
        "context": context,
        "written_context": written_context,
        "data": req.data or {},
        "template": template,
        "placeholders": req.placeholders or {},
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


@app.get("/health")
async def health() -> dict[str, bool]:
    return {"ok": True}


@app.post("/v2/email/draft")
async def draft_email(req: EmailDraftRequest) -> dict:
    """Draft an email body from context using Azure OpenAI."""
    if not req.context.strip() and not req.written_context.strip() and not req.template and not req.data:
        logger.warning("draft_email: request rejected — no usable input provided")
        raise HTTPException(status_code=400, detail="Provide at least one of: context, written_context, template, or data")

    trace_id = f"tr_{uuid.uuid4().hex}"
    logger.info(
        "draft_email: request received | trace_id=%s | has_context=%s | has_written_context=%s | has_template=%s | has_data=%s",
        trace_id,
        bool(req.context.strip()),
        bool(req.written_context.strip()),
        bool(req.template),
        bool(req.data),
    )

    user_input = _build_user_input(req)

    started = datetime.now(timezone.utc)
    try:
        logger.info("draft_email: calling Azure OpenAI | trace_id=%s | model=%s", trace_id, _MODEL)
        raw_text = await _call_llm(_EMAIL_DRAFT_PROMPT, user_input)
    except Exception:
        logger.exception("draft_email: LLM call failed | trace_id=%s", trace_id)
        raise HTTPException(status_code=502, detail="LLM call failed")

    latency_ms = int((datetime.now(timezone.utc) - started).total_seconds() * 1000)
    logger.info("draft_email: LLM response received | trace_id=%s | latency_ms=%d | chars=%d", trace_id, latency_ms, len(raw_text))

    try:
        parsed = _parse_json_response(raw_text)
        body = parsed.get("body", raw_text.strip())
        if "error" in parsed:
            logger.warning("draft_email: LLM returned error | trace_id=%s | error=%s", trace_id, parsed["error"])
    except (ValueError, AttributeError):
        logger.warning("draft_email: JSON parse failed, using raw text | trace_id=%s", trace_id)
        body = raw_text.strip()

    email_id = f"email_{uuid.uuid4().hex[:8]}"
    logger.info("draft_email: done | trace_id=%s | email_id=%s | latency_ms=%d", trace_id, email_id, latency_ms)

    return {
        "trace_id": trace_id,
        "email_id": email_id,
        "latency_ms": latency_ms,
        "body": body,
        "timestamp": _utc_now(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("email_api:app", host="0.0.0.0", port=8093, reload=False)
