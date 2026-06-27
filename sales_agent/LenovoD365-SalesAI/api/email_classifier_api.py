import json
import logging
import os
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

load_dotenv(_PROJECT_ROOT / ".env", override=False)
load_dotenv(_PROJECT_ROOT / ".github" / ".env", override=False)  # fallback for CI/CD envs

_openai = AsyncAzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
)
_MODEL = os.getenv("AZURE_OPENAI_MODEL") or os.environ["AZURE_OPENAI_DEPLOYMENT"]

app = FastAPI(title="Email Classifier API", version="1.0.0")


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("email_classifier_api")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)
    file_handler = logging.FileHandler(_PROJECT_ROOT / "email_classifier_api.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


logger = _setup_logger()
logger.info("Email Classifier API starting up | model=%s", _MODEL)


_CLASSIFY_PROMPT = """\
You are an email classification engine for a sales outreach assistant. Given \
data about an email the user is about to write (context, prior thread, recipient \
and deal facts, or an existing draft), classify it so the system can pick the \
right tone and template before drafting.

You receive a single JSON object as input with these fields:
- "context": free-text background about the recipient, deal, or situation.
- "email_body": an existing draft or prior email text, if any.
- "data": a structured object of facts (account, deal, stage, contact, etc.).

Return ONLY valid JSON with this exact schema:
{
  "category": "meeting_follow_up | introduction | proposal | negotiation | scheduling | check_in | thank_you | reminder | escalation | closing | other",
  "intent": "one short phrase describing the purpose of the email"
}

Rules:
- Choose exactly one value for category.
- ONLY use facts explicitly present in the input. Do NOT invent details.
- intent must be a single concise phrase grounded in the input.
- If there is not enough information to classify, return: \
{"error": "Insufficient input to classify the email."}
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


class EmailClassifyRequest(BaseModel):
    context: str = Field(default="", description="Free-text context about the recipient, deal, or situation")
    email_body: str = Field(default="", description="Existing draft or prior email text, if any")
    data: dict = Field(default_factory=dict, description="Structured frontend payload of facts and values")


def _build_user_input(req: EmailClassifyRequest) -> str:
    payload = {
        "context": req.context.strip(),
        "email_body": req.email_body.strip(),
        "data": req.data or {},
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


@app.get("/health")
async def health() -> dict[str, bool]:
    return {"ok": True}


@app.post("/v2/email/classify")
async def classify_email(req: EmailClassifyRequest) -> dict:
    """Classify email data so the draft step can pick the right tone and template."""
    if not req.context.strip() and not req.email_body.strip() and not req.data:
        logger.warning("classify_email: request rejected — no usable input provided")
        raise HTTPException(status_code=400, detail="Provide at least one of: context, email_body, or data")

    trace_id = f"tr_{uuid.uuid4().hex}"
    logger.info(
        "classify_email: request received | trace_id=%s | has_context=%s | has_body=%s | has_data=%s",
        trace_id,
        bool(req.context.strip()),
        bool(req.email_body.strip()),
        bool(req.data),
    )

    user_input = _build_user_input(req)

    started = datetime.now(timezone.utc)
    try:
        logger.info("classify_email: calling Azure OpenAI | trace_id=%s | model=%s", trace_id, _MODEL)
        raw_text = await _call_llm(_CLASSIFY_PROMPT, user_input)
    except Exception:
        logger.exception("classify_email: LLM call failed | trace_id=%s", trace_id)
        raise HTTPException(status_code=502, detail="LLM call failed")

    latency_ms = int((datetime.now(timezone.utc) - started).total_seconds() * 1000)
    logger.info("classify_email: LLM response received | trace_id=%s | latency_ms=%d | chars=%d", trace_id, latency_ms, len(raw_text))

    try:
        parsed = _parse_json_response(raw_text)
        valid_json = True
        if "error" in parsed:
            logger.warning("classify_email: LLM returned error | trace_id=%s | error=%s", trace_id, parsed["error"])
    except (ValueError, AttributeError):
        logger.warning("classify_email: JSON parse failed, returning raw text | trace_id=%s", trace_id)
        parsed = None
        valid_json = False

    logger.info("classify_email: done | trace_id=%s | latency_ms=%d | valid_json=%s", trace_id, latency_ms, valid_json)

    return {
        "trace_id": trace_id,
        "latency_ms": latency_ms,
        "valid_json": valid_json,
        "classification": parsed,
        "raw_text": raw_text if not valid_json else None,
        "timestamp": _utc_now(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("email_classifier_api:app", host="0.0.0.0", port=8094, reload=False)
