import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from agent_framework import Message
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from email_refine_agent.agent import agent as refine_agent  # type: ignore[import-not-found]
from email_refine_agent.agent import validate_refine_json  # type: ignore[import-not-found]
from logger import logger  # api/refine_api/logger.py

app = FastAPI(title="Email Refine Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("Email Refine Agent API starting up")


class RefineRequest(BaseModel):
    body: str = Field(..., description="The drafted email body to refine")
    instruction: str = Field(..., description="Refinement instruction, e.g. 'make it shorter'")


@app.get("/health")
async def health() -> dict[str, bool]:
    return {"ok": True}


@app.post("/v2/email/refine")
async def refine_email(req: RefineRequest) -> dict:
    """Refine a drafted email body per the user's instruction using the EmailRefineAgent."""
    body = req.body.strip()
    instruction = req.instruction.strip()

    if not body:
        logger.warning("refine_email: empty email body received")
        raise HTTPException(status_code=400, detail="body is required")
    if not instruction:
        logger.warning("refine_email: empty instruction received")
        raise HTTPException(status_code=400, detail="instruction is required")

    trace_id = f"tr_{uuid.uuid4().hex}"

    logger.info(
        "refine_email: request received | trace_id=%s | instruction=%r | body_chars=%d",
        trace_id, instruction, len(body),
    )

    session = refine_agent.create_session()
    started = datetime.now(timezone.utc)

    # Fence the untrusted email so the agent treats it as data, not instructions.
    user_message = (
        "[BEGIN ORIGINAL EMAIL]\n"
        f"{body}\n"
        "[END ORIGINAL EMAIL]\n\n"
        f"REFINEMENT INSTRUCTION: {instruction}"
    )

    try:
        logger.info("refine_email: calling EmailRefineAgent | trace_id=%s", trace_id)
        response = await refine_agent.run(
            [Message(role="user", contents=[user_message])],
            stream=False,
            session=session,
        )

        raw_text = response.text or ""
        valid_json, parsed = validate_refine_json(raw_text)
        latency_ms = int((datetime.now(timezone.utc) - started).total_seconds() * 1000)

        logger.info(
            "refine_email: agent response received | trace_id=%s | valid_json=%s | latency_ms=%d",
            trace_id, valid_json, latency_ms,
        )

        refined_obj = json.loads(parsed) if valid_json else None
        refined_body = (refined_obj or {}).get("body", "").strip()
        change_summary = (refined_obj or {}).get("change_summary", "").strip()
        error = (refined_obj or {}).get("error", "").strip()

        if error:
            logger.warning("refine_email: agent reported error | trace_id=%s | error=%s", trace_id, error)

        logger.info(
            "refine_email: done | trace_id=%s | latency_ms=%d | refined_chars=%d",
            trace_id, latency_ms, len(refined_body),
        )

        return {
            "trace_id": trace_id,
            "latency_ms": latency_ms,
            "valid_json": valid_json,
            "body": refined_body or (None if valid_json else raw_text.strip()),
            "change_summary": change_summary or None,
            "error": error or None,
            "raw_text": raw_text if not valid_json else None,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }

    except Exception as exc:
        logger.exception("refine_email: unhandled error | trace_id=%s", trace_id)
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import os

    import uvicorn

    reload = os.getenv("REFINE_API_RELOAD", "0") == "1"
    uvicorn.run("refine_api:app", host="0.0.0.0", port=8094, reload=reload)
