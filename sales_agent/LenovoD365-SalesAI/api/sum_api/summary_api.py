import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from agent_framework import Message
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import requests


_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
# _shared.py lives in c:\work\bot\agents — needed by summary_agent.agent
_AGENTS_DIR = Path(__file__).resolve().parents[2] / "agents"
if str(_AGENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENTS_DIR))

from summary_agent.agent import agent as summary_agent  # type: ignore[import-not-found]
from summary_agent.agent import validate_summary_json  # type: ignore[import-not-found]
from logger import logger  # api/sum_api/logger.py

app = FastAPI(title="Summary Agent API", version="1.0.0")

logger.info("Summary Agent API starting up")

# Reuse session per meeting_id so multi-turn follow-ups keep context
_SESSIONS = {}


class SummaryRequest(BaseModel):
    transcript: str = Field(..., description="Raw meeting transcript text")
    meeting_id: str = Field(..., description="Unique meeting identifier")


@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/v2/summary")
async def summarize(req: SummaryRequest):
    transcript = req.transcript.strip()
    if not transcript:
        logger.warning("summarize: empty transcript received | meeting_id=%s", req.meeting_id)
        raise HTTPException(status_code=400, detail="transcript is required")

    logger.info(
        "summarize: request received | meeting_id=%s | transcript_chars=%d",
        req.meeting_id, len(transcript),
    )

    session = _SESSIONS.get(req.meeting_id)
    if session is None:
        session = summary_agent.create_session()
        _SESSIONS[req.meeting_id] = session
        logger.info("summarize: new session created | meeting_id=%s", req.meeting_id)
    else:
        logger.info("summarize: reusing existing session | meeting_id=%s", req.meeting_id)

    trace_id = f"tr_{uuid.uuid4().hex}"
    started = datetime.now(timezone.utc)

    # Fence the untrusted transcript so the agent treats it as data, not instructions.
    fenced_transcript = (
        "[BEGIN UNTRUSTED TRANSCRIPT]\n"
        f"{transcript}\n"
        "[END UNTRUSTED TRANSCRIPT]"
    )

    try:
        logger.info("summarize: calling SummaryAgent | trace_id=%s | meeting_id=%s", trace_id, req.meeting_id)
        response = await summary_agent.run(
            [Message(role="user", contents=[fenced_transcript])],
            stream=False,
            session=session,
        )

        raw_text = response.text or ""
        valid_json, parsed = validate_summary_json(raw_text)
        latency_ms = int((datetime.now(timezone.utc) - started).total_seconds() * 1000)

        logger.info(
            "summarize: agent response received | trace_id=%s | meeting_id=%s | valid_json=%s | latency_ms=%d",
            trace_id, req.meeting_id, valid_json, latency_ms,
        )

        # Forward the summary to the downstream API which is Nemisha api
        summary_obj = json.loads(parsed) if valid_json else None
        downstream_status = None
        if valid_json and summary_obj is not None:
            # Use the incoming meeting_id if it's already a valid UUID,
            # otherwise derive a stable UUID from it.
            try:
                meeting_uuid = str(uuid.UUID(req.meeting_id))
            except ValueError:
                meeting_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, req.meeting_id))

            # Downstream schema: { payload: { meeting_id, transcript, meetingHeader, meetingSummary } }.
            # Flatten the summary structure - no "summary" wrapper
            downstream_payload = {
                "payload": {
                    "meeting_id": meeting_uuid,
                    "transcript": transcript,
                    "meetingHeader": summary_obj.get("meetingHeader", {}),
                    "meetingSummary": summary_obj.get("meetingSummary", {}),
                },
            }
            try:
                logger.info("summarize: posting to downstream API | meeting_id=%s | uuid=%s", req.meeting_id, meeting_uuid)
                ds_resp = requests.post(
                    "http://127.0.0.1:9101/api/summary-details/",
                    json=downstream_payload,
                    timeout=30,
                )
                downstream_status = ds_resp.status_code
                logger.info(
                    "summarize: downstream POST complete | meeting_id=%s | status=%s",
                    req.meeting_id, downstream_status,
                )
            except Exception:
                downstream_status = "error"
                logger.exception("summarize: downstream POST failed | meeting_id=%s", req.meeting_id)

        logger.info(
            "summarize: done | trace_id=%s | meeting_id=%s | downstream_status=%s",
            trace_id, req.meeting_id, downstream_status,
        )

        return {
            "trace_id": trace_id,
            "meeting_id": req.meeting_id,
            "latency_ms": latency_ms,
            "valid_json": valid_json,
            "summary": summary_obj,
            "raw_text": raw_text if not valid_json else None,
            "downstream_status": downstream_status,
        }

    except Exception as exc:
        logger.exception("summarize: unhandled error | trace_id=%s | meeting_id=%s", trace_id, req.meeting_id)
        raise HTTPException(status_code=500, detail=str(exc))

if __name__ == "__main__":
    import os

    import uvicorn
    reload = os.getenv("SUMMARY_API_RELOAD", "0") == "1"
    uvicorn.run("summary_api:app", host="127.0.0.1", port=8092, reload=reload)
