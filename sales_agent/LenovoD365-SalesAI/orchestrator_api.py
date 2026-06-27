import sys
import uuid
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_framework import Message
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

_ROOT = Path(__file__).resolve().parent
_AGENTS_DIR = _ROOT / "agents"
if str(_AGENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENTS_DIR))

from orchestrator.agent import agent as orchestrator_agent  # type: ignore[import-not-found]

from core.envelope import build_envelope
from db.session_store import SessionStore

app = FastAPI(title="Orchestrator API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_LOGS_DIR = _ROOT / "logs"
_LOGS_DIR.mkdir(exist_ok=True)
_LOG_PATH = _ROOT / "orchestrator_api.log"  # always-on rolling file
_DATED_LOG_PATH = _LOGS_DIR / f"orchestrator_api_{datetime.now().strftime('%Y-%m-%d')}.log"
_MAX_HISTORY_MESSAGES = 20
_SESSIONS = SessionStore(_ROOT / "db" / "chat_history.json", max_messages=_MAX_HISTORY_MESSAGES)


@app.get("/")
async def root() -> FileResponse:
    return FileResponse(_ROOT / "web" / "index.html")


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("orchestrator_api")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # Main rolling log (always appended)
    file_handler = logging.FileHandler(_LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Dated log — new file per day, e.g. logs/orchestrator_api_2026-06-22.log
    dated_handler = logging.FileHandler(_DATED_LOG_PATH, encoding="utf-8")
    dated_handler.setFormatter(formatter)
    logger.addHandler(dated_handler)

    # Also wire root logger so module loggers (e.g. scheduler.*) are visible.
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console)

    def _has_file_handler(path: Path) -> bool:
        return any(
            isinstance(h, logging.FileHandler) and Path(getattr(h, "baseFilename", "")) == path
            for h in root_logger.handlers
        )

    if not _has_file_handler(_LOG_PATH):
        root_file_handler = logging.FileHandler(_LOG_PATH, encoding="utf-8")
        root_file_handler.setFormatter(formatter)
        root_logger.addHandler(root_file_handler)

    if not _has_file_handler(_DATED_LOG_PATH):
        root_dated_handler = logging.FileHandler(_DATED_LOG_PATH, encoding="utf-8")
        root_dated_handler.setFormatter(formatter)
        root_logger.addHandler(root_dated_handler)

    logger.propagate = False
    return logger


_LOGGER = _setup_logger()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _extract_handoff_info(response_text: str) -> tuple[str | None, str | None]:
    try:
        payload = json.loads(response_text)
    except Exception:
        return None, None

    if not isinstance(payload, dict):
        return None, None
    agent_name = payload.get("agent_name")
    action = payload.get("action")
    return (
        str(agent_name) if isinstance(agent_name, str) and agent_name else None,
        str(action) if isinstance(action, str) and action else None,
    )


def _extract_response_observability(native: dict[str, Any]) -> dict[str, Any]:
    messages = native.get("messages") if isinstance(native, dict) else []
    if not isinstance(messages, list):
        messages = []

    content_types: list[str] = []
    reasoning_count = 0
    assistant_preview = ""

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        contents = msg.get("contents")
        if not isinstance(contents, list):
            continue
        for content in contents:
            if not isinstance(content, dict):
                continue
            ctype = content.get("type")
            if isinstance(ctype, str):
                content_types.append(ctype)
                if ctype == "text_reasoning":
                    reasoning_count += 1

            if not assistant_preview and msg.get("role") == "assistant":
                text = content.get("text")
                if isinstance(text, str) and text.strip():
                    assistant_preview = text.strip().replace("\n", " ")[:240]

    return {
        "response_id": native.get("response_id") if isinstance(native, dict) else None,
        "message_count": len(messages),
        "content_types": sorted(set(content_types)),
        "reasoning_count": reasoning_count,
        "assistant_preview": assistant_preview,
    }


def _message_text(msg: dict) -> str:
    contents = msg.get("contents") if isinstance(msg, dict) else None
    if not isinstance(contents, list):
        return ""
    parts = [c.get("text") for c in contents if isinstance(c, dict) and isinstance(c.get("text"), str)]
    return "\n".join(p for p in parts if p and p.strip())


def _clean_history_messages(messages: list) -> list[dict[str, Any]]:
    """Return {role, text} rows with assistant turns run through the envelope so
    stored raw scheduler JSON renders as display text, not a JSON blob."""
    rows: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role") or "unknown"
        text = _message_text(msg)
        if role == "assistant" and text:
            text = build_envelope(text).display.text or text
        rows.append({"role": role, "contents": [{"type": "text", "text": text}]})
    return rows


class OrchestratorChatRequest(BaseModel):
    text: str = Field(..., description="User message for the orchestrator")
    conversation_id: str = Field(default="conv_default", description="Conversation/session key")
    trace_id: str | None = Field(default=None, description="Caller trace id")


@app.get("/v1/orchestrator/history/{conversation_id}")
async def orchestrator_history(conversation_id: str) -> dict[str, Any]:
    _LOGGER.info("📚 history.request | conversation_id=%s", conversation_id)
    session = _SESSIONS.get(conversation_id)
    if session is None:
        _LOGGER.warning("⚠️ history.miss | conversation_id=%s", conversation_id)
        raise HTTPException(status_code=404, detail="conversation_id not found")

    cleaned = _clean_history_messages(_SESSIONS.messages(conversation_id))

    _LOGGER.info(
        "📚 history.ok | conversation_id=%s messages=%d",
        conversation_id,
        len(cleaned),
    )

    return {
        "type": "history_response",
        "conversation_id": conversation_id,
        "messages": cleaned,
        "count": len(cleaned),
    }


@app.post("/v1/orchestrator/reset/{conversation_id}")
async def orchestrator_reset(conversation_id: str) -> dict[str, Any]:
    """Clear a conversation's history. Frontend calls this on page load so each
    fresh page starts with an empty chat."""
    existed = _SESSIONS.delete(conversation_id)
    _LOGGER.info("\U0001f9f9 reset | conversation_id=%s existed=%s", conversation_id, existed)
    return {
        "type": "reset_response",
        "conversation_id": conversation_id,
        "cleared": existed,
    }


@app.get("/health")
async def health() -> dict[str, bool]:
    return {"ok": True}


@app.post("/v1/orchestrator/chat")
async def orchestrator_chat(req: OrchestratorChatRequest) -> dict[str, Any]:
    trace_id = req.trace_id or f"tr_{uuid.uuid4().hex}"
    started = datetime.now(timezone.utc)
    _LOGGER.info(
        "🚀 chat.request | trace_id=%s conversation_id=%s ts=%s",
        trace_id,
        req.conversation_id,
        _utc_now(),
    )

    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    session = _SESSIONS.get(req.conversation_id)
    if session is None:
        session = orchestrator_agent.create_session()
        _SESSIONS.add(req.conversation_id, session)
        _LOGGER.info("🧠 session.created | trace_id=%s conversation_id=%s", trace_id, req.conversation_id)
    else:
        _LOGGER.info("🧠 session.reused | trace_id=%s conversation_id=%s", trace_id, req.conversation_id)

    try:
        _LOGGER.info("🤖 orchestrator.run.start | trace_id=%s", trace_id)
        response = await orchestrator_agent.run(
            [Message(role="user", contents=[text])],
            stream=False,
            session=session,
        )
        latency_ms = int((datetime.now(timezone.utc) - started).total_seconds() * 1000)
        native = response.to_dict(exclude_none=True)
        obs = _extract_response_observability(native)
        _LOGGER.info(
            "📝 response.native | trace_id=%s response_id=%s messages=%d content_types=%s",
            trace_id,
            obs.get("response_id") or "",
            obs.get("message_count") or 0,
            ",".join(obs.get("content_types") or []),
        )
        if (obs.get("reasoning_count") or 0) > 0:
            _LOGGER.info(
                "💭 thinking.detected | trace_id=%s reasoning_items=%d",
                trace_id,
                obs.get("reasoning_count") or 0,
            )
        if obs.get("assistant_preview"):
            _LOGGER.info(
                "🧾 response.preview | trace_id=%s text=%s",
                trace_id,
                obs.get("assistant_preview"),
            )

        agent_name, action = _extract_handoff_info(response.text)
        if agent_name and agent_name != "Orchestrator":
            _LOGGER.info(
                "🔀 handoff.detected | trace_id=%s to_agent=%s action=%s latency_ms=%d",
                trace_id,
                agent_name,
                action or "",
                latency_ms,
            )
        _LOGGER.info("✅ orchestrator.run.ok | trace_id=%s latency_ms=%d", trace_id, latency_ms)
        _SESSIONS.trim(session)
        _SESSIONS.save()
    except Exception as exc:
        latency_ms = int((datetime.now(timezone.utc) - started).total_seconds() * 1000)
        _LOGGER.exception("❌ orchestrator.run.fail | trace_id=%s latency_ms=%d error=%s", trace_id, latency_ms, str(exc))
        raise HTTPException(status_code=500, detail=str(exc))

    envelope = build_envelope(response.text, req.conversation_id, trace_id)
    return envelope.model_dump()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("orchestrator_api:app", host="0.0.0.0", port=9091, reload=False)
