"""Root Agent — WorkflowBuilder with a sales gateway.

Flow:
    User message -> RootAgent -> SalesAgent -> specialist agent when needed.

RootAgent only normalizes intake and forwards to SalesAgent.
SalesAgent is the gateway: it either handles the request itself or routes the
request to EmailAgent or SchedulerAgent.
"""

import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from _shared import build_chat_client
from agent_framework import AgentResponse, Message, WorkflowAgent, WorkflowBuilder, WorkflowContext, executor
from schemas import RootPayload, SalesPayload, SchedulerOutput

from sales_agent.agent import agent as sales_agent_raw
from scheduler_agent.agent import agent as scheduler_agent_raw


_LOG_PATH = Path(__file__).resolve().parents[2] / "orchestrator_api.log"


def _setup_flow_logger() -> logging.Logger:
    logger = logging.getLogger("orchestrator_flow")
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

    file_handler = logging.FileHandler(_LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger


_FLOW_LOGGER = _setup_flow_logger()

_IST = timezone(timedelta(hours=5, minutes=30))
_CANONICAL_ORGANISER = "Lenovo_D365_PoC@sutherlandglobal.com"


def _canonical_organiser(raw: str) -> str:
    """Normalize the organiser email to the canonical Lenovo_D365 form.

    Microsoft Graph returns the mailbox as `LenovoD365_PoC@...` (no underscore
    after 'Lenovo'); the business-facing value must be `Lenovo_D365_PoC@...`.
    """
    value = (raw or "").strip()
    if not value:
        return _CANONICAL_ORGANISER
    if value.lower() == "lenovod365_poc@sutherlandglobal.com":
        return _CANONICAL_ORGANISER
    return value


def _to_ist_iso(raw: str) -> str:
    """Render a datetime in Asia/Kolkata (IST) ISO form.

    Accepts UTC `...Z`, offset-aware, or naive values. Naive values are assumed
    to already be local IST and returned unchanged.
    """
    value = (raw or "").strip()
    if not value:
        return ""
    try:
        if value.endswith("Z"):
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        else:
            dt = datetime.fromisoformat(value)
            if dt.tzinfo is None:
                # No timezone info -> treat as already-local IST wall-clock time.
                return dt.isoformat(timespec="seconds")
        return dt.astimezone(_IST).isoformat(timespec="seconds")
    except ValueError:
        return raw


def _to_utc_z(raw: str) -> str:
    value = (raw or "").strip()
    if not value:
        return ""
    try:
        if value.endswith("Z"):
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        else:
            dt = datetime.fromisoformat(value)
            if dt.tzinfo is None:
                # Naive -> assume IST wall-clock, then express in UTC.
                dt = dt.replace(tzinfo=_IST)
        return dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    except ValueError:
        return raw


def _localize_scheduler_times(payload: SchedulerOutput) -> None:
    """Make api_payload meeting times user-facing IST while keeping UTC copies."""
    api_payload = payload.api_payload
    if api_payload is None:
        return
    raw_start = api_payload.meeting_start_time
    raw_end = api_payload.meeting_end_time
    api_payload.meeting_start_time_utc = _to_utc_z(raw_start)
    api_payload.meeting_end_time_utc = _to_utc_z(raw_end)
    api_payload.meeting_start_time = _to_ist_iso(raw_start)
    api_payload.meeting_end_time = _to_ist_iso(raw_end)
    api_payload.organiser_name = _canonical_organiser(api_payload.organiser_name)



_ROOT_INSTRUCTIONS = """\
You are the Lenovo Nitrogen Copilot root intake agent.

Voice and greeting policy:
- For a simple greeting from the user (for example: hi, hello, hey), the orchestrator greeting text should be:
    "Hey, I am Lenovo sales AI assistant. How can I help you today?"

You do not solve the user's request except for simple greetings.

Respond with ONLY a JSON object:
{"target": "sales" | "orchestrator", "message": "<normalized text>"}

Rules:
- Always set target="sales".
- Preserve the user's original message exactly in the message field.
- For a simple greeting (for example: hi, hello, hey), set target="orchestrator"
    and set message to exactly:
    "Hey, I am Lenovo sales AI assistant. How can I help you today?"
- For all non-greeting requests, do not answer the request.
- Route by intent from the user's query:
    - target="email" for email tasks (draft/send/refine/rewrite email, subject/body edits).
    - target="scheduler" for scheduling tasks (schedule/reschedule/cancel meeting, book call, change time/attendees).
    - target="sales" for CRM/data lookup/general sales requests and anything unclear.
"""

_root_agent = build_chat_client().as_agent(

    name="RootAgent",
    instructions=_ROOT_INSTRUCTIONS,
)

_root_session = _root_agent.create_session()
_sales_session = sales_agent_raw.create_session()
_scheduler_session = scheduler_agent_raw.create_session()


@dataclass
class RoutedMessage:
    target: str
    text: str


def _truncate(value: Any, limit: int = 300) -> str:
    text = str(value).replace("\n", " ").strip()
    return text if len(text) <= limit else text[:limit] + "…"


def _sanitize_scheduler_reply(payload: SchedulerOutput) -> str:
    reply = (payload.ai_reply or "").strip()
    if not reply:
        return reply
    if payload.action == "none" and payload.status == "pending" and "event id" in reply.lower():
        return (
            "I could not find the exact meeting yet. Please confirm attendee email and the current "
            "meeting start time in ISO format (for example 2026-06-23T21:30:00.000Z). "
            "I will use organiser Lenovo_D365_PoC@sutherlandglobal.com automatically."
        )
    return reply


def _log_agent_io(agent_name: str, response: AgentResponse) -> None:
    """Log DevUI-style detail: reasoning, tool calls, tool results, and reply."""
    for message in response.messages or []:
        for content in message.contents or []:
            ctype = getattr(content, "type", None)
            if ctype == "text_reasoning":
                if getattr(content, "text", None):
                    _FLOW_LOGGER.info("🧠 reasoning | %s | %s", agent_name, _truncate(content.text))
            elif ctype == "function_call":
                _FLOW_LOGGER.info(
                    "🔧 tool.call | %s | %s(%s)",
                    agent_name,
                    getattr(content, "name", "?"),
                    _truncate(getattr(content, "arguments", "")),
                )
            elif ctype == "function_result":
                exc = getattr(content, "exception", None)
                if exc:
                    _FLOW_LOGGER.info("🔧 tool.result | %s | ERROR: %s", agent_name, _truncate(exc))
                else:
                    _FLOW_LOGGER.info("🔧 tool.result | %s | %s", agent_name, _truncate(getattr(content, "result", "")))
            elif ctype == "text":
                if getattr(content, "text", None):
                    _FLOW_LOGGER.info("💬 reply | %s | %s", agent_name, _truncate(content.text))


async def _yield_text(
    ctx: WorkflowContext[Any, str],
    reply: str,
) -> None:
    await ctx.yield_output(reply)


@executor(id="run_root")
async def run_root(messages: list[Message], ctx: WorkflowContext[RoutedMessage]) -> None:
    """Run the root intake agent and route by intent."""
    _FLOW_LOGGER.debug("node.run_root.enter")
    text = ""
    for msg in reversed(messages):
        if msg.role == "user":
            text = " ".join(str(content) for content in msg.contents) if msg.contents else ""
            break

    response: AgentResponse = await _root_agent.run(
        [Message(role="user", contents=[text])],
        stream=False,
        session=_root_session,
        options={"response_format": RootPayload},
    )
    _log_agent_io("RootAgent", response)

    payload = response.value
    if isinstance(payload, RootPayload):
        normalized = payload.message.strip() if payload.message else text
        target = payload.target
    else:
        normalized = text
        target = "sales"

    if target == "orchestrator":
        _FLOW_LOGGER.info("💬 reply | Orchestrator | %s", _truncate(normalized))
        await _yield_text(ctx, normalized)
        return

    _FLOW_LOGGER.info("➡️ route | root -> %s", target)

    await ctx.send_message(RoutedMessage(target=target, text=normalized))


@executor(id="run_sales")
async def run_sales(msg: RoutedMessage, ctx: WorkflowContext[RoutedMessage, str]) -> None:
    """Run SalesAgent as the only gateway and optionally route to specialists."""
    _FLOW_LOGGER.debug("node.run_sales.enter")
    text = msg.text
    response: AgentResponse = await sales_agent_raw.run(
        [Message(role="user", contents=[text])],
        stream=False,
        session=_sales_session,
        options={"response_format": SalesPayload},
    )
    _log_agent_io("SalesAgent", response)

    payload = response.value
    if not isinstance(payload, SalesPayload):
        await _yield_text(ctx, response.text)
        return

    target = payload.target
    if target == "scheduler":
        _FLOW_LOGGER.info("➡️ route | sales -> scheduler")
        await ctx.send_message(RoutedMessage(target="scheduler", text=payload.message or text))
        return

    await _yield_text(ctx, payload.reply or response.text)


def _route_case(expected: str):
    def condition(message: Any) -> bool:
        return isinstance(message, RoutedMessage) and message.target == expected

    return condition


@executor(id="run_scheduler")
async def run_scheduler(msg: RoutedMessage, ctx: WorkflowContext[RoutedMessage, str]) -> None:
    """Run SchedulerAgent after SalesAgent routes the request."""
    _FLOW_LOGGER.debug("node.run_scheduler.enter")
    response: AgentResponse = await scheduler_agent_raw.run(
        [Message(role="user", contents=[msg.text])],
        stream=False,
        session=_scheduler_session,
        options={"response_format": SchedulerOutput},
    )
    _log_agent_io("SchedulerAgent", response)

    payload = response.value
    if isinstance(payload, SchedulerOutput):
        if payload.handoff == "sales":
            _FLOW_LOGGER.info("➡️ route | scheduler -> sales (out-of-scope)")
            await ctx.send_message(RoutedMessage(target="sales", text=payload.message or msg.text))
            return
        # Booked meeting -> keep full structured JSON so api_payload survives.
        # Question / disambiguation -> surface the readable reply.
        if payload.api_payload is not None:
            _localize_scheduler_times(payload)
            await _yield_text(ctx, payload.model_dump_json())
        else:
            await _yield_text(ctx, _sanitize_scheduler_reply(payload) or response.text)
        return

    await _yield_text(ctx, response.text)


_workflow = (
    WorkflowBuilder(start_executor=run_root)
    # Root is pure intake — always forwards to sales
    .add_edge(run_root, run_sales, condition=_route_case("sales"))
    # Sales is the domain coordinator — routes to scheduler when needed
    .add_edge(run_sales, run_scheduler, condition=_route_case("scheduler"))
    # Loopback: scheduler hands back to sales when out of scope
    .add_edge(run_scheduler, run_sales, condition=_route_case("sales"))
    .build()
)

agent = WorkflowAgent(
    workflow=_workflow,
    name="Orchestrator",
    description="Lenovo Nitrogen Copilot — RootAgent forwards to SalesAgent, which routes to SchedulerAgent when needed. Email is handled via REST API.",
)

