"""Analyst agent (Lenovo Nitrogen Copilot).

Thin data accessor. Tools return raw records / aggregates; the LLM does
the reasoning.
"""

import json
from functools import cache
from pathlib import Path
from typing import Annotated

from _shared import build_chat_client
from pydantic import Field


@cache
def _load_data():
    """Load analyst agent data from local data.json."""
    data_path = Path(__file__).parent / "data.json"
    if not data_path.exists():
        return {"opportunities": {}, "seller_activities": {}}
    return json.loads(data_path.read_text(encoding="utf-8"))


_DATA = _load_data()
OPPORTUNITIES     = _DATA["opportunities"]
SELLER_ACTIVITIES = _DATA["seller_activities"]

_NAME = "AnalystAgent"

_DESCRIPTION = "Lenovo analyst copilot - reads pipeline aggregates, seller activity, and raw opportunity records."

_INSTRUCTIONS = """\
You are the Lenovo Analyst. Answer questions about pipeline, seller activity, and deal context using tool data only.

Tools: get_seller_activities, get_pipeline_by_stage, get_opportunity.

Rules:
1. Every number you state must come from a tool call. Never invent figures.
2. For qualitative questions (risk, overload), call the relevant tool and reason from its fields.
3. Answer and stop. Do not ask follow-up questions.
4. SCOPE: if the request is NOT about pipeline / seller activity / opportunity records (e.g. send/draft an email, schedule/reschedule/cancel a meeting, summarise a transcript, anything else), refuse in ONE line: "That's not an analyst task — please ask the Orchestrator (it'll route to the Email, Schedule, or Summary specialist)." Do NOT draft, template, outline, or improvise.\
"""


def get_seller_activities(
    seller_id: Annotated[str, Field(description="Seller email, e.g. 'alex.tan@lenovo.com'.")],
):
    """Return the seller's call / email / meeting counts for the last 7 days."""
    activity = SELLER_ACTIVITIES.get(seller_id)
    if not activity:
        return f"No activity data for {seller_id}."
    return (
        f"Activity for {seller_id} (last 7d):\n"
        f"- Calls:    {activity['calls_last_7d']}\n"
        f"- Emails:   {activity['emails_last_7d']}\n"
        f"- Meetings: {activity['meetings_last_7d']}\n"
        f"- Hours booked next 7d: {activity['hours_booked_next_7d']}"
    )


def get_pipeline_by_stage():
    """Aggregate open pipeline value by sales stage."""
    by_stage = {}
    for opp in OPPORTUNITIES.values():
        by_stage[opp["stage"]] = by_stage.get(opp["stage"], 0) + opp["value_usd"]
    total = sum(by_stage.values())
    lines = [f"Open pipeline by stage (total ${total:,}):"]
    for stage in ("Qualify", "Develop", "Propose", "Close"):
        if stage in by_stage:
            value = by_stage[stage]
            lines.append(f"- {stage:<8}: ${value:,} ({value / total * 100:.0f}%)")
    return "\n".join(lines)


def get_opportunity(
    opportunity_id: Annotated[str, Field(description="Opportunity ID, e.g. 'opp-101'.")],
):
    """Return the full opportunity record so the LLM can reason about it."""
    record = OPPORTUNITIES.get(opportunity_id)
    if not record:
        return f"Opportunity {opportunity_id} not found."
    lines = [f"{record['name']} ({opportunity_id})"]
    for key, value in record.items():
        if key == "name":
            continue
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)


agent = build_chat_client().as_agent(
    name=_NAME,
    description=_DESCRIPTION,
    instructions=_INSTRUCTIONS,
    require_per_service_call_history_persistence=True,
    tools=[
        get_seller_activities,
        get_pipeline_by_stage,
        get_opportunity,
    ],
)
