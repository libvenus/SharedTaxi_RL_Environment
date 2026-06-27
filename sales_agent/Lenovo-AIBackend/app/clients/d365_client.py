"""HTTP client for the Lenovo D365 Sales backend.

Sprint 1A · US01 — Joining the Meetings  (resolve-opportunity)
Sprint 1A · US02 — Consent & Recording   (resolve-by-emails)
Sprint 1A · US04 — Data Hygiene           (list-active-opps + fetch-risks)

The Note-Taking Agent doesn't have access to the D365 CRM tables, so:

  1. Before the bot schedules itself, it asks D365 "which deal does
     this meeting belong to?" via ``POST /api/meetings/resolve-opportunity``
     (US01 — see :func:`resolve_opportunity`).

  2. Once the bot is in the meeting and the transcript has started,
     it asks D365 "tell me about these attendees" via
     ``POST /api/contacts/resolve-by-emails`` (US02 — see
     :func:`resolve_contacts_by_email`). The bot uses the result to
     tag each utterance with the speaker's CRM name / role / contact_id.

  3. The daily-scan job (US04) enumerates active opportunities via
     ``GET /api/opportunities`` (paginated) and fetches per-deal
     risk flags via ``GET /api/opportunities/{id}/risks`` to
     materialise data-hygiene tasks. See :func:`list_active_opportunities`
     and :func:`fetch_opportunity_risks`.

All functions follow the same error-handling contract:

  - 404 from D365 → return ``None`` / empty list. 404 is benign.
  - 5xx / timeout / 422 / unexpected payload → raise
    :class:`D365ClientError`. Caller logs and continues (do NOT crash
    the bot's / scan's flow on D365 hiccups).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterator, Optional
from uuid import UUID

import httpx

from app.core.config import D365_BASE_URL, D365_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)

_RESOLVE_PATH = "/api/meetings/resolve-opportunity"
_CONTACT_RESOLVE_PATH = "/api/contacts/resolve-by-emails"
_OPPORTUNITIES_PATH = "/api/opportunities"
_OPPORTUNITY_RISKS_PATH_TMPL = "/api/opportunities/{opportunity_id}/risks"
_BRIEFING_CONTEXT_PATH = "/api/briefing/context"


@dataclass(frozen=True)
class ResolveResult:
    """Successful resolver match — value object the bot writes back to its row."""

    opportunity_id: UUID
    account_id: Optional[UUID]
    opportunity_name: Optional[str]
    account_name: Optional[str]
    match_score: float
    matched_by: str            # 'contact_email' | 'subject_keyword' | 'both'
    matched_contact_count: int


@dataclass(frozen=True)
class ContactResolveResult:
    """Per-email lookup result from D365's contact resolver.

    All optional fields are ``None`` if the email isn't an active CRM
    contact — the bot uses that signal to render the speaker as
    "Unknown Attendee" in the transcript (per US02 AC #3).

    ``role`` comes from the most senior ``lvo_opportunitycontact`` link
    (decision-maker first, then alphabetic by role). ``None`` if the
    contact isn't on any active opportunity.
    """

    email: str                              # echoed back, lower-cased
    contact_id: Optional[UUID] = None
    name: Optional[str] = None
    job_title: Optional[str] = None
    account_id: Optional[UUID] = None
    account_name: Optional[str] = None
    role: Optional[str] = None


class D365ClientError(RuntimeError):
    """Raised on transport errors / non-2xx that aren't a benign 404.

    Callers may retry, queue the call, or surface to a human-in-the-loop
    path. We deliberately don't subclass for 4xx vs 5xx vs timeout — the
    bot's behaviour is the same for all three (don't block the join, log
    it, move on).
    """


def resolve_opportunity(
    attendee_emails: list[str],
    subject: Optional[str] = None,
    organiser_email: Optional[str] = None,
    *,
    base_url: Optional[str] = None,
    timeout: Optional[float] = None,
) -> Optional[ResolveResult]:
    """Ask D365 which CRM opportunity a meeting belongs to.

    Returns
    -------
    ResolveResult | None
        ``None`` if D365 returned 404 (no matching active deal — this is
        a normal outcome, not an error). Otherwise a populated result.

    Raises
    ------
    D365ClientError
        On 5xx, 422, transport timeout, or unexpected response shape.
        Bot's caller should log + continue (not crash the schedule loop).
    """
    if not attendee_emails:
        # Defensive — D365 would 422 anyway, but no point spending an
        # HTTP round-trip to learn that.
        raise ValueError("attendee_emails must contain at least one address")

    payload = {"attendeeEmails": attendee_emails}
    if subject:
        payload["subject"] = subject
    if organiser_email:
        payload["organiserEmail"] = organiser_email

    url = (base_url or D365_BASE_URL).rstrip("/") + _RESOLVE_PATH
    request_timeout = timeout if timeout is not None else D365_TIMEOUT_SECONDS

    try:
        response = httpx.post(url, json=payload, timeout=request_timeout)
    except httpx.TimeoutException as exc:
        logger.warning(
            "D365 resolver timed out after %.1fs (url=%s)",
            request_timeout,
            url,
        )
        raise D365ClientError(f"D365 resolver timeout: {exc}") from exc
    except httpx.HTTPError as exc:
        logger.warning("D365 resolver transport error: %s", exc)
        raise D365ClientError(f"D365 resolver transport error: {exc}") from exc

    if response.status_code == 404:
        # Benign — no matching deal. Bot joins without an opportunity tag.
        logger.info(
            "D365 resolver: no match for %d emails / subject=%r",
            len(attendee_emails),
            subject,
        )
        return None

    if response.status_code >= 400:
        logger.warning(
            "D365 resolver returned %d: %s",
            response.status_code,
            response.text[:500],
        )
        raise D365ClientError(
            f"D365 resolver returned {response.status_code}: {response.text[:500]}"
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise D365ClientError(
            f"D365 resolver returned non-JSON body: {response.text[:500]}"
        ) from exc

    try:
        return ResolveResult(
            opportunity_id=UUID(str(data["opportunityId"])),
            account_id=(
                UUID(str(data["accountId"])) if data.get("accountId") else None
            ),
            opportunity_name=data.get("opportunityName"),
            account_name=data.get("accountName"),
            match_score=float(data["matchScore"]),
            matched_by=str(data["matchedBy"]),
            matched_contact_count=int(data.get("matchedContactCount", 0)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise D365ClientError(
            f"D365 resolver returned unexpected shape: {data!r}"
        ) from exc


# ---------------------------------------------------------------------------
# US02 — Contact-by-email batch resolver
# ---------------------------------------------------------------------------


def resolve_contacts_by_email(
    emails: list[str],
    *,
    base_url: Optional[str] = None,
    timeout: Optional[float] = None,
) -> list[ContactResolveResult]:
    """Bulk-resolve attendee emails to CRM contact + opportunity-role context.

    Called by the Note-Taking Agent ONCE at the start of a meeting (after
    the consent message + transcript-start succeed). The bot then tags
    each transcript utterance locally with the result — no per-utterance
    round-trip back to D365.

    Returns
    -------
    list[ContactResolveResult]
        One entry per supplied email (in the same order), with NULL
        fields when an email isn't a known CRM contact. Bot should
        render those speakers as "Unknown Attendee" in the transcript.

    Raises
    ------
    ValueError
        If ``emails`` is empty (caught client-side to avoid a wasted
        round-trip — D365 would 422 anyway).
    D365ClientError
        On 5xx, 422 (unexpected — we already guard above), transport
        timeout, or unexpected response shape.
    """
    if not emails:
        raise ValueError("emails must contain at least one address")

    payload = {"emails": emails}

    url = (base_url or D365_BASE_URL).rstrip("/") + _CONTACT_RESOLVE_PATH
    request_timeout = timeout if timeout is not None else D365_TIMEOUT_SECONDS

    try:
        response = httpx.post(url, json=payload, timeout=request_timeout)
    except httpx.TimeoutException as exc:
        logger.warning(
            "D365 contact-resolver timed out after %.1fs (url=%s, n_emails=%d)",
            request_timeout,
            url,
            len(emails),
        )
        raise D365ClientError(f"D365 contact-resolver timeout: {exc}") from exc
    except httpx.HTTPError as exc:
        logger.warning("D365 contact-resolver transport error: %s", exc)
        raise D365ClientError(
            f"D365 contact-resolver transport error: {exc}"
        ) from exc

    if response.status_code >= 400:
        logger.warning(
            "D365 contact-resolver returned %d: %s",
            response.status_code,
            response.text[:500],
        )
        raise D365ClientError(
            f"D365 contact-resolver returned {response.status_code}: "
            f"{response.text[:500]}"
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise D365ClientError(
            f"D365 contact-resolver returned non-JSON: {response.text[:500]}"
        ) from exc

    raw_results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(raw_results, list):
        raise D365ClientError(
            f"D365 contact-resolver returned unexpected shape: {data!r}"
        )

    out: list[ContactResolveResult] = []
    for entry in raw_results:
        try:
            out.append(
                ContactResolveResult(
                    email=str(entry["email"]),
                    contact_id=(
                        UUID(str(entry["contactId"]))
                        if entry.get("contactId")
                        else None
                    ),
                    name=entry.get("name"),
                    job_title=entry.get("jobTitle"),
                    account_id=(
                        UUID(str(entry["accountId"]))
                        if entry.get("accountId")
                        else None
                    ),
                    account_name=entry.get("accountName"),
                    role=entry.get("role"),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise D365ClientError(
                f"D365 contact-resolver entry malformed: {entry!r}"
            ) from exc

    logger.info(
        "D365 contact-resolver matched %d / %d supplied emails",
        sum(1 for r in out if r.contact_id is not None),
        len(emails),
    )
    return out


# ---------------------------------------------------------------------------
# US04 — Data-hygiene scan helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OpportunityScanRow:
    """Subset of an opportunity-list row that the data-hygiene detectors
    actually need.

    We deliberately project down from the full ``OpportunityListItem`` to
    just the columns the deterministic detectors read — this keeps the
    coupling to the D365 schema minimal and makes the daily-scan job
    cheap to maintain when D365 evolves.

    All fields are nullable because the upstream payload allows nulls
    (a half-filled opp is the WHOLE POINT of the data-hygiene story).
    """

    opportunity_id: UUID
    name: Optional[str] = None
    account_id: Optional[UUID] = None
    owner_id: Optional[UUID] = None
    stage: Optional[str] = None
    statecode: Optional[str] = None
    estimated_value: Optional[float] = None
    close_date: Optional[date] = None
    last_activity: Optional[datetime] = None


@dataclass(frozen=True)
class OpportunityRisk:
    """One risk flag from D365's ``GET /api/opportunities/{id}/risks``.

    The data-hygiene scan materialises one ``DataTask`` per risk so
    sellers see all open risks in their To-Do queue alongside the
    deterministic detectors.
    """

    category: str
    name: str
    message: str
    risk_id: Optional[str] = None        # lvo_dealriskid when persisted
    detected_at: Optional[datetime] = None


def _parse_date_or_none(raw: object) -> Optional[date]:
    """Best-effort ISO-date parse — returns None on bad / missing input."""
    if raw is None:
        return None
    if isinstance(raw, date) and not isinstance(raw, datetime):
        return raw
    try:
        # FastAPI / Pydantic emits dates as 'YYYY-MM-DD'. Accept the full
        # datetime form too just in case the upstream payload changes.
        return date.fromisoformat(str(raw)[:10])
    except (TypeError, ValueError):
        return None


def _parse_datetime_or_none(raw: object) -> Optional[datetime]:
    """Best-effort ISO-datetime parse — handles trailing 'Z'."""
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw
    text = str(raw).strip()
    if not text:
        return None
    # Python <3.11 doesn't accept the trailing 'Z'; replace with +00:00.
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _parse_uuid_or_none(raw: object) -> Optional[UUID]:
    """Best-effort UUID parse — returns None for missing / malformed values.

    D365's payloads sometimes include hyphenless or lowercased UUIDs
    depending on the column; ``UUID(str(...))`` handles both.
    """
    if raw is None:
        return None
    try:
        return UUID(str(raw))
    except (ValueError, AttributeError, TypeError):
        return None


def list_active_opportunities(
    *,
    page_size: int = 100,
    max_records: Optional[int] = None,
    base_url: Optional[str] = None,
    timeout: Optional[float] = None,
) -> Iterator[OpportunityScanRow]:
    """Stream all opportunities visible to the daily-scan job.

    Yields one ``OpportunityScanRow`` at a time, paginating
    ``GET /api/opportunities?page=N&pageSize=page_size`` from D365 Sales
    until the response is empty or ``max_records`` is reached.

    NOTE on filtering:
      D365's ``/api/opportunities`` endpoint doesn't accept a state-code
      filter directly; we filter to active opps (``statecode != 'Closed*'``)
      *client-side* here. That's fine for v1 — most tenants have far more
      open than closed deals, so the over-fetch is bounded — but if it
      becomes a hotspot, push the predicate into D365.

    Yields
    ------
    OpportunityScanRow
        Active opportunities only — Closed Won / Closed Lost are filtered out.

    Raises
    ------
    D365ClientError
        On 5xx / timeout / unexpected shape. The caller (scan job)
        catches this per-page and surfaces ``opportunities_with_errors``
        in the run summary.
    """
    if page_size < 1 or page_size > 1000:
        raise ValueError("page_size must be between 1 and 1000")

    url = (base_url or D365_BASE_URL).rstrip("/") + _OPPORTUNITIES_PATH
    request_timeout = timeout if timeout is not None else D365_TIMEOUT_SECONDS

    yielded = 0
    page = 1
    while True:
        params = {"page": page, "pageSize": page_size}
        try:
            response = httpx.get(url, params=params, timeout=request_timeout)
        except httpx.TimeoutException as exc:
            logger.warning(
                "D365 opportunities-list timed out (page=%d): %s", page, exc
            )
            raise D365ClientError(f"D365 opportunities-list timeout: {exc}") from exc
        except httpx.HTTPError as exc:
            logger.warning("D365 opportunities-list transport error: %s", exc)
            raise D365ClientError(
                f"D365 opportunities-list transport error: {exc}"
            ) from exc

        if response.status_code >= 400:
            raise D365ClientError(
                f"D365 opportunities-list returned {response.status_code}: "
                f"{response.text[:500]}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise D365ClientError(
                f"D365 opportunities-list returned non-JSON: {response.text[:500]}"
            ) from exc

        items = data.get("items") if isinstance(data, dict) else None
        if not isinstance(items, list):
            raise D365ClientError(
                f"D365 opportunities-list returned unexpected shape: {data!r}"
            )

        if not items:
            # Empty page = end of stream. (D365 still 200s past the last
            # page rather than 404'ing.)
            return

        for entry in items:
            if not isinstance(entry, dict):
                continue
            opp_id = _parse_uuid_or_none(entry.get("id"))
            if opp_id is None:
                # Defensive — every opp must have an id; skip the row
                # rather than abort the whole scan if D365 returns a
                # malformed entry.
                continue

            statecode = entry.get("statecode")
            # Filter closed deals client-side. AC #6: "Closed Won and
            # Closed Lost opportunities are excluded" from duplicate
            # detection — we apply the same exclusion to ALL detectors
            # since none of them are useful on a closed deal.
            if isinstance(statecode, str) and statecode.lower().startswith("closed"):
                continue

            yield OpportunityScanRow(
                opportunity_id=opp_id,
                name=entry.get("name"),
                account_id=_parse_uuid_or_none(entry.get("accountId")),
                owner_id=_parse_uuid_or_none(entry.get("ownerId")),
                stage=(
                    entry.get("stage", {}).get("name")
                    if isinstance(entry.get("stage"), dict)
                    else entry.get("stage")
                ),
                statecode=statecode,
                estimated_value=(
                    float(entry["value"]) if entry.get("value") is not None else None
                ),
                close_date=_parse_date_or_none(entry.get("closeDate")),
                last_activity=_parse_datetime_or_none(entry.get("lastActivity")),
            )

            yielded += 1
            if max_records is not None and yielded >= max_records:
                return

        # If we got fewer than a full page, we're done — don't issue
        # another round-trip.
        if len(items) < page_size:
            return
        page += 1


def fetch_opportunity_risks(
    opportunity_id: UUID,
    *,
    base_url: Optional[str] = None,
    timeout: Optional[float] = None,
) -> list[OpportunityRisk]:
    """Fetch the persisted risk flags for an opportunity.

    Wraps ``GET /api/opportunities/{id}/risks``. D365 already runs
    ``deal_risks.compute_risks()`` and persists into ``lvo_dealrisk``;
    this endpoint returns those rows (or falls back to a live eval).

    The data-hygiene scan calls this per-opp and creates one
    ``DataTask`` per returned risk (``task_kind='risk_flag'``,
    ``evidence_text=risk.message``).

    Returns
    -------
    list[OpportunityRisk]
        Empty list if the opp has no active risks. Empty list on 404
        too — the scan treats "no data" the same as "no risks."

    Raises
    ------
    D365ClientError
        On 5xx / timeout / unexpected shape. The scan continues past
        per-opp errors and surfaces them in the run summary.
    """
    url = (base_url or D365_BASE_URL).rstrip("/") + _OPPORTUNITY_RISKS_PATH_TMPL.format(
        opportunity_id=opportunity_id
    )
    request_timeout = timeout if timeout is not None else D365_TIMEOUT_SECONDS

    try:
        response = httpx.get(url, timeout=request_timeout)
    except httpx.TimeoutException as exc:
        logger.warning("D365 opp-risks timed out: %s", exc)
        raise D365ClientError(f"D365 opp-risks timeout: {exc}") from exc
    except httpx.HTTPError as exc:
        logger.warning("D365 opp-risks transport error: %s", exc)
        raise D365ClientError(f"D365 opp-risks transport error: {exc}") from exc

    if response.status_code == 404:
        # Benign — opp was deleted between list-time and risk-fetch-time,
        # or its risks haven't been persisted yet AND a fresh eval found none.
        return []

    if response.status_code >= 400:
        raise D365ClientError(
            f"D365 opp-risks returned {response.status_code}: {response.text[:500]}"
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise D365ClientError(
            f"D365 opp-risks returned non-JSON: {response.text[:500]}"
        ) from exc

    items = data.get("items") if isinstance(data, dict) else None
    if not isinstance(items, list):
        raise D365ClientError(
            f"D365 opp-risks returned unexpected shape: {data!r}"
        )

    out: list[OpportunityRisk] = []
    for entry in items:
        if not isinstance(entry, dict):
            continue
        try:
            out.append(
                OpportunityRisk(
                    risk_id=(
                        str(entry["id"]) if entry.get("id") is not None else None
                    ),
                    category=str(entry["category"]),
                    name=str(entry["name"]),
                    message=str(entry["message"]),
                    detected_at=_parse_datetime_or_none(entry.get("detectedAt")),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise D365ClientError(
                f"D365 opp-risks entry malformed: {entry!r}"
            ) from exc
    return out


def fetch_briefing_context(
    seller_id: str,
    opportunity_id: str,
    *,
    account_id: str | None = None,
    max_summary_words: int = 100,
    base_url: str | None = None,
    timeout: float | None = None,
) -> dict:
    """Fetch D365 facts + signals for pre-meeting briefing generation."""
    url = (base_url or D365_BASE_URL).rstrip("/") + _BRIEFING_CONTEXT_PATH
    request_timeout = timeout if timeout is not None else D365_TIMEOUT_SECONDS
    params = {
        "sellerId": seller_id,
        "opportunityId": opportunity_id,
        "maxSummaryWords": max_summary_words,
    }
    if account_id:
        params["accountId"] = account_id

    try:
        response = httpx.get(url, params=params, timeout=request_timeout)
    except httpx.TimeoutException as exc:
        raise D365ClientError(f"D365 briefing-context timeout: {exc}") from exc
    except httpx.HTTPError as exc:
        raise D365ClientError(f"D365 briefing-context transport error: {exc}") from exc

    if response.status_code == 403:
        raise D365ClientError("Seller does not own this opportunity.")
    if response.status_code == 404:
        raise D365ClientError("Opportunity not found in D365.")
    if response.status_code >= 400:
        raise D365ClientError(
            f"D365 briefing-context returned {response.status_code}: {response.text[:500]}"
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise D365ClientError(
            f"D365 briefing-context returned non-JSON: {response.text[:500]}"
        ) from exc
    if not isinstance(data, dict):
        raise D365ClientError(
            f"D365 briefing-context returned unexpected shape: {data!r}"
        )
    return data


_SOM_CONTEXT_LAKE_PATH = "/api/sales-operating-model/context-lake"


def fetch_som_context_lake(
    *,
    base_url: str | None = None,
    timeout: float | None = None,
) -> dict:
    """Fetch Sales Operating Model Context Lake for AI agent grounding."""
    url = (base_url or D365_BASE_URL).rstrip("/") + _SOM_CONTEXT_LAKE_PATH
    request_timeout = timeout if timeout is not None else D365_TIMEOUT_SECONDS

    try:
        response = httpx.get(url, timeout=request_timeout)
    except httpx.TimeoutException as exc:
        raise D365ClientError(f"D365 context-lake timeout: {exc}") from exc
    except httpx.HTTPError as exc:
        raise D365ClientError(f"D365 context-lake transport error: {exc}") from exc

    if response.status_code >= 400:
        raise D365ClientError(
            f"D365 context-lake returned {response.status_code}: {response.text[:500]}"
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise D365ClientError(
            f"D365 context-lake returned non-JSON: {response.text[:500]}"
        ) from exc
    if not isinstance(data, dict):
        raise D365ClientError(
            f"D365 context-lake returned unexpected shape: {data!r}"
        )
    return data
