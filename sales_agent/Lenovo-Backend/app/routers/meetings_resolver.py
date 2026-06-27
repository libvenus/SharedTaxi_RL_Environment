"""Meeting → CRM resolution endpoint.

Sprint 1A · User Story 01 — Joining the Meetings

The Note-Taking Agent (Lenovo-AIBackend) needs to know which CRM deal a
Teams meeting belongs to BEFORE the bot schedules itself to join. The
agent doesn't have access to our CRM tables, so it sends us the
attendee emails (and optionally the subject) and asks "which deal is
this?".

Algorithm
---------
1. Lower-case all supplied attendee emails (D365 stores ``contact.emailaddress1``
   in mixed case; we compare case-insensitively).
2. Find every Active ``contact`` whose email matches any supplied address.
3. Walk ``lvo_opportunitycontact`` (Active links only) → ``opportunity``
   (non-Cancelled deals only).
4. Group by opportunity, count how many of the supplied emails landed on
   each deal — that's the base score.
5. If ``subject`` was supplied AND it contains any token from the deal's
   ``name`` (case-insensitive, length ≥ 4), boost the score.
6. Return the highest-scoring deal. 404 if every candidate scores 0.

Score formula (kept simple for v1; revisit when we have data):

    contact_score   = matched_contacts_on_this_deal / total_supplied_emails
    subject_score   = 0.3  if subject contains a deal-name token
                    | 0.0  otherwise
    final           = clamp(contact_score + subject_score, 0.0, 1.0)

So with all attendees matching one deal AND the subject hitting:
    1.0 + 0.3 → clamped to 1.0  (matched_by="both")

With 2/3 attendees matching one deal:
    0.67   (matched_by="contact_email")

With no attendee matches but subject hits a unique deal name:
    0.3    (matched_by="subject_keyword")  -- NOT covered in v1, see notes
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Iterable

from fastapi import APIRouter, Body, Depends, HTTPException, status
from sqlalchemy import String, cast, func, select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Account, Contact, Opportunity, OpportunityContact
from app.schemas import (
    OpportunityResolveMatchedBy,
    OpportunityResolveRequest,
    OpportunityResolveResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/meetings", tags=["meetings-resolver"])


# Cancelled deals are excluded from candidates regardless of contact match —
# the bot should never be told to join a meeting tied to a dead deal.
_EXCLUDED_OPP_STATECODES = ("Canceled", "Cancelled", "Lost")

# Subject-keyword extraction: words ≥ 4 chars, alphanumeric. Filters out the
# usual short connectives ("the", "and", "for") that produce false positives.
_SUBJECT_TOKEN_RE = re.compile(r"\b[A-Za-z0-9]{4,}\b")

# How much weight a subject hit adds to the final score.
_SUBJECT_BOOST = 0.3


def _normalise_emails(emails: Iterable[str]) -> list[str]:
    """Lower-case, strip, de-dupe — preserving original order."""
    seen: set[str] = set()
    out: list[str] = []
    for raw in emails:
        if not raw:
            continue
        cleaned = raw.strip().lower()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            out.append(cleaned)
    return out


def _subject_tokens(subject: str | None) -> set[str]:
    """Tokenise a meeting subject for keyword boosting."""
    if not subject:
        return set()
    return {tok.lower() for tok in _SUBJECT_TOKEN_RE.findall(subject)}


def _name_tokens(name: str | None) -> set[str]:
    """Same tokenisation rules applied to a deal name for fair comparison."""
    return _subject_tokens(name)


@router.post(
    "/resolve-opportunity",
    response_model=OpportunityResolveResponse,
    summary="Resolve a Teams meeting to a CRM opportunity (attendee emails + subject)",
    responses={
        404: {"description": "No matching active deal found"},
        422: {"description": "attendee_emails is empty or malformed"},
    },
)
def resolve_opportunity(
    payload: OpportunityResolveRequest = Body(...),
    db: Session = Depends(get_db),
) -> OpportunityResolveResponse:
    """Match a meeting to its CRM deal so the Note-Taking Agent can scope itself."""
    emails = _normalise_emails(payload.attendee_emails)
    if not emails:
        # Pydantic min_length=1 already catches truly-empty lists; this guards
        # against lists of all-empty / whitespace strings.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "EMPTY_ATTENDEE_EMAILS",
                "message": "attendee_emails contains no usable addresses.",
            },
        )

    # ---- 1. Find the contacts whose email matches any supplied address ----
    contact_rows = db.execute(
        select(Contact.contactid, Contact.emailaddress1)
        .where(
            func.lower(Contact.emailaddress1).in_(emails),
            Contact.statecode == "Active",
        )
    ).all()

    if not contact_rows:
        logger.info(
            "No CRM contacts matched the supplied emails (%d emails)",
            len(emails),
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "NO_CONTACT_MATCH",
                "message": (
                    "None of the supplied attendee emails match an active "
                    "contact in the CRM."
                ),
            },
        )

    contact_ids = [str(row.contactid) for row in contact_rows]

    # ---- 2. Find Active opportunity-contact links → candidate deals --------
    deal_rows = db.execute(
        select(
            OpportunityContact.lvo_opportunityid,
            OpportunityContact.lvo_contactid,
            Opportunity.opportunityid,
            Opportunity.name,
            Opportunity.accountid,
            Opportunity.statecode,
        )
        .join(
            Opportunity,
            func.upper(cast(Opportunity.opportunityid, String))
            == func.upper(OpportunityContact.lvo_opportunityid),
        )
        .where(
            func.upper(OpportunityContact.lvo_contactid).in_(
                [cid.upper() for cid in contact_ids]
            ),
            OpportunityContact.statecode == "Active",
            (Opportunity.statecode.notin_(_EXCLUDED_OPP_STATECODES))
            | (Opportunity.statecode.is_(None)),
        )
    ).all()

    if not deal_rows:
        logger.info(
            "Contacts matched (%d) but none are linked to an active deal",
            len(contact_ids),
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "NO_DEAL_MATCH",
                "message": (
                    "Matched contacts exist but none are linked to an "
                    "active opportunity."
                ),
            },
        )

    # ---- 3. Group by deal, count distinct matched emails per deal ----------
    # We tally distinct contacts per deal so a contact appearing on multiple
    # links doesn't double-count.
    by_deal: dict[str, dict] = defaultdict(
        lambda: {
            "name": None,
            "account_id": None,
            "matched_contact_ids": set(),
        }
    )
    for row in deal_rows:
        opp_id_upper = str(row.opportunityid).upper()
        bucket = by_deal[opp_id_upper]
        bucket["name"] = row.name
        bucket["account_id"] = row.accountid
        bucket["matched_contact_ids"].add(str(row.lvo_contactid).upper())

    # ---- 4. Score each candidate -------------------------------------------
    subject_toks = _subject_tokens(payload.subject)
    total_supplied = len(emails)

    scored: list[tuple[str, float, OpportunityResolveMatchedBy, dict]] = []
    for opp_id_upper, bucket in by_deal.items():
        contact_score = len(bucket["matched_contact_ids"]) / max(total_supplied, 1)

        deal_toks = _name_tokens(bucket["name"])
        subject_hit = bool(subject_toks & deal_toks)
        subject_score = _SUBJECT_BOOST if subject_hit else 0.0

        # Cap at 1.0 so the response stays in [0, 1]
        final_score = min(contact_score + subject_score, 1.0)

        if contact_score > 0 and subject_hit:
            matched_by: OpportunityResolveMatchedBy = "both"
        elif subject_hit:
            matched_by = "subject_keyword"
        else:
            matched_by = "contact_email"

        scored.append((opp_id_upper, final_score, matched_by, bucket))

    # ---- 5. Pick the winner ------------------------------------------------
    # Highest score wins; ties broken by most-matched-contacts then alphabetic
    # by name for determinism.
    scored.sort(
        key=lambda x: (
            -x[1],                                # score desc
            -len(x[3]["matched_contact_ids"]),    # contact count desc
            (x[3]["name"] or "").lower(),         # name asc for tie-break
        )
    )
    winner_opp_id, winner_score, winner_matched_by, winner_bucket = scored[0]

    if winner_score == 0.0:
        # Defensive — shouldn't happen given the joins above, but if every
        # candidate ends up at 0 we treat it as no match.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "NO_DEAL_MATCH",
                "message": "Best candidate scored 0; no deal returned.",
            },
        )

    # ---- 6. Hydrate the account name (cheap single-row lookup) -------------
    account_name: str | None = None
    if winner_bucket["account_id"]:
        account_name = db.execute(
            select(Account.name).where(
                func.upper(cast(Account.accountid, String))
                == str(winner_bucket["account_id"]).upper()
            )
        ).scalar_one_or_none()

    return OpportunityResolveResponse(
        opportunity_id=winner_opp_id,
        account_id=(
            str(winner_bucket["account_id"])
            if winner_bucket["account_id"]
            else None
        ),
        opportunity_name=winner_bucket["name"],
        account_name=account_name,
        match_score=round(winner_score, 3),
        matched_by=winner_matched_by,
        matched_contact_count=len(winner_bucket["matched_contact_ids"]),
    )
