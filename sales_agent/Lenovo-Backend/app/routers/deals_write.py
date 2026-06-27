"""Write (mutating) endpoints for the Deal Update user story.

Endpoints
---------
PATCH  /api/opportunities/{opportunity_id}
    Update editable fields on an existing deal.
    - Open deals: name, estimatedValue, estimatedCloseDate, closeProbability,
      forecastCategory, saleMotion are all editable.
    - Closed Won / Closed Lost: those fields are locked; only competitors and
      next-actions (via their own sub-routes) may be updated.
    - Stage is NOT editable via this endpoint (stage advancement is a separate
      workflow; Qualify-stage deals are stage-locked per user story).
    - Owner is read-only and never included in update payloads.

POST   /api/opportunities/{opportunity_id}/competitors
    Add a competitor to the deal.

PATCH  /api/opportunities/{opportunity_id}/competitors/{competitor_id}
    Update an existing competitor association.

DELETE /api/opportunities/{opportunity_id}/competitors/{competitor_id}
    Soft-delete (Inactive) a competitor from the deal.

GET    /api/opportunities/{opportunity_id}/next-actions
    List all active next actions for the deal.

POST   /api/opportunities/{opportunity_id}/next-actions
    Add a new next action (status defaults to Open).

PATCH  /api/opportunities/{opportunity_id}/next-actions/{action_id}
    Update a next action, including marking it as Completed.
    Completed actions are retained for history.

All C/U/D operations append a row to lvo_audit_log.
Pass the optional X-User-Id header to record who made the change.
"""

import json
import uuid
from datetime import datetime, timezone

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Body,
    Depends,
    Header,
    HTTPException,
    Path,
    status,
)
from sqlalchemy import String, cast, func, select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import FileNotes, NextAction, Opportunity, OpportunityCompetitor
from app.normalizers import normalise_sale_motion, normalise_stage
from app.services.audit_log import write_audit_event
from app.services.deal_recalc import recalculate_async
from app.schemas import (
    Competitor,
    CompetitorCreateRequest,
    CompetitorDeleteResponse,
    CompetitorUpdateRequest,
    NextActionCreateRequest,
    NextActionItem,
    NextActionListResponse,
    NextActionUpdateRequest,
    Note,
    NoteCreateRequest,
    NoteResponse,
    NoteUpdateRequest,
    OpportunityDeleteResponse,
    OpportunityUpdateRequest,
    OpportunityUpdateResponse,
    SaleMotionRef,
    StageRef,
)

router = APIRouter(prefix="/api/opportunities", tags=["deals-write"])

# ---------------------------------------------------------------------------
# Stage / forecast business vocabulary
# ---------------------------------------------------------------------------

OPEN_STAGES = {"Qualify", "Develop", "Propose", "Execute"}
CLOSED_STAGES = {"Closed Won", "Closed Lost"}
CLOSED_STATECODES = {"Won", "Closed Won", "Lost", "Closed Lost"}

# Valid forecastCategory values for open-stage deals.
VALID_OPEN_FORECAST_CATEGORIES = {"Open", "Pipeline", "Best Case", "Most Likely", "Commit"}

# Fields that cannot be changed once a deal is closed.
_CLOSED_LOCKED_FIELDS = frozenset(
    {"name", "estimated_value", "estimated_close_date",
     "close_probability", "forecast_category", "sale_motion"}
)

def _get_note_or_404(
    db: Session,
    note_id: str
):
    note = db.execute(
        select(FileNotes).where(
            FileNotes.opportunityid == id,
            
        )
    ).scalar_one_or_none()

    if not note:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "NOTE_NOT_FOUND",
                "message": "Note not found"
            }
        )

    return note
def _enqueue_recalc(background_tasks: BackgroundTasks, opportunity_id: str) -> None:
    """Enqueue an async deal-health recalculation.

    Called after every successful C/U/D so the score and risks reflect the
    latest state. The task opens its own DB session because the request
    session is closed by the time FastAPI runs the callback.
    """
    background_tasks.add_task(recalculate_async, opportunity_id)

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _get_opportunity_or_404(db: Session, opportunity_id: str) -> Opportunity:
    opp = db.get(Opportunity, opportunity_id)
    if opp is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Opportunity '{opportunity_id}' not found.",
        )
    return opp


def _get_competitor_or_404(
    db: Session, opportunity_id: str, competitor_id: str
) -> OpportunityCompetitor:
    comp = db.get(OpportunityCompetitor, competitor_id)
    if comp is None or comp.statecode != "Active":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Competitor '{competitor_id}' not found.",
        )
    if (comp.lvo_opportunityid or "").upper() != opportunity_id.upper():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Competitor '{competitor_id}' does not belong to "
                f"opportunity '{opportunity_id}'."
            ),
        )
    return comp


def _get_next_action_or_404(
    db: Session, opportunity_id: str, action_id: str
) -> NextAction:
    action = db.get(NextAction, action_id)
    if action is None or action.statecode != "Active":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Next action '{action_id}' not found.",
        )
    if (action.lvo_opportunityid or "").upper() != opportunity_id.upper():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Next action '{action_id}' does not belong to "
                f"opportunity '{opportunity_id}'."
            ),
        )
    return action


def _is_closed(opp: Opportunity) -> bool:
    """Return True when the deal is in a Closed Won / Closed Lost state."""
    return (
        (opp.statecode or "") in CLOSED_STATECODES
        or (opp.stagename or "") in CLOSED_STAGES
    )


def _opp_snapshot(opp: Opportunity) -> dict:
    """Serialisable snapshot of opportunity fields used in audit diffs.

    Covers every column the PATCH endpoint can mutate so the audit log
    captures a faithful before/after — including the Complete-Information
    fields added in the v0.13.0 / 2026-06-12 surface expansion.
    """
    return {
        "name": opp.name,
        "estimatedvalue": (
            str(opp.estimatedvalue) if opp.estimatedvalue is not None else None
        ),
        "estimatedclosedate": (
            str(opp.estimatedclosedate) if opp.estimatedclosedate is not None else None
        ),
        "closeprobability": (
            str(opp.closeprobability) if opp.closeprobability is not None else None
        ),
        "lvo_forecastcategory": opp.lvo_forecastcategory,
        "lvo_salesmotion": opp.lvo_salesmotion,
        "stagename": opp.stagename,
        "statecode": opp.statecode,
        # Complete-Information form fields.
        "lvo_summary": opp.lvo_summary,
        "lvo_priority": opp.lvo_priority,
        "lvo_leadorigin": opp.lvo_leadorigin,
        "lvo_partnerinvolved": bool(opp.lvo_partnerinvolved)
        if opp.lvo_partnerinvolved is not None
        else None,
        "lvo_parentopportunityid": opp.lvo_parentopportunityid,
        "lvo_stageentrydate": (
            str(opp.lvo_stageentrydate)
            if opp.lvo_stageentrydate is not None
            else None
        ),
        "owninguser": opp.owninguser,
    }


# ---------------------------------------------------------------------------
# Parent / Child Opportunity guards
# ---------------------------------------------------------------------------

# Maximum depth we'll walk when checking for a cycle. Caps runtime on a deeply
# nested or genuinely-corrupt parent chain; any tree this deep is almost
# certainly broken data and the caller will see ``INVALID_PARENT_OPPORTUNITY``.
_MAX_PARENT_CYCLE_DEPTH = 10


def _assert_no_parent_cycle(
    db: Session, deal_id: str, proposed_parent_id: str
) -> None:
    """Reject an update where the proposed parent would create a cycle.

    Walks ``proposed_parent_id`` upward via ``lvo_parentopportunityid`` and
    raises 422 ``INVALID_PARENT_OPPORTUNITY`` if it ever lands on
    ``deal_id`` itself, OR if the chain is longer than
    ``_MAX_PARENT_CYCLE_DEPTH`` (defensive cap).

    Self-references are caught by the caller before this helper is invoked,
    but the loop here also protects against ``proposed_parent_id == deal_id``
    in case that pre-check is ever skipped.
    """
    deal_id_upper = str(deal_id).upper()
    cursor: str | None = str(proposed_parent_id).upper()
    seen: set[str] = set()

    for _ in range(_MAX_PARENT_CYCLE_DEPTH):
        if cursor is None:
            return
        if cursor == deal_id_upper:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "code": "INVALID_PARENT_OPPORTUNITY",
                    "message": (
                        "parentOpportunityId would create a cycle in the "
                        "parent / child hierarchy."
                    ),
                },
            )
        if cursor in seen:
            # Cycle that doesn't involve our deal — still bad data, refuse.
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "code": "INVALID_PARENT_OPPORTUNITY",
                    "message": (
                        "Detected a pre-existing cycle in the parent / child "
                        "hierarchy. Resolve the existing data before reparenting."
                    ),
                },
            )
        seen.add(cursor)
        next_cursor = db.execute(
            select(Opportunity.lvo_parentopportunityid).where(
                func.upper(cast(Opportunity.opportunityid, String)) == cursor
            )
        ).scalar_one_or_none()
        if next_cursor is None:
            return
        cursor = next_cursor.upper()

    # Ran out of iterations — treat as "tree too deep, assume cycle".
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail={
            "code": "INVALID_PARENT_OPPORTUNITY",
            "message": (
                f"parentOpportunityId chain exceeds the maximum depth of "
                f"{_MAX_PARENT_CYCLE_DEPTH}. Investigate the data manually."
            ),
        },
    )


def _write_audit(
    db: Session,
    *,
    entity_type: str,
    entity_id: str,
    opportunity_id: str | None,
    action: str,
    changed_by: str | None,
    diff: dict | None = None,
) -> None:
    """Append one row to lvo_audit_log (not yet committed — caller commits)."""
    write_audit_event(
        db,
        entity_type=entity_type,
        entity_id=entity_id,
        opportunity_id=opportunity_id,
        action=action,
        category="crm_writeback",
        actor_type="seller",
        changed_by=changed_by,
        diff=diff,
    )


# ---------------------------------------------------------------------------
# PATCH /api/opportunities/{opportunity_id}  — Update deal
# ---------------------------------------------------------------------------

@router.get(
    "/opportunity/{opportunity_id}/notes",
    response_model=list[Note]
)
def get_notes(
    opportunity_id: str = Path(..., description="opportunity.opportunityid (UUID)"),
    db: Session = Depends(get_db)
):

    notes = db.execute(
        select(FileNotes).where(
            FileNotes.opportunity_id == opportunity_id
            
        )
    ).scalars().all()

    return [
        Note(
            id=str(note.id),
            notes=note.notes
        )
        for note in notes
    ]
@router.post(
    "/notes",
    response_model=NoteResponse,
    status_code=status.HTTP_200_OK,
    summary="Create or Update Notes"
)
def save_notes(
  
    body: NoteUpdateRequest = None,
    db: Session = Depends(get_db),
):
    
    note = db.execute(
        select(FileNotes).where(
            FileNotes.opportunity_id == body.opportunity_id,
            FileNotes.statecode == "Active"
        )
    ).scalar_one_or_none()

    # UPDATE
    if note:
        note.notes = body.notes

        db.commit()
        db.refresh(note)

        return NoteResponse(
            id=str(note.id),
            opportunity_id=body.opportunity_id,
            notes=note.notes,
            action="updated"
        )

    # INSERT
    note = FileNotes(
        id=str(uuid.uuid4()),
        opportunity_id=body.opportunity_id,
        notes=body.notes,
        statecode="Active"
    )

    db.add(note)
    db.commit()
    db.refresh(note)

    return NoteResponse(
        
        notes=note.notes,
        action="created"
    )

@router.patch(
    "/{opportunity_id}",
    response_model=OpportunityUpdateResponse,
    summary="Update editable fields on an existing deal",
    responses={
        400: {"description": "Closed-deal field locked / business-rule violation"},
        404: {"description": "Opportunity not found"},
        422: {"description": "Validation error (ERR_MSG_0012, duplicate name, …)"},
    },
)
def update_opportunity(
    background_tasks: BackgroundTasks,
    opportunity_id: str = Path(..., description="opportunity.opportunityid (UUID)"),
    body: OpportunityUpdateRequest = Body(...),
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    db: Session = Depends(get_db),
) -> OpportunityUpdateResponse:
    """Update an existing deal.

    Business rules enforced
    -----------------------
    * **Closed deal lock** — `name`, `estimatedValue`, `estimatedCloseDate`,
      `closeProbability`, `forecastCategory`, and `saleMotion` are read-only
      once a deal is Closed Won or Closed Lost.
    * **Stage lock** — stage is never modified by this endpoint.  It is
      flagged as locked in the response when the deal is at Qualify stage or
      is closed.
    * **ERR_MSG_0012** — `estimatedValue == 0` is rejected with this code.
    * **Duplicate name** — rejected if another opportunity already uses the
      same name (case-insensitive).
    * **Forecast alignment** — for open-stage deals the value must be one of:
      Open, Pipeline, Best Case, Most Likely, Commit.
    * **Audit** — every successful update appends a row to `lvo_audit_log`
      with a before/after diff.
    """
    opp = _get_opportunity_or_404(db, opportunity_id)
    closed = _is_closed(opp)

    provided: set[str] = body.model_fields_set

    # -- Closed-deal field lock --------------------------------------------
    if closed:
        locked_provided = provided & _CLOSED_LOCKED_FIELDS
        if locked_provided:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "code": "CLOSED_DEAL_LOCKED",
                    "message": (
                        f"Fields {sorted(locked_provided)} cannot be modified on a "
                        "Closed Won / Closed Lost deal. Only competitors and next "
                        "actions may be updated via their respective sub-routes."
                    ),
                },
            )

    before = _opp_snapshot(opp)

    # -- name --------------------------------------------------------------
    if "name" in provided:
        if not body.name or not body.name.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"code": "VALIDATION_ERROR", "message": "name cannot be empty."},
            )
        name_clean = body.name.strip()
        dup = db.execute(
            select(Opportunity.opportunityid).where(
                func.lower(Opportunity.name) == name_clean.lower(),
                cast(Opportunity.opportunityid, String) != opportunity_id,
            )
        ).scalar()
        if dup:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "code": "DUPLICATE_NAME",
                    "message": "A deal with this name already exists.",
                },
            )
        opp.name = name_clean

    # -- estimated_value ---------------------------------------------------
    if "estimated_value" in provided:
        val = body.estimated_value
        if val is not None:
            if val == 0:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail={
                        "code": "ERR_MSG_0012",
                        "message": "Estimated revenue must be greater than zero.",
                    },
                )
            if val < 0:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail={
                        "code": "VALIDATION_ERROR",
                        "message": "Estimated revenue cannot be negative.",
                    },
                )
        opp.estimatedvalue = val

    # -- estimated_close_date ----------------------------------------------
    if "estimated_close_date" in provided:
        opp.estimatedclosedate = body.estimated_close_date

    # -- close_probability -------------------------------------------------
    if "close_probability" in provided:
        opp.closeprobability = body.close_probability

    # -- forecast_category -------------------------------------------------
    if "forecast_category" in provided and body.forecast_category is not None:
        stage = opp.stagename or ""
        if stage in OPEN_STAGES and body.forecast_category not in VALID_OPEN_FORECAST_CATEGORIES:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "code": "INVALID_FORECAST_CATEGORY",
                    "message": (
                        f"For stage '{stage}', forecastCategory must be one of "
                        f"{sorted(VALID_OPEN_FORECAST_CATEGORIES)}."
                    ),
                },
            )
        opp.lvo_forecastcategory = body.forecast_category

    # -- sale_motion -------------------------------------------------------
    if "sale_motion" in provided:
        opp.lvo_salesmotion = body.sale_motion

    # ----------------------------------------------------------------------
    # Complete-Information (Deal Summary) form fields.
    #
    # Each field below is intentionally NOT in ``_CLOSED_LOCKED_FIELDS`` —
    # they're metadata / categorisation rather than commercial values, and
    # the user story allows editing them after the deal closes (e.g. fixing
    # a wrong priority on an already-won deal). If business rules need to
    # lock specific ones later, just add them to the frozenset above.
    # ----------------------------------------------------------------------
    if "summary" in provided:
        opp.lvo_summary = body.summary

    if "priority" in provided:
        # Pydantic literal already validates the enum; defensive belt-and-
        # braces in case a future caller bypasses model validation.
        if body.priority is not None and body.priority not in {"High", "Medium", "Low"}:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "code": "VALIDATION_ERROR",
                    "message": "priority must be one of High / Medium / Low.",
                },
            )
        opp.lvo_priority = body.priority

    if "lead_origin" in provided:
        # Free-text in v1 (FE governs the dropdown values). Empty string is
        # treated as null so the form's "clear selection" works cleanly.
        opp.lvo_leadorigin = (body.lead_origin or "").strip() or None

    if "partner_involved" in provided:
        opp.lvo_partnerinvolved = bool(body.partner_involved)

    if "parent_opportunity_id" in provided:
        new_parent = body.parent_opportunity_id
        if new_parent is None or new_parent == "":
            opp.lvo_parentopportunityid = None
        else:
            new_parent_str = str(new_parent)
            if new_parent_str.upper() == opportunity_id.upper():
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail={
                        "code": "INVALID_PARENT_OPPORTUNITY",
                        "message": "A deal cannot be its own parent.",
                    },
                )
            # Confirm the proposed parent exists and isn't cancelled —
            # otherwise the child_opportunities derivation on the GET would
            # silently filter it back out.
            parent_row = db.execute(
                select(Opportunity.statecode).where(
                    func.upper(cast(Opportunity.opportunityid, String))
                    == new_parent_str.upper()
                )
            ).scalar_one_or_none()
            if parent_row is None:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail={
                        "code": "INVALID_PARENT_OPPORTUNITY",
                        "message": (
                            f"parentOpportunityId '{new_parent_str}' does not "
                            "exist."
                        ),
                    },
                )
            if (parent_row or "") == "Canceled":
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail={
                        "code": "INVALID_PARENT_OPPORTUNITY",
                        "message": (
                            "Cannot link to a cancelled opportunity. Restore "
                            "the parent or pick a different one."
                        ),
                    },
                )
            _assert_no_parent_cycle(db, opportunity_id, new_parent_str)
            opp.lvo_parentopportunityid = new_parent_str

    if "stage_entry_date" in provided:
        opp.lvo_stageentrydate = body.stage_entry_date

    if "owner_id" in provided:
        # Owner is just a UUID pointer — we don't validate against systemuser
        # (some dumps don't ship that table). The audit-log entry captures the
        # change and the GET endpoint resolves the display name on read.
        opp.owninguser = body.owner_id

    # -- Audit + persist ---------------------------------------------------
    after = _opp_snapshot(opp)
    _write_audit(
        db,
        entity_type="opportunity",
        entity_id=opportunity_id,
        opportunity_id=opportunity_id,
        action="update",
        changed_by=x_user_id,
        diff={"before": before, "after": after},
    )
    db.commit()
    _enqueue_recalc(background_tasks, opportunity_id)

    return OpportunityUpdateResponse(
        id=str(opp.opportunityid),
        name=opp.name,
        stage=StageRef(raw=opp.stagename, label=normalise_stage(opp.stagename)),
        statecode=opp.statecode,
        estimated_value=(
            float(opp.estimatedvalue) if opp.estimatedvalue is not None else None
        ),
        estimated_close_date=opp.estimatedclosedate,
        close_probability=(
            float(opp.closeprobability) if opp.closeprobability is not None else None
        ),
        forecast_category=opp.lvo_forecastcategory,
        sale_motion=SaleMotionRef(
            raw=opp.lvo_salesmotion,
            label=normalise_sale_motion(opp.lvo_salesmotion),
        ),
        owner_id=opp.owninguser,
        is_stage_locked=(opp.stagename == "Qualify") or closed,
    )


# ---------------------------------------------------------------------------
# POST /api/opportunities/{opportunity_id}/competitors  — Add competitor
# ---------------------------------------------------------------------------


@router.post(
    "/{opportunity_id}/competitors",
    response_model=Competitor,
    status_code=status.HTTP_201_CREATED,
    summary="Add a competitor to a deal",
    responses={
        409: {"description": "Competitor already associated with this deal"},
    },
)
def add_competitor(
    background_tasks: BackgroundTasks,
    opportunity_id: str = Path(...),
    body: CompetitorCreateRequest = Body(...),
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    db: Session = Depends(get_db),
) -> Competitor:
    """Add a competitor to the deal.

    * `competitorName` is mandatory.
    * `competitorType` must be ``Incumbent`` or ``Secondary`` when provided.
    * A competitor already associated (Active) with this deal cannot be added
      again (duplicate check is case-insensitive on `lvo_competitorname`).
    """
    _get_opportunity_or_404(db, opportunity_id)

    if not body.competitor_name or not body.competitor_name.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "VALIDATION_ERROR", "message": "competitorName is mandatory."},
        )
    competitor_name = body.competitor_name.strip()

    # Duplicate check — same deal, same competitor name (case-insensitive)
    already_exists = db.execute(
        select(OpportunityCompetitor.lvo_opportunitycompetitorid).where(
            func.upper(OpportunityCompetitor.lvo_opportunityid) == opportunity_id.upper(),
            func.lower(OpportunityCompetitor.lvo_competitorname) == competitor_name.lower(),
            OpportunityCompetitor.statecode == "Active",
        )
    ).scalar()
    if already_exists:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "DUPLICATE_COMPETITOR",
                "message": f"'{competitor_name}' is already associated with this deal.",
            },
        )

    if body.competitor_type is not None and body.competitor_type not in {
        "Incumbent",
        "Secondary",
    }:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "VALIDATION_ERROR",
                "message": "competitorType must be 'Incumbent' or 'Secondary'.",
            },
        )

    new_id = str(uuid.uuid4())
    comp = OpportunityCompetitor(
        lvo_opportunitycompetitorid=new_id,
        lvo_name=competitor_name,
        lvo_opportunityid=opportunity_id.upper(),
        lvo_competitorname=competitor_name,
        lvo_competitortype=body.competitor_type,
        lvo_resellingpartner=body.reselling_partner,
        statecode="Active",
    )
    db.add(comp)

    _write_audit(
        db,
        entity_type="competitor",
        entity_id=new_id,
        opportunity_id=opportunity_id,
        action="create",
        changed_by=x_user_id,
        diff={
            "competitor_name": competitor_name,
            "competitor_type": body.competitor_type,
            "reselling_partner": body.reselling_partner,
        },
    )
    db.commit()
    _enqueue_recalc(background_tasks, opportunity_id)

    return Competitor(
        id=comp.lvo_opportunitycompetitorid,
        opportunity_id=opportunity_id,
        name=comp.lvo_name,
        competitor_name=comp.lvo_competitorname,
        competitor_type=comp.lvo_competitortype,
        reselling_partner_id=comp.lvo_resellingpartner,
    )


# ---------------------------------------------------------------------------
# PATCH /api/opportunities/{opportunity_id}/competitors/{competitor_id}
# ---------------------------------------------------------------------------


@router.patch(
    "/{opportunity_id}/competitors/{competitor_id}",
    response_model=Competitor,
    summary="Update an existing competitor association",
)
def update_competitor(
    background_tasks: BackgroundTasks,
    opportunity_id: str = Path(...),
    competitor_id: str = Path(...),
    body: CompetitorUpdateRequest = Body(...),
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    db: Session = Depends(get_db),
) -> Competitor:
    """Update a competitor record on the deal.

    * `competitorName` is mandatory.
    * The new name must not clash with another active competitor on the same
      deal (excluding the record being updated).
    """
    _get_opportunity_or_404(db, opportunity_id)
    comp = _get_competitor_or_404(db, opportunity_id, competitor_id)

    if not body.competitor_name or not body.competitor_name.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "VALIDATION_ERROR", "message": "competitorName is mandatory."},
        )
    competitor_name = body.competitor_name.strip()

    # Duplicate check — exclude the record being updated
    dup = db.execute(
        select(OpportunityCompetitor.lvo_opportunitycompetitorid).where(
            func.upper(OpportunityCompetitor.lvo_opportunityid) == opportunity_id.upper(),
            func.lower(OpportunityCompetitor.lvo_competitorname) == competitor_name.lower(),
            OpportunityCompetitor.statecode == "Active",
            OpportunityCompetitor.lvo_opportunitycompetitorid != competitor_id,
        )
    ).scalar()
    if dup:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "DUPLICATE_COMPETITOR",
                "message": f"'{competitor_name}' is already associated with this deal.",
            },
        )

    if body.competitor_type is not None and body.competitor_type not in {
        "Incumbent",
        "Secondary",
    }:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "VALIDATION_ERROR",
                "message": "competitorType must be 'Incumbent' or 'Secondary'.",
            },
        )

    before = {
        "competitor_name": comp.lvo_competitorname,
        "competitor_type": comp.lvo_competitortype,
        "reselling_partner": comp.lvo_resellingpartner,
    }

    comp.lvo_name = competitor_name
    comp.lvo_competitorname = competitor_name
    comp.lvo_competitortype = body.competitor_type
    comp.lvo_resellingpartner = body.reselling_partner

    _write_audit(
        db,
        entity_type="competitor",
        entity_id=competitor_id,
        opportunity_id=opportunity_id,
        action="update",
        changed_by=x_user_id,
        diff={
            "before": before,
            "after": {
                "competitor_name": competitor_name,
                "competitor_type": body.competitor_type,
                "reselling_partner": body.reselling_partner,
            },
        },
    )
    db.commit()
    _enqueue_recalc(background_tasks, opportunity_id)

    return Competitor(
        id=comp.lvo_opportunitycompetitorid,
        opportunity_id=opportunity_id,
        name=comp.lvo_name,
        competitor_name=comp.lvo_competitorname,
        competitor_type=comp.lvo_competitortype,
        reselling_partner_id=comp.lvo_resellingpartner,
    )


# ---------------------------------------------------------------------------
# DELETE /api/opportunities/{opportunity_id}/competitors/{competitor_id}
# ---------------------------------------------------------------------------


@router.delete(
    "/{opportunity_id}/competitors/{competitor_id}",
    response_model=CompetitorDeleteResponse,
    summary="Remove a competitor from a deal (soft-delete)",
)
def delete_competitor(
    background_tasks: BackgroundTasks,
    opportunity_id: str = Path(...),
    competitor_id: str = Path(...),
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    db: Session = Depends(get_db),
) -> CompetitorDeleteResponse:
    """Soft-delete a competitor by setting its statecode to ``Inactive``.

    The record is retained for history.  The response includes the
    `competitor_id` so the frontend can implement an undo action by
    calling the add-competitor endpoint again.
    """
    _get_opportunity_or_404(db, opportunity_id)
    comp = _get_competitor_or_404(db, opportunity_id, competitor_id)

    removed_name = comp.lvo_competitorname or competitor_id
    comp.statecode = "Inactive"

    _write_audit(
        db,
        entity_type="competitor",
        entity_id=competitor_id,
        opportunity_id=opportunity_id,
        action="delete",
        changed_by=x_user_id,
        diff={"competitor_name": removed_name},
    )
    db.commit()
    _enqueue_recalc(background_tasks, opportunity_id)

    return CompetitorDeleteResponse(
        opportunity_id=opportunity_id,
        competitor_id=competitor_id,
        message=f"Competitor '{removed_name}' has been removed from the deal.",
    )


# ---------------------------------------------------------------------------
# GET /api/opportunities/{opportunity_id}/next-actions
# ---------------------------------------------------------------------------


@router.get(
    "/{opportunity_id}/next-actions",
    response_model=NextActionListResponse,
    summary="List next actions for a deal (Open and Completed)",
)
def list_next_actions(
    opportunity_id: str = Path(...),
    db: Session = Depends(get_db),
) -> NextActionListResponse:
    """Return all active next actions for the deal, newest first.

    Completed actions are included so they are available for history and
    tracking per acceptance criteria.
    """
    _get_opportunity_or_404(db, opportunity_id)

    rows = (
        db.execute(
            select(NextAction)
            .where(
                func.upper(NextAction.lvo_opportunityid) == opportunity_id.upper(),
                NextAction.statecode == "Active",
            )
            .order_by(NextAction.lvo_createdat.desc())
        )
        .scalars()
        .all()
    )

    items = [
        NextActionItem(
            id=a.lvo_nextactionid,
            opportunity_id=a.lvo_opportunityid,
            description=a.lvo_description,
            due_date=a.lvo_duedate,
            verbal_written_acceptance=a.verbal_written_acceptance,
            verbal_commit_date=a.verbal_commit_date,
            status=a.lvo_status,
            created_at=a.lvo_createdat,
            updated_at=a.lvo_updatedat,
            created_by=a.lvo_createdby,
        )
        for a in rows
    ]

    return NextActionListResponse(
        opportunity_id=opportunity_id,
        total=len(items),
        items=items,
    )


# ---------------------------------------------------------------------------
# POST /api/opportunities/{opportunity_id}/next-actions  — Add next action
# ---------------------------------------------------------------------------


@router.post(
    "/{opportunity_id}/next-actions",
    response_model=NextActionItem,
    status_code=status.HTTP_201_CREATED,
    summary="Add a next action to a deal",
)
def add_next_action(
    background_tasks: BackgroundTasks,
    opportunity_id: str = Path(...),
    body: NextActionCreateRequest = Body(...),
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    db: Session = Depends(get_db),
) -> NextActionItem:
    """Create a new next action with status **Open**.

    * `description` is mandatory.
    * `dueDate` is optional.
    * Multiple next actions per deal are supported.
    """
    _get_opportunity_or_404(db, opportunity_id)

    if not body.description or not body.description.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "VALIDATION_ERROR", "message": "description is mandatory."},
        )

    now = datetime.now(timezone.utc)
    new_id = str(uuid.uuid4())
    action = NextAction(
        lvo_nextactionid=new_id,
        lvo_opportunityid=opportunity_id.upper(),
        lvo_description=body.description.strip(),
        lvo_duedate=body.due_date,
        lvo_status="Open",
        verbal_written_acceptance=body.verbal_written_acceptance.strip(),
        verbal_commit_date=body.verbal_commit_date,
        lvo_createdat=now,
        lvo_updatedat=now,
        lvo_createdby=x_user_id,
        statecode="Active",
    )
    db.add(action)

    _write_audit(
        db,
        entity_type="next_action",
        entity_id=new_id,
        opportunity_id=opportunity_id,
        action="create",
        changed_by=x_user_id,
        diff={
            "description": body.description.strip(),
            "verbal_written_acceptance":action.verbal_written_acceptance,
            "verbal_commit_date":str(action.verbal_commit_date) if action.verbal_commit_date else None,
            "due_date": str(body.due_date) if body.due_date else None,
            "status": "Open",
        },
    )
    db.commit()
    _enqueue_recalc(background_tasks, opportunity_id)

    return NextActionItem(
        id=action.lvo_nextactionid,
        opportunity_id=action.lvo_opportunityid,
        description=action.lvo_description,
        due_date=action.lvo_duedate,
        status=action.lvo_status,
        verbal_written_acceptance=action.verbal_written_acceptance,
        verbal_commit_date=action.verbal_commit_date,
        created_at=action.lvo_createdat,
        updated_at=action.lvo_updatedat,
        created_by=action.lvo_createdby,
    )


# ---------------------------------------------------------------------------
# PATCH /api/opportunities/{opportunity_id}/next-actions/{action_id}
# ---------------------------------------------------------------------------


@router.patch(
    "/{opportunity_id}/next-actions/{action_id}",
    response_model=NextActionItem,
    summary="Update a next action (including mark as Completed)",
)
def update_next_action(
    background_tasks: BackgroundTasks,
    opportunity_id: str = Path(...),
    action_id: str = Path(...),
    body: NextActionUpdateRequest = Body(...),
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    db: Session = Depends(get_db),
) -> NextActionItem:
    """Update description, due date, or status of a next action.

    Set ``status`` to ``"Completed"`` to mark the action done.  Completed
    actions are **retained** (not deleted) for history and tracking.

    Only fields that are explicitly provided in the request body are applied
    (true PATCH semantics via `model_fields_set`).
    """
    _get_opportunity_or_404(db, opportunity_id)
    action = _get_next_action_or_404(db, opportunity_id, action_id)

    provided: set[str] = body.model_fields_set

    before = {
        "description": action.lvo_description,
        "due_date": str(action.lvo_duedate) if action.lvo_duedate else None,
        "verbal_written_acceptance":action.verbal_written_acceptance,
        "verbal_commit_date":str(action.verbal_commit_date) if action.verbal_commit_date else None,
        "status": action.lvo_status,
    }

    if "description" in provided:
        if body.description is not None and not body.description.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "code": "VALIDATION_ERROR",
                    "message": "description cannot be empty.",
                },
            )
        if body.description is not None:
            action.lvo_description = body.description.strip()

    if "due_date" in provided:
        action.lvo_duedate = body.due_date

    if "status" in provided and body.status is not None:
        action.lvo_status = body.status
    if "verbal_written_acceptance" in provided:
        action.lvo_verbal_written_acceptance = body.verbal_written_acceptance.strip() if body.verbal_written_acceptance else None
    if "verbal_commit_date" in provided:
        action.lvo_verbal_commit_date = body.verbal_commit_date
    action.lvo_updatedat = datetime.now(timezone.utc)

    after = {
        "description": action.lvo_description,
        "due_date": str(action.lvo_duedate) if action.lvo_duedate else None,
        "status": action.lvo_status,
        "verbal_written_acceptance": action.lvo_verbal_written_acceptance,
        "verbal_commit_date": str(action.lvo_verbal_commit_date) if action.lvo_verbal_commit_date else None,
    }

    _write_audit(
        db,
        entity_type="next_action",
        entity_id=action_id,
        opportunity_id=opportunity_id,
        action="update",
        changed_by=x_user_id,
        diff={"before": before, "after": after},
    )
    db.commit()
    _enqueue_recalc(background_tasks, opportunity_id)

    return NextActionItem(
        id=action.lvo_nextactionid,
        opportunity_id=action.lvo_opportunityid,
        description=action.lvo_description,
        due_date=action.lvo_duedate,
        status=action.lvo_status,
        verbal_written_acceptance=action.lvo_verbal_written_acceptance,
        verbal_commit_date=action.lvo_verbal_commit_date,
        created_at=action.lvo_createdat,
        updated_at=action.lvo_updatedat,
        created_by=action.lvo_createdby,
    )


# ---------------------------------------------------------------------------
# DELETE /api/opportunities/{opportunity_id}  — Soft-delete (Canceled)
# ---------------------------------------------------------------------------


CANCELED_STATECODE = "Canceled"


@router.delete(
    "/{opportunity_id}",
    response_model=OpportunityDeleteResponse,
    summary="Soft-delete a deal (statecode='Canceled')",
    responses={
        404: {"description": "Opportunity not found"},
        409: {"description": "Deal already canceled"},
    },
)
def delete_opportunity(
    background_tasks: BackgroundTasks,
    opportunity_id: str = Path(..., description="opportunity.opportunityid (UUID)"),
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    db: Session = Depends(get_db),
) -> OpportunityDeleteResponse:
    """Soft-delete the deal by setting ``statecode='Canceled'``.

    Why a soft delete?
    ------------------
    The "View deal details" story wants Delete to remove a deal from the
    pipeline view, but the rest of the app relies on the row continuing to
    exist for audit, account-rollup history, and the activity timeline.
    Setting ``statecode='Canceled'`` (distinct from Open / Won / Lost) ticks
    both boxes — the KPI buckets in ``app/routers/opportunities.py``
    exclude this state implicitly because none of their predicates match it.

    Side-effects
    ------------
    * Writes an ``lvo_audit_log`` row tagged ``action='delete'`` so the
      mutation is traceable to whoever sent ``X-User-Id``.
    * Enqueues a deal-health recalculation so the on-disk score and risks
      reflect the new state.

    Idempotency
    -----------
    Calling DELETE on an already-canceled deal returns 409 with code
    ``ALREADY_CANCELED`` so the FE can distinguish a real error from a
    no-op retry.
    """
    opp = _get_opportunity_or_404(db, opportunity_id)

    if (opp.statecode or "") == CANCELED_STATECODE:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "ALREADY_CANCELED",
                "message": "Deal has already been canceled.",
            },
        )

    before_state = opp.statecode
    opp.statecode = CANCELED_STATECODE

    _write_audit(
        db,
        entity_type="opportunity",
        entity_id=opportunity_id,
        opportunity_id=opportunity_id,
        action="delete",
        changed_by=x_user_id,
        diff={
            "before": {"statecode": before_state},
            "after": {"statecode": CANCELED_STATECODE},
        },
    )
    db.commit()
    _enqueue_recalc(background_tasks, opportunity_id)

    return OpportunityDeleteResponse(
        id=opportunity_id,
        statecode=CANCELED_STATECODE,
        message=f"Deal '{opp.name or opportunity_id}' has been canceled.",
    )
