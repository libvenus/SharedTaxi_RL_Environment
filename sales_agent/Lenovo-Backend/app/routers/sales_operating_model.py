"""Sprint 2 · US 3.2.1 — Sales Operating Model · Interview-First Setup."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from sqlalchemy.orm import Session
from starlette.responses import Response

from app.database import get_db
from app.schemas import (
    SomAgentImpactDetail,
    SomConfigurationStatusResponse,
    SomContextLakeInterviewBlock,
    SomContextLakeOrgIntentBlock,
    SomContextLakeResponse,
    SomContextLakeRoleBlock,
    SomContextLakeTimelineBlock,
    SomIntentCardItem,
    SomIntentCardsResponse,
    SomInterviewDraftRequest,
    SomInterviewQuestionAdminItem,
    SomInterviewQuestionCreateRequest,
    SomInterviewQuestionItem,
    SomInterviewQuestionUpdateRequest,
    SomInterviewSaveRequest,
    SomInterviewSaveResponse,
    SomInterviewSetupResponse,
    SomOrganizationalIntentDetailResponse,
    SomOrganizationalIntentListResponse,
    SomOrganizationalIntentCustomMetricItem,
    SomOrganizationalIntentMetricCreateRequest,
    SomOrganizationalIntentMetricUpdateRequest,
    SomFieldNotFoundDetail,
    SomOrganizationalIntentFieldDeleteResponse,
    SomOrganizationalIntentBulkSaveRequest,
    SomOrganizationalIntentBulkSaveResponse,
    SomOrganizationalIntentDeleteResponse,
    SomOrganizationalIntentSaveRequest,
    SomOrganizationalIntentSaveResponse,
    SomOrganizationalIntentSummary,
    SomTimelineCardDetailResponse,
    SomTimelineCardListResponse,
    SomTimelineCardSaveRequest,
    SomTimelineCardSaveResponse,
    SomTimelineCardSummary,
    SomTimelineConfigurationStatusResponse,
    SomTimelineSectionStatusResponse,
    SomValidationErrorDetail,
)
from app.services.sales_operating_model import (
    ERR_MSG_0021,
    InterviewSetupData,
    build_interview_setup,
    create_interview_question,
    delete_interview_question,
    get_context_lake,
    list_intent_cards,
    list_interview_questions,
    save_draft_responses,
    save_interview_responses,
    update_interview_question,
    validate_role,
)
from app.services.som_organizational_intent import (
    ERR_MSG_0024,
    OrgIntentBulkSaveResult,
    OrgIntentCustomMetric,
    OrgIntentDetail,
    OrgIntentDeleteResult,
    OrgIntentFieldDeleteResult,
    OrgIntentSaveResult,
    SomDeleteConflictError,
    SomFieldNotFoundError,
    SomFieldValidationError,
    SomMetricNotFoundError,
    add_organizational_intent_metric,
    delete_organizational_intent,
    delete_organizational_intent_field,
    delete_organizational_intent_metric,
    get_configuration_status,
    get_organizational_intent,
    list_organizational_intents,
    save_organizational_intent,
    save_organizational_intents_bulk,
    update_organizational_intent_metric,
    validate_intent_type,
)
from app.services.som_timeline_classification import (
    ERR_MSG_0029,
    ERR_MSG_0030,
    INFO_MSG_0006,
    INFO_MSG_0006_MESSAGE,
    SomAgentImpactRequired,
    SomFieldValidationError as TimelineFieldValidationError,
    SomSectionLockedError,
    TimelineCardDetail,
    TimelineSaveResult,
    get_timeline_card,
    get_timeline_configuration_status,
    get_timeline_section_status,
    list_timeline_cards,
    save_timeline_card,
    validate_card_type,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/sales-operating-model",
    tags=["sales-operating-model"],
)


def _to_setup_response(data: InterviewSetupData) -> SomInterviewSetupResponse:
    return SomInterviewSetupResponse(
        role=data.role,  # type: ignore[arg-type]
        role_display=data.role_display,
        scope_label=data.scope_label,
        questions=[
            SomInterviewQuestionItem(
                question_id=q.question_id,
                sort_order=q.sort_order,
                text=q.text,
            )
            for q in data.questions
        ],
        responses=data.draft_responses,
        saved_responses=data.saved_responses,
        captured_count=data.captured_count,
        total_questions=data.total_questions,
        verify_enabled=data.verify_enabled,
        intent_card_status=data.intent_card_status,  # type: ignore[arg-type]
        configured_at=data.configured_at,
    )


def _to_save_response(data: InterviewSetupData) -> SomInterviewSaveResponse:
    return SomInterviewSaveResponse(
        role=data.role,  # type: ignore[arg-type]
        intent_card_status=data.intent_card_status,  # type: ignore[arg-type]
        captured_count=data.captured_count,
        total_questions=data.total_questions,
        verify_enabled=data.verify_enabled,
        configured_at=data.configured_at,
    )


def _responses_payload(body: SomInterviewDraftRequest | SomInterviewSaveRequest) -> list[dict[str, str]]:
    return [{"question_id": r.question_id, "text": r.text} for r in body.responses]


def _interview_intent_cards_response(db: Session) -> SomIntentCardsResponse:
    cards = list_intent_cards(db)
    return SomIntentCardsResponse(
        items=[
            SomIntentCardItem(
                role=c.role,  # type: ignore[arg-type]
                role_display=c.role_display,
                scope_label=c.scope_label,
                status=c.status,  # type: ignore[arg-type]
                configured_at=c.configured_at,
            )
            for c in cards
        ]
    )


def _context_lake_response(raw: dict) -> SomContextLakeResponse:
    interview_roles: dict[str, SomContextLakeRoleBlock] = {}
    for role_key, block in (raw.get("interview") or {}).get("roles") or {}:
        if isinstance(block, dict):
            interview_roles[role_key] = SomContextLakeRoleBlock(
                role_display=block.get("roleDisplay", ""),
                scope_label=block.get("scopeLabel", ""),
                status=block.get("status", "NOT_CONFIGURED"),
                configured_at=block.get("configuredAt"),
                interview_responses=block.get("interviewResponses") or [],
            )

    org_out: dict[str, SomContextLakeOrgIntentBlock] = {}
    for key, block in (raw.get("organizationalIntents") or {}).items():
        if isinstance(block, dict):
            org_out[key] = SomContextLakeOrgIntentBlock(
                display_name=block.get("displayName", key),
                status=block.get("status", "NOT_CONFIGURED"),
                last_synced_at=block.get("lastSyncedAt"),
                is_timeboxed=bool(block.get("isTimeboxed")),
                is_guardrail=bool(block.get("isGuardrail")),
                expiry_date=block.get("expiryDate"),
                fields=block.get("fields") or {},
            )

    timeline_out: dict[str, SomContextLakeTimelineBlock] = {}
    for key, block in (raw.get("timelineClassification") or {}).items():
        if isinstance(block, dict):
            timeline_out[key] = SomContextLakeTimelineBlock(
                display_name=block.get("displayName", key),
                status=block.get("status", "NOT_CONFIGURED"),
                last_synced_at=block.get("lastSyncedAt"),
                fields=block.get("fields") or {},
            )

    return SomContextLakeResponse(
        version=int(raw.get("version") or 2),
        cycle_id=raw.get("cycleId"),
        updated_at=raw.get("updatedAt"),
        interview=SomContextLakeInterviewBlock(roles=interview_roles),
        organizational_intents=org_out,
        timeline_classification=timeline_out,
    )


def _org_metric_item(data: OrgIntentCustomMetric) -> SomOrganizationalIntentCustomMetricItem:
    return SomOrganizationalIntentCustomMetricItem(
        id=data.id,
        label=data.label,
        description=data.description,
        sort_order=data.sort_order,
    )


def _org_detail_response(data: OrgIntentDetail) -> SomOrganizationalIntentDetailResponse:
    return SomOrganizationalIntentDetailResponse(
        intent_type=data.intent_type,  # type: ignore[arg-type]
        display_name=data.display_name,
        status=data.status,  # type: ignore[arg-type]
        is_timeboxed=data.is_timeboxed,
        is_guardrail=data.is_guardrail,
        guardrail_warning=data.guardrail_warning,
        last_synced_at=data.last_synced_at,
        expiry_date=data.expiry_date,
        fields=data.fields,
        field_labels=data.field_labels,
        custom_metrics=[_org_metric_item(m) for m in data.custom_metrics],
    )


def _org_save_response(data: OrgIntentSaveResult) -> SomOrganizationalIntentSaveResponse:
    return SomOrganizationalIntentSaveResponse(
        intent_type=data.intent_type,  # type: ignore[arg-type]
        display_name=data.display_name,
        status=data.status,  # type: ignore[arg-type]
        is_timeboxed=data.is_timeboxed,
        is_guardrail=data.is_guardrail,
        last_synced_at=data.last_synced_at,
        field_preview=data.field_preview,
        all_configured=data.all_configured,
        success_code=data.success_code,
    )


def _org_bulk_save_response(
    data: OrgIntentBulkSaveResult,
) -> SomOrganizationalIntentBulkSaveResponse:
    return SomOrganizationalIntentBulkSaveResponse(
        items=[_org_save_response(item) for item in data.items],
        all_configured=data.all_configured,
        configured_count=data.configured_count,
        total_count=data.total_count,
        success_code=data.success_code,
    )


def _org_validation_http_exception(exc: SomFieldValidationError) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail=SomValidationErrorDetail(
            code=exc.code,
            field=exc.field,
            message=exc.message,
            intent_type=exc.intent_type,
        ).model_dump(by_alias=True),
    )


def _org_delete_response(data: OrgIntentDeleteResult) -> SomOrganizationalIntentDeleteResponse:
    return SomOrganizationalIntentDeleteResponse(
        intent_type=data.intent_type,  # type: ignore[arg-type]
        display_name=data.display_name,
        status=data.status,  # type: ignore[arg-type]
        deleted_at=data.deleted_at,
        configured_count=data.configured_count,
        total_count=data.total_count,
    )


def _org_field_delete_response(
    data: OrgIntentFieldDeleteResult,
) -> SomOrganizationalIntentFieldDeleteResponse:
    return SomOrganizationalIntentFieldDeleteResponse(
        intent_type=data.intent_type,  # type: ignore[arg-type]
        display_name=data.display_name,
        status=data.status,  # type: ignore[arg-type]
        deleted_field=data.deleted_field,
        is_timeboxed=data.is_timeboxed,
        is_guardrail=data.is_guardrail,
        last_synced_at=data.last_synced_at,
        fields=data.fields,
        custom_metrics=[_org_metric_item(m) for m in data.custom_metrics],
        field_preview=data.field_preview,
        all_configured=data.all_configured,
        configured_count=data.configured_count,
        total_count=data.total_count,
        success_code=data.success_code,
    )


def _timeline_detail_response(data: TimelineCardDetail) -> SomTimelineCardDetailResponse:
    return SomTimelineCardDetailResponse(
        card_type=data.card_type,  # type: ignore[arg-type]
        display_name=data.display_name,
        status=data.status,  # type: ignore[arg-type]
        last_synced_at=data.last_synced_at,
        fields=data.fields,
        field_labels=data.field_labels,
        defaults=data.defaults,
        required_fields=data.required_fields,
    )


def _timeline_save_response(data: TimelineSaveResult) -> SomTimelineCardSaveResponse:
    return SomTimelineCardSaveResponse(
        card_type=data.card_type,  # type: ignore[arg-type]
        display_name=data.display_name,
        status=data.status,  # type: ignore[arg-type]
        last_synced_at=data.last_synced_at,
        all_configured=data.all_configured,
        success_code=data.success_code,
    )


def _status_badge(status: str) -> str:
    return "ACTIVE" if status == "CONFIGURED" else status


@router.get(
    "/interview-setup",
    response_model=SomInterviewSetupResponse,
    summary="Interview-First Setup form for a leadership role tab",
)
def get_interview_setup(
    role: str = Query(
        default="national_manager",
        description="national_manager | regional_manager | seller_manager",
    ),
    db: Session = Depends(get_db),
) -> SomInterviewSetupResponse:
    try:
        validate_role(role)
        return _to_setup_response(build_interview_setup(db, role))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@router.put(
    "/interview-setup/{role}/draft",
    response_model=SomInterviewSetupResponse,
    summary="Persist draft responses while admin types (optional autosave)",
)
def put_interview_draft(
    role: str,
    body: SomInterviewDraftRequest,
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    db: Session = Depends(get_db),
) -> SomInterviewSetupResponse:
    try:
        data = save_draft_responses(
            db,
            role,
            _responses_payload(body),
            captured_by=x_user_id,
        )
        return _to_setup_response(data)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@router.post(
    "/interview-setup/{role}/save",
    response_model=SomInterviewSaveResponse,
    summary="Commit responses from Verify & Edit review panel",
)
def post_interview_save(
    role: str,
    body: SomInterviewSaveRequest,
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    db: Session = Depends(get_db),
) -> SomInterviewSaveResponse:
    try:
        data = save_interview_responses(
            db,
            role,
            _responses_payload(body),
            saved_by=x_user_id,
        )
        return _to_save_response(data)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except RuntimeError as exc:
        if str(exc) == ERR_MSG_0021:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=ERR_MSG_0021,
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@router.get(
    "/interview-intent-cards",
    response_model=SomIntentCardsResponse,
    summary="Interview-First Setup intent card status per leadership role (US 3.2.1)",
)
def get_interview_intent_cards(db: Session = Depends(get_db)) -> SomIntentCardsResponse:
    try:
        return _interview_intent_cards_response(db)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@router.get(
    "/intent-cards",
    response_model=SomIntentCardsResponse,
    summary="Deprecated alias — use /interview-intent-cards",
    deprecated=True,
)
def get_intent_cards_deprecated(db: Session = Depends(get_db)) -> SomIntentCardsResponse:
    return get_interview_intent_cards(db)


@router.get(
    "/organizational-intent-cards",
    response_model=SomOrganizationalIntentListResponse,
    summary="Organizational intent cards list (US 3.2.2)",
)
def get_organizational_intent_cards(
    db: Session = Depends(get_db),
) -> SomOrganizationalIntentListResponse:
    try:
        items = list_organizational_intents(db)
        return SomOrganizationalIntentListResponse(
            items=[
                SomOrganizationalIntentSummary(
                    intent_type=i.intent_type,  # type: ignore[arg-type]
                    display_name=i.display_name,
                    status=i.status,  # type: ignore[arg-type]
                    is_timeboxed=i.is_timeboxed,
                    is_guardrail=i.is_guardrail,
                    last_synced_at=i.last_synced_at,
                    field_preview=i.field_preview,
                )
                for i in items
            ]
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@router.put(
    "/organizational-intent-cards",
    response_model=SomOrganizationalIntentBulkSaveResponse,
    summary="Bulk save organizational intent cards (atomic)",
)
def put_organizational_intent_cards_bulk(
    body: SomOrganizationalIntentBulkSaveRequest,
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    db: Session = Depends(get_db),
) -> SomOrganizationalIntentBulkSaveResponse:
    """Save one or more intent cards in a single transaction.

    One Context Lake rebuild and one commit — use when the FE Save button
    persists multiple cards at once.
    """
    try:
        cards_payload = {
            intent_type: {"fields": dict(card.fields)}
            for intent_type, card in body.cards.items()
        }
        result = save_organizational_intents_bulk(
            db,
            cards_payload,
            saved_by=x_user_id,
        )
        return _org_bulk_save_response(result)
    except SomFieldValidationError as exc:
        raise _org_validation_http_exception(exc) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except RuntimeError as exc:
        if str(exc) == ERR_MSG_0024:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=ERR_MSG_0024,
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@router.get(
    "/organizational-intent-cards/{intent_type}",
    response_model=SomOrganizationalIntentDetailResponse,
    summary="Organizational intent card detail for edit panel",
)
def get_organizational_intent_card(
    intent_type: str,
    db: Session = Depends(get_db),
) -> SomOrganizationalIntentDetailResponse:
    try:
        validate_intent_type(intent_type)
        return _org_detail_response(get_organizational_intent(db, intent_type))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except LookupError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Intent card not found.") from None
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@router.put(
    "/organizational-intent-cards/{intent_type}",
    response_model=SomOrganizationalIntentSaveResponse,
    summary="Save organizational intent card configuration",
)
def put_organizational_intent_card(
    intent_type: str,
    body: SomOrganizationalIntentSaveRequest,
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    db: Session = Depends(get_db),
) -> SomOrganizationalIntentSaveResponse:
    try:
        result = save_organizational_intent(
            db,
            intent_type,
            dict(body.fields),
            saved_by=x_user_id,
        )
        return _org_save_response(result)
    except SomFieldValidationError as exc:
        raise _org_validation_http_exception(exc) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except LookupError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Intent card not found.") from None
    except RuntimeError as exc:
        if str(exc) == ERR_MSG_0024:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=ERR_MSG_0024,
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@router.delete(
    "/organizational-intent-cards/{intent_type}/fields/{field_key}",
    response_model=SomOrganizationalIntentFieldDeleteResponse,
    summary="Delete one field row inside an organizational intent card",
)
def delete_organizational_intent_card_field(
    intent_type: str,
    field_key: str,
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    db: Session = Depends(get_db),
) -> SomOrganizationalIntentFieldDeleteResponse:
    """Remove a single metric row (e.g. revenueAndQuality or additionalContext).

    Deleting a required metric leaves remaining values in place but sets the
    card to ``NOT_CONFIGURED`` until all required fields are saved again.
    Partial cards (already ``NOT_CONFIGURED`` with leftover values) are supported.
    """
    try:
        result = delete_organizational_intent_field(
            db,
            intent_type,
            field_key,
            deleted_by=x_user_id,
        )
        return _org_field_delete_response(result)
    except SomDeleteConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except SomFieldNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=SomFieldNotFoundDetail(
                intent_type=exc.intent_type,
                field=exc.field_key,
                message=str(exc),
                available_fields=exc.available_fields,
            ).model_dump(by_alias=True),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except LookupError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Intent card not found."
        ) from None
    except RuntimeError as exc:
        if str(exc) == ERR_MSG_0024:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=ERR_MSG_0024,
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@router.post(
    "/organizational-intent-cards/{intent_type}/metrics",
    response_model=SomOrganizationalIntentCustomMetricItem,
    status_code=status.HTTP_201_CREATED,
    summary="Add a custom metric row (label + description)",
)
def post_organizational_intent_metric(
    intent_type: str,
    body: SomOrganizationalIntentMetricCreateRequest,
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    db: Session = Depends(get_db),
) -> SomOrganizationalIntentCustomMetricItem:
    """Add a user-defined metric, like adding a new interview question."""
    try:
        metric = add_organizational_intent_metric(
            db,
            intent_type,
            label=body.label,
            description=body.description,
            sort_order=body.sort_order,
            saved_by=x_user_id,
        )
        return _org_metric_item(metric)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except LookupError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Intent card not found."
        ) from None
    except RuntimeError as exc:
        if str(exc) == ERR_MSG_0024:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=ERR_MSG_0024,
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@router.patch(
    "/organizational-intent-cards/{intent_type}/metrics/{metric_id}",
    response_model=SomOrganizationalIntentCustomMetricItem,
    summary="Update custom metric label, description, or order",
)
def patch_organizational_intent_metric(
    intent_type: str,
    metric_id: str,
    body: SomOrganizationalIntentMetricUpdateRequest,
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    db: Session = Depends(get_db),
) -> SomOrganizationalIntentCustomMetricItem:
    try:
        metric = update_organizational_intent_metric(
            db,
            intent_type,
            metric_id,
            label=body.label,
            description=body.description,
            sort_order=body.sort_order,
            saved_by=x_user_id,
        )
        return _org_metric_item(metric)
    except SomMetricNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Custom metric not found.",
        ) from None
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except LookupError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Intent card not found."
        ) from None
    except RuntimeError as exc:
        if str(exc) == ERR_MSG_0024:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=ERR_MSG_0024,
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@router.delete(
    "/organizational-intent-cards/{intent_type}/metrics/{metric_id}",
    response_model=SomOrganizationalIntentFieldDeleteResponse,
    summary="Delete a custom metric row",
)
def delete_organizational_intent_card_metric(
    intent_type: str,
    metric_id: str,
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    db: Session = Depends(get_db),
) -> SomOrganizationalIntentFieldDeleteResponse:
    try:
        result = delete_organizational_intent_metric(
            db,
            intent_type,
            metric_id,
            deleted_by=x_user_id,
        )
        return _org_field_delete_response(result)
    except SomMetricNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Custom metric not found.",
        ) from None
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except LookupError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Intent card not found."
        ) from None
    except RuntimeError as exc:
        if str(exc) == ERR_MSG_0024:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=ERR_MSG_0024,
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@router.delete(
    "/organizational-intent-cards/{intent_type}",
    response_model=SomOrganizationalIntentDeleteResponse,
    summary="Delete (clear) organizational intent card configuration",
)
def delete_organizational_intent_card(
    intent_type: str,
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    db: Session = Depends(get_db),
) -> SomOrganizationalIntentDeleteResponse:
    """Clears saved configuration after FE confirmation dialog (US 3.2.3)."""
    try:
        result = delete_organizational_intent(
            db,
            intent_type,
            deleted_by=x_user_id,
        )
        return _org_delete_response(result)
    except SomDeleteConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except LookupError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Intent card not found.") from None
    except RuntimeError as exc:
        if str(exc) == ERR_MSG_0024:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=ERR_MSG_0024,
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@router.get(
    "/configuration-status",
    response_model=SomConfigurationStatusResponse,
    summary="All organizational intent cards configured check",
)
def get_som_configuration_status(
    db: Session = Depends(get_db),
) -> SomConfigurationStatusResponse:
    try:
        status_data = get_configuration_status(db)
        return SomConfigurationStatusResponse(
            all_configured=status_data.all_configured,
            configured_count=status_data.configured_count,
            total_count=status_data.total_count,
            success_code=status_data.success_code,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@router.get(
    "/timeline-classification-section-status",
    response_model=SomTimelineSectionStatusResponse,
    summary="Timeline section gate and progress (US 3.4.1)",
)
def get_timeline_section_status_endpoint(
    db: Session = Depends(get_db),
) -> SomTimelineSectionStatusResponse:
    try:
        data = get_timeline_section_status(db)
        return SomTimelineSectionStatusResponse(
            section_unlocked=data.section_unlocked,
            organizational_intent_configured=data.organizational_intent_configured,
            timeline_configured_count=data.timeline_configured_count,
            timeline_total_count=data.timeline_total_count,
            all_timeline_configured=data.all_timeline_configured,
            success_code=data.success_code,
            message_code=data.message_code,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@router.get(
    "/timeline-classification-cards",
    response_model=SomTimelineCardListResponse,
    summary="Timeline classification cards grid (US 3.4.1)",
)
def get_timeline_classification_cards(
    db: Session = Depends(get_db),
) -> SomTimelineCardListResponse:
    try:
        items = list_timeline_cards(db)
        return SomTimelineCardListResponse(
            items=[
                SomTimelineCardSummary(
                    card_type=i.card_type,  # type: ignore[arg-type]
                    display_name=i.display_name,
                    status=i.status,  # type: ignore[arg-type]
                    status_badge=_status_badge(i.status),
                    last_synced_at=i.last_synced_at,
                    field_preview=i.field_preview,
                )
                for i in items
            ]
        )
    except SomSectionLockedError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=ERR_MSG_0030,
        ) from None
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@router.get(
    "/timeline-classification-cards/{card_type}",
    response_model=SomTimelineCardDetailResponse,
    summary="Timeline classification card detail for edit panel",
)
def get_timeline_classification_card(
    card_type: str,
    db: Session = Depends(get_db),
) -> SomTimelineCardDetailResponse:
    try:
        validate_card_type(card_type)
        return _timeline_detail_response(get_timeline_card(db, card_type))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except LookupError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Timeline card not found.") from None
    except SomSectionLockedError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=ERR_MSG_0030,
        ) from None
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@router.put(
    "/timeline-classification-cards/{card_type}",
    response_model=SomTimelineCardSaveResponse,
    summary="Save timeline classification card configuration",
    responses={
        428: {"description": "Agent impact confirmation required (INFO_MSG_0006)"},
    },
)
def put_timeline_classification_card(
    card_type: str,
    body: SomTimelineCardSaveRequest,
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    db: Session = Depends(get_db),
) -> SomTimelineCardSaveResponse:
    try:
        result = save_timeline_card(
            db,
            card_type,
            dict(body.fields),
            confirm_agent_impact=body.confirm_agent_impact,
            saved_by=x_user_id,
        )
        return _timeline_save_response(result)
    except SomAgentImpactRequired:
        raise HTTPException(
            status_code=status.HTTP_428_PRECONDITION_REQUIRED,
            detail=SomAgentImpactDetail(
                code=INFO_MSG_0006,
                message=INFO_MSG_0006_MESSAGE,
            ).model_dump(by_alias=True),
        ) from None
    except TimelineFieldValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=SomValidationErrorDetail(
                code=exc.code,
                field=exc.field,
                message=exc.message,
            ).model_dump(by_alias=True),
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except LookupError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Timeline card not found.") from None
    except SomSectionLockedError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=ERR_MSG_0030,
        ) from None
    except RuntimeError as exc:
        if str(exc) == ERR_MSG_0029:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=ERR_MSG_0029,
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@router.get(
    "/timeline-classification-configuration-status",
    response_model=SomTimelineConfigurationStatusResponse,
    summary="All timeline classification cards configured check",
)
def get_timeline_configuration_status_endpoint(
    db: Session = Depends(get_db),
) -> SomTimelineConfigurationStatusResponse:
    try:
        data = get_timeline_configuration_status(db)
        return SomTimelineConfigurationStatusResponse(
            all_configured=data.all_configured,
            configured_count=data.configured_count,
            total_count=data.total_count,
            success_code=data.success_code,
        )
    except SomSectionLockedError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=ERR_MSG_0030,
        ) from None
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@router.get(
    "/context-lake",
    response_model=SomContextLakeResponse,
    summary="Sales Operating Model Context Lake for AI agents",
)
def get_som_context_lake(db: Session = Depends(get_db)) -> SomContextLakeResponse:
    try:
        return _context_lake_response(get_context_lake(db))
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@router.get(
    "/interview-questions",
    response_model=list[SomInterviewQuestionAdminItem],
    summary="List active interview questions (admin CRUD)",
)
def get_interview_questions(
    role: str | None = Query(default=None),
    db: Session = Depends(get_db),
) -> list[SomInterviewQuestionAdminItem]:
    try:
        if role:
            validate_role(role)
        rows = list_interview_questions(db, role=role)
        return [
            SomInterviewQuestionAdminItem(
                question_id=r.question_id,
                sort_order=r.sort_order,
                text=r.text,
                role=r.role or "national_manager",  # type: ignore[arg-type]
            )
            for r in rows
        ]
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@router.post(
    "/interview-questions",
    response_model=SomInterviewQuestionAdminItem,
    status_code=status.HTTP_201_CREATED,
    summary="Add an interview question",
)
def post_interview_question(
    body: SomInterviewQuestionCreateRequest,
    db: Session = Depends(get_db),
) -> SomInterviewQuestionAdminItem:
    try:
        row = create_interview_question(
            db,
            role=body.role,
            question_text=body.question_text,
            sort_order=body.sort_order,
        )
        return SomInterviewQuestionAdminItem(
            question_id=row.question_id,
            sort_order=row.sort_order,
            text=row.text,
            role=body.role,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@router.patch(
    "/interview-questions/{question_id}",
    response_model=SomInterviewQuestionAdminItem,
    summary="Update interview question text or order",
)
def patch_interview_question(
    question_id: str,
    body: SomInterviewQuestionUpdateRequest,
    db: Session = Depends(get_db),
) -> SomInterviewQuestionAdminItem:
    try:
        row = update_interview_question(
            db,
            question_id,
            question_text=body.question_text,
            sort_order=body.sort_order,
        )
        return SomInterviewQuestionAdminItem(
            question_id=row.question_id,
            sort_order=row.sort_order,
            text=row.text,
            role=row.role or "national_manager",  # type: ignore[arg-type]
        )
    except LookupError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Interview question not found.") from None
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc


@router.delete(
    "/interview-questions/{question_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Soft-delete an interview question",
)
def remove_interview_question(
    question_id: str,
    db: Session = Depends(get_db),
) -> Response:
    try:
        delete_interview_question(db, question_id)
    except LookupError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Interview question not found.") from None
    return Response(status_code=status.HTTP_204_NO_CONTENT)
