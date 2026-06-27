from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.schema.post_meeting_review import KeyPointRequest,UpdateKeyPointRequest,UpdateNextStepRequest,CrmUpdateEditRequest,UpdateCrmRequest,UpdateSummaryRequest
from app.services.post_meeting_review import get_post_meeting_preview,get_postmeeting_review_by_meeting_id,add_key_point,delete_key_point,update_key_point,delete_nextSteps,update_nextSteps,update_crm_update,complete_keypoints,complete_nextstep,update_summary
from uuid import UUID
router = APIRouter(prefix="/post-meeting-preview", tags=["Execue: Post-Meeting Preview"])


@router.get("/meeting")
def post_meeting_preview(
    seller_id: str,
    db: Session = Depends(get_db)
    
):
    return get_post_meeting_preview(db, seller_id)

@router.get("/{meeting_id}")
def fetch_keypoints_nextSteps_crmUpdates(
    meeting_id: str,
    seller_id: str,
    db: Session = Depends(get_db)
):
    return get_postmeeting_review_by_meeting_id(
        db=db,
        meeting_id=meeting_id,
        seller_id=seller_id
    )
    

@router.post("/{meeting_id}/key-points")
def create_key_point(
    meeting_id: str,
    payload: KeyPointRequest,
    db: Session = Depends(get_db)
):
    return add_key_point(
        db=db,
        meeting_id=meeting_id,
        payload=payload.model_dump()
    )

@router.delete("/{meeting_id}/key-points/{key_point_id}")
def remove_key_point(
    meeting_id: str,
    key_point_id: str,
    db: Session = Depends(get_db)
):
    return delete_key_point(
        db=db,
        meeting_id=meeting_id,
        key_point_id=key_point_id
    )

@router.put(
    "/{meeting_id}/key-points/{key_point_id}"
)
def edit_key_point(
    meeting_id: str,
    key_point_id: str,
    payload: UpdateKeyPointRequest,
    db: Session = Depends(get_db)
):
    return update_key_point(
        db=db,
        meeting_id=meeting_id,
        key_point_id=key_point_id,
        payload=payload.model_dump()
    )


@router.delete("/{meeting_id}/nextSteps/{nextSteps_id}")
def remove_nextSteps(
    meeting_id: str,
    nextSteps_id: str,
    db: Session = Depends(get_db)
):
    return delete_nextSteps(
        db=db,
        meeting_id=meeting_id,
        nextSteps_id=nextSteps_id
    )


@router.put(
    "/{meeting_id}/nextSteps/{nextSteps_id}"
)
def edit_nextSteps(
    meeting_id: str,
    nextSteps_id: str,
    payload: UpdateNextStepRequest,
    db: Session = Depends(get_db)
):
    return update_nextSteps(
        db=db,
        meeting_id=meeting_id,
        nextSteps_id=nextSteps_id,
        payload=payload.model_dump()
    )

# @router.put(
#     "/{meeting_id}/crm_updates/{seq_id}"
# )
# def edit_crm_update(
#     meeting_id: str,
#     seq_id: int,
#     payload: CrmUpdateEditRequest,
#     db: Session = Depends(get_db)
# ):
#     return update_crm_update(
#         db=db,
#         meeting_id=meeting_id,
#         seq_id=seq_id,
#         payload=payload
#     )


@router.patch("/{meeting_id}/nextstep/complete")
def complete_meeting_nextstep(
    meeting_id: str,
    db: Session = Depends(get_db)
):
    return complete_nextstep(
        db=db,
        meeting_id=meeting_id
    )

@router.patch("/{meeting_id}/keypoints/complete")
def complete_meeting_keypoints(
    meeting_id: str,
    db: Session = Depends(get_db)
):
    return complete_keypoints(
        db=db,
        meeting_id=meeting_id
    )

@router.put("/{meeting_id}/update_crm/{seq_id}")
def update_crm_updates(
    meeting_id: str,
    seq_id: int,
    payload: UpdateCrmRequest,
    db: Session = Depends(get_db)
):
    return update_crm_update(
        db=db,
        meeting_id=meeting_id,
        seq_id=seq_id,
        payload=payload
    )

@router.put("/summary/{meeting_id}")
def update_summary_notes(
    meeting_id: str,
    payload: UpdateSummaryRequest,
    db: Session = Depends(get_db)
):
    return update_summary(
        db=db,
        meeting_id=meeting_id,
        payload=payload
    )