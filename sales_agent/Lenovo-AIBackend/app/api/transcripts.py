"""FastAPI router for the meeting-transcript pipeline.

Sprint 1A · US02 — Consent & Recording

Five routes mounted at /transcripts:

    POST   /transcripts/                        # start (consent required)
    POST   /transcripts/{meeting_id}/segments   # append (batched)
    POST   /transcripts/{meeting_id}/finalize   # clean end
    POST   /transcripts/{meeting_id}/terminate  # organizer-removed / partial
    GET    /transcripts/{meeting_id}            # FE / read

The bot calls all four POST routes; the GET is for the FE Activity tab
later. Casing convention is snake_case throughout.
"""

from uuid import UUID

from fastapi import APIRouter, Body, Depends, Path, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schema.transcript import (
    TranscriptFinalizeRequest,
    TranscriptResponse,
    TranscriptSegmentResponse,
    TranscriptSegmentsAppendRequest,
    TranscriptSegmentsAppendResponse,
    TranscriptStartRequest,
    TranscriptTerminateRequest,
    TranscriptWithSegmentsResponse,
)
from app.services.transcript_service import (
    append_segments,
    finalize_transcript,
    get_transcript,
    start_transcript,
    terminate_transcript,
)

router = APIRouter(
    prefix="/transcripts",
    tags=["Meeting Transcripts"],
)


# ---------------------------------------------------------------------------
# POST /transcripts/   — start the transcript (after bot sends consent msg)
# ---------------------------------------------------------------------------


@router.post(
    "/",
    response_model=TranscriptResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Start a transcript (called after the bot sends CONF_MSG_0004 in chat)",
    responses={
        400: {"description": "Bot is not yet in/approaching the meeting (bot_status check)"},
        404: {"description": "Meeting not found"},
        409: {"description": "A transcript already exists for this meeting"},
        422: {
            "description": (
                "Schema validation failed (missing consent fields, "
                "started_at < consent_sent_at, etc.)"
            )
        },
    },
)
def start_transcript_route(
    payload: TranscriptStartRequest = Body(...),
    db: Session = Depends(get_db),
) -> TranscriptResponse:
    """Persist the consent proof and open the transcript for segment writes."""
    transcript = start_transcript(
        db=db,
        meeting_id=payload.meeting_id,
        consent_message_text=payload.consent_message_text,
        consent_sent_at=payload.consent_sent_at,
        started_at=payload.started_at,
    )
    return TranscriptResponse.model_validate(transcript)


# ---------------------------------------------------------------------------
# POST /transcripts/{meeting_id}/segments   — append batched utterances
# ---------------------------------------------------------------------------


@router.post(
    "/{meeting_id}/segments",
    response_model=TranscriptSegmentsAppendResponse,
    summary="Append a batch of transcript segments (continuous during meeting)",
    responses={
        400: {"description": "Transcript is not in 'in_progress' state"},
        404: {"description": "No transcript exists for this meeting"},
        422: {"description": "Segment validation failed (confidence range, time order, etc.)"},
    },
)
def append_segments_route(
    meeting_id: UUID = Path(..., description="The meeting whose transcript is being written."),
    payload: TranscriptSegmentsAppendRequest = Body(...),
    db: Session = Depends(get_db),
) -> TranscriptSegmentsAppendResponse:
    """Bulk-INSERT a batch of segments. The bot may call this many times per meeting."""
    transcript, appended = append_segments(
        db=db,
        meeting_id=meeting_id,
        segments=payload.segments,
    )
    return TranscriptSegmentsAppendResponse(
        transcript_id=transcript.transcript_id,
        meeting_id=meeting_id,
        appended_count=appended,
        segment_count=transcript.segment_count,
    )


# ---------------------------------------------------------------------------
# POST /transcripts/{meeting_id}/finalize   — clean meeting end
# ---------------------------------------------------------------------------


@router.post(
    "/{meeting_id}/finalize",
    response_model=TranscriptResponse,
    summary="Finalise a transcript (clean meeting end with overall confidence)",
    responses={
        400: {"description": "Transcript already finalised / terminated"},
        404: {"description": "No transcript exists for this meeting"},
        422: {"description": "overall_confidence_score out of [0, 1]"},
    },
)
def finalize_transcript_route(
    meeting_id: UUID = Path(..., description="The meeting whose transcript is finalising."),
    payload: TranscriptFinalizeRequest = Body(...),
    db: Session = Depends(get_db),
) -> TranscriptResponse:
    """Mark the transcript as 'finalized' with terminated_reason='meeting_ended'."""
    transcript = finalize_transcript(
        db=db,
        meeting_id=meeting_id,
        overall_confidence_score=payload.overall_confidence_score,
        finalized_at=payload.finalized_at,
    )
    return TranscriptResponse.model_validate(transcript)


# ---------------------------------------------------------------------------
# POST /transcripts/{meeting_id}/terminate   — early stop
# ---------------------------------------------------------------------------


@router.post(
    "/{meeting_id}/terminate",
    response_model=TranscriptResponse,
    summary="Terminate a transcript early (organiser removed / all-left / bot failure)",
    responses={
        400: {"description": "Transcript already finalised / terminated"},
        404: {"description": "No transcript exists for this meeting"},
        422: {"description": "Reason not in {organizer_removed, all_left, bot_failure}"},
    },
)
def terminate_transcript_route(
    meeting_id: UUID = Path(..., description="The meeting whose transcript is being cut short."),
    payload: TranscriptTerminateRequest = Body(...),
    db: Session = Depends(get_db),
) -> TranscriptResponse:
    """Mark the transcript as 'terminated_partial' with the supplied reason."""
    transcript = terminate_transcript(
        db=db,
        meeting_id=meeting_id,
        reason=payload.reason,
        terminated_at=payload.terminated_at,
        overall_confidence_score=payload.overall_confidence_score,
    )
    return TranscriptResponse.model_validate(transcript)


# ---------------------------------------------------------------------------
# GET /transcripts/{meeting_id}   — read (FE Activity tab eventually)
# ---------------------------------------------------------------------------


@router.get(
    "/{meeting_id}",
    response_model=TranscriptWithSegmentsResponse,
    summary="Fetch a transcript and all its segments (ordered by start_time)",
    responses={
        404: {"description": "No transcript exists for this meeting"},
    },
)
def get_transcript_route(
    meeting_id: UUID = Path(..., description="The meeting whose transcript is being read."),
    db: Session = Depends(get_db),
) -> TranscriptWithSegmentsResponse:
    """Return the transcript metadata + every segment, ordered by start_time."""
    transcript, segments = get_transcript(db=db, meeting_id=meeting_id)
    return TranscriptWithSegmentsResponse(
        transcript=TranscriptResponse.model_validate(transcript),
        segments=[
            TranscriptSegmentResponse.model_validate(seg) for seg in segments
        ],
    )
