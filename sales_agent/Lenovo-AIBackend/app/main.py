from fastapi import APIRouter, FastAPI

from app.db.database import (
    Base,
    engine
)
Base.metadata.create_all(
    bind=engine
)

app = FastAPI()

# from app.models.activityDetails import *
from app.models.schedulemeeting import *
from app.models.transcript import *
from app.models.consent_email import *
from app.models.data_task import *

from app.models.meeting_briefing import *  # noqa: F401, F403
from app.models.email import *  # noqa: F401, F403

from app.core.config import AIBACKEND_API_PREFIX

from app.api.meeting_details import (
    router as meeting_router
)
from app.api.post_meeting_preview import (
    router as post_meeting_preview_router
)
from app.api.summary_details import (
    router as summary_router
)

from app.api.meeting_prep import (
    router as meeting_prep
)
from app.api.to_do_list import (
    router as to_do_list
)
from app.api.transcripts import (
    router as transcripts_router
)
from app.api.consent_emails import (
    router as consent_emails_router,
    meeting_consent_status_router,
)
from app.api.data_tasks import (
    router as data_tasks_router
)
from app.api.outreach import (
    router as outreach_router)
from app.api.summary_details import (
    router as summary_details_router
)
from app.api.email_insight import (
    router as email_insight_router
)
from app.api.summary_details import (
    router as summary_details_router
)
from app.api.vexa_webhook import (
    router as vexa_webhook_router
)


# All HTTP APIs live under /ai-api so Nginx can proxy this service without
# colliding with D365 Sales (/api/...). Individual routers keep their own
# sub-prefix (e.g. /meeting-details, /transcripts).
api_router = APIRouter(prefix=AIBACKEND_API_PREFIX)

api_router.include_router(meeting_router)
api_router.include_router(to_do_list)
api_router.include_router(transcripts_router)
api_router.include_router(post_meeting_preview_router)
api_router.include_router(meeting_prep)

api_router.include_router(consent_emails_router)
api_router.include_router(meeting_consent_status_router)
api_router.include_router(data_tasks_router)
api_router.include_router(outreach_router)
api_router.include_router(summary_details_router)
api_router.include_router(email_insight_router)
api_router.include_router(summary_details_router)
api_router.include_router(vexa_webhook_router)

app.include_router(api_router)
