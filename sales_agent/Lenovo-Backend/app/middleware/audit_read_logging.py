"""Optional GET request audit logging driven by ``lvo_audit_config``."""

from __future__ import annotations

import logging

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.database import SessionLocal
from app.services.audit_log import (
    get_audit_config,
    infer_actor_type_from_path,
    should_log_read,
    write_audit_event,
)

logger = logging.getLogger(__name__)


class AuditReadLoggingMiddleware(BaseHTTPMiddleware):
    """Log configured read actions after the response is sent."""

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)

        if request.method != "GET":
            return response

        path = request.url.path
        actor_header = request.headers.get("X-Actor-Type")
        actor_type = infer_actor_type_from_path(path, actor_header)
        user_id = request.headers.get("X-User-Id")
        correlation_id = request.headers.get("X-Correlation-Id")

        db = SessionLocal()
        try:
            config = get_audit_config(db)
            if not should_log_read(config, actor_type=actor_type, path=path):
                return response

            outcome = "success" if response.status_code < 400 else "failure"
            write_audit_event(
                db,
                entity_type="api_request",
                entity_id=path,
                action="read",
                category="read_action",
                actor_type=actor_type,
                changed_by=user_id,
                outcome=outcome,
                correlation_id=correlation_id,
                diff={
                    "method": request.method,
                    "path": path,
                    "query": str(request.url.query),
                    "statusCode": response.status_code,
                },
                failure_reason=(
                    None if response.status_code < 400 else f"HTTP {response.status_code}"
                ),
            )
            db.commit()
        except Exception:
            logger.exception("Failed to write read-action audit for %s", path)
            db.rollback()
        finally:
            db.close()

        return response
