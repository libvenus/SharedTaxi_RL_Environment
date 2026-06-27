from __future__ import annotations

import logging

from fastapi import Depends, Header, HTTPException, status

from app.core.config import EMAIL_API_KEY

logger = logging.getLogger(__name__)

if not EMAIL_API_KEY:
    logger.warning(
        "EMAIL_API_KEY is unset; the email send endpoint will allow requests without an API key in development."
    )


def verify_email_api_key(
    x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> None:
    if not EMAIL_API_KEY:
        return None

    if x_api_key != EMAIL_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )

    return None
