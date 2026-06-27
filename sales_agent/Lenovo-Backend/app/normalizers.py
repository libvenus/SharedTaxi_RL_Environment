"""Vocabulary normalizers — convert DB enum strings to UI labels.

The Opportunities UI uses slightly different stage/sale-motion labels from
what the database stores. Centralising the mapping here keeps the API
layer consistent and easy to update.
"""

# Sale motion (UI pill on Screen 1)
SALE_MOTION_LABELS: dict[str, str] = {
    "Net-New": "Net new",
    "Expansion": "Expansion",
    "Renewal": "Renewal",
}

# Stage (UI pill on Screen 3)
# DB values seen in sample data: Qualify, Develop, Propose, Execute, Closed Won.
STAGE_LABELS: dict[str, str] = {
    "Qualify": "Qualification",
    "Develop": "Discovery",
    "Propose": "Proposal",
    "Execute": "Negotiation",
    "Closed Won": "Closed Won",
    "Closed Lost": "Closed Lost",
}


def normalise_sale_motion(raw: str | None) -> str | None:
    if raw is None:
        return None
    return SALE_MOTION_LABELS.get(raw, raw)


def normalise_stage(raw: str | None) -> str | None:
    if raw is None:
        return None
    return STAGE_LABELS.get(raw, raw)


_VALID_DEAL_PRIORITIES = frozenset({"High", "Medium", "Low"})


def normalise_deal_priority(raw: str | None) -> str | None:
    """Map ``lvo_priority`` to the API contract; blank or invalid → ``None``.

    Legacy rows may store an empty string instead of NULL — treat both as unset
    so ``OpportunityDetail.priority`` validation does not 500 on read.
    """
    if raw is None:
        return None
    trimmed = raw.strip()
    if trimmed in _VALID_DEAL_PRIORITIES:
        return trimmed
    return None


def normalise_risk_score(raw: int | None) -> int | None:
    """Map lvo_riskscore to API contract (1–5); treat 0/out-of-range as unset."""
    if raw is None or not 1 <= raw <= 5:
        return None
    return raw


def slugify(value: str | None) -> str:
    """Lower-case, replace non-alphanumeric runs with single hyphens (per char).

    Mirrors the slug used by `/api/filters/products` ids so that the
    same slug round-trips back through ``/api/opportunities?products=...``.

    "ThinkPad"                  -> "thinkpad"
    "DaaS Managed Device Bundle"-> "daas-managed-device-bundle"
    """
    if not value:
        return ""
    return "".join(c.lower() if c.isalnum() else "-" for c in value).strip("-")
