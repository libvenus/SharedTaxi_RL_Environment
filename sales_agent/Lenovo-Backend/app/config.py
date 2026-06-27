from functools import lru_cache
from pathlib import Path
from urllib.parse import quote_plus

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent


class Settings(BaseSettings):
    """Runtime configuration loaded from environment / .env.

    `.env` is searched in two locations (project root first, then `app/`)
    so the server starts whether you place it next to `requirements.txt`
    or next to `main.py`.

    Two ways to configure the database — pick one:

    1. Discrete components (recommended for managed Postgres instances):
       DATABASE_HOST, DATABASE_PORT, DATABASE_NAME, DATABASE_USERNAME,
       DATABASE_PASSWORD. The full SQLAlchemy URL is assembled from them.

    2. A single full URL via DATABASE_URL (e.g. for local development).

    If both are set, the discrete components win.
    """

    model_config = SettingsConfigDict(
        env_file=(
            str(_PROJECT_ROOT / ".env"),
            str(_HERE / ".env"),
        ),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_host: str | None = Field(default=None, alias="DATABASE_HOST")
    database_port: int | None = Field(default=None, alias="DATABASE_PORT")
    database_name: str | None = Field(default=None, alias="DATABASE_NAME")
    database_username: str | None = Field(default=None, alias="DATABASE_USERNAME")
    database_password: str | None = Field(default=None, alias="DATABASE_PASSWORD")
    database_sslmode: str | None = Field(
        default=None,
        alias="DATABASE_SSLMODE",
        description=(
            "Optional Postgres sslmode (e.g. 'require' for managed providers "
            "like RDS / Supabase / Azure). Omit for local instances."
        ),
    )

    database_url: str = Field(
        default="postgresql+psycopg2://postgres:postgres@localhost:5432/lenovo_nitro",
        alias="DATABASE_URL",
    )

    cors_origins: str = Field(
        default="http://localhost:3000,http://localhost:5173",
        alias="CORS_ORIGINS",
    )

    fiscal_year_start_month: int = Field(
        default=4,
        alias="FISCAL_YEAR_START_MONTH",
        description=(
            "Month (1–12) when the fiscal year begins. Default 4 = April "
            "(Lenovo-style FY naming: FY2026 = Apr 2025 – Mar 2026)."
        ),
    )

    compliance_api_key: str | None = Field(
        default=None,
        alias="COMPLIANCE_API_KEY",
        description=(
            "Optional API key for /api/compliance/* ingest and query. "
            "When unset, endpoints are open (local dev only)."
        ),
    )

    audit_read_logging_enabled: bool = Field(
        default=True,
        alias="AUDIT_READ_LOGGING_ENABLED",
        description="Master switch for GET read-action audit middleware.",
    )

    @model_validator(mode="after")
    def _assemble_database_url(self) -> "Settings":
        """Build database_url from discrete vars when they are provided."""
        components = [
            self.database_host,
            self.database_name,
            self.database_username,
            self.database_password,
        ]
        if all(c is not None and c != "" for c in components):
            user = quote_plus(self.database_username or "")
            pwd = quote_plus(self.database_password or "")
            port = self.database_port or 5432
            url = (
                f"postgresql+psycopg2://{user}:{pwd}"
                f"@{self.database_host}:{port}/{self.database_name}"
            )
            if self.database_sslmode:
                url += f"?sslmode={self.database_sslmode}"
            self.database_url = url
        return self

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
