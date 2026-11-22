
import os
import secrets
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, HTTPException
from fastapi.security import APIKeyHeader

# Load local configuration without overriding cloud variables.
load_dotenv(
    Path(__file__).resolve().parents[1] / ".env",
    override=False,
)

key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
)


def require_api_key(
    provided: str | None = Depends(key_header),
):
    expected = os.getenv("CHURNGUARD_API_KEY")

    if not expected:
        raise HTTPException(
            status_code=503,
            detail="API key not configured",
        )

    if not provided or not secrets.compare_digest(
        provided, expected
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
        )
