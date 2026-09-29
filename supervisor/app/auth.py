import secrets

from fastapi import HTTPException, Request

from .config import get_settings


def require_key(request: Request) -> None:
    """Bearer token (OpenAI style), X-API-Key, or ?key= (EventSource can't set headers)."""
    expected = get_settings().api_key
    auth = request.headers.get("authorization", "")
    supplied = auth[7:] if auth.lower().startswith("bearer ") else None
    supplied = supplied or request.headers.get("x-api-key") or request.query_params.get("key")
    if not supplied or not secrets.compare_digest(supplied.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="invalid or missing API key")
