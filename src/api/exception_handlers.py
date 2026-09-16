"""Global exception handlers.

Every error response — validation failures, explicit `HTTPException`s, and
truly unexpected exceptions — is normalized to
`src.models.errors.ErrorResponse`, so API callers see one consistent error
shape regardless of what went wrong. Unexpected exceptions are logged with
full detail server-side (see CLAUDE.md "Logging requirements") but never
expose internal detail to the caller — only a generic message and the
request ID to quote when reporting the issue.
"""

from __future__ import annotations

from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.core.logging import get_logger, log_error, log_warning
from src.models.errors import ErrorResponse

logger = get_logger(__name__)

_ERROR_CODES_BY_STATUS: dict[int, str] = {
    status.HTTP_400_BAD_REQUEST: "BAD_REQUEST",
    status.HTTP_401_UNAUTHORIZED: "UNAUTHORIZED",
    status.HTTP_403_FORBIDDEN: "FORBIDDEN",
    status.HTTP_404_NOT_FOUND: "NOT_FOUND",
    status.HTTP_405_METHOD_NOT_ALLOWED: "METHOD_NOT_ALLOWED",
    status.HTTP_422_UNPROCESSABLE_CONTENT: "VALIDATION_ERROR",
    status.HTTP_501_NOT_IMPLEMENTED: "NOT_IMPLEMENTED",
}
_DEFAULT_ERROR_CODE = "HTTP_ERROR"


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def _error_response(
    status_code: int, error_code: str, message: str, request_id: str | None
) -> JSONResponse:
    body = ErrorResponse(error_code=error_code, message=message, request_id=request_id)
    return JSONResponse(status_code=status_code, content=body.model_dump(mode="json"))


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return a consistent `ErrorResponse` for Pydantic request-validation failures."""
    log_warning(
        logger,
        "http.validation_failed",
        "request validation failed",
        path=request.url.path,
        error_count=len(exc.errors()),
    )
    return _error_response(
        status.HTTP_422_UNPROCESSABLE_CONTENT,
        "VALIDATION_ERROR",
        "The request failed validation.",
        _request_id(request),
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Return a consistent `ErrorResponse` for explicit `HTTPException`s."""
    error_code = _ERROR_CODES_BY_STATUS.get(exc.status_code, _DEFAULT_ERROR_CODE)
    log_warning(
        logger,
        "http.error_response",
        "request handled with an error status",
        path=request.url.path,
        status_code=exc.status_code,
    )
    return _error_response(exc.status_code, error_code, str(exc.detail), _request_id(request))


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all for exceptions no other handler caught.

    Never exposes internal exception details to the caller (see CLAUDE.md
    "Security principles") — the full exception is logged server-side, and
    the client sees only a generic message plus the request ID.
    """
    log_error(
        logger,
        "http.unhandled_exception",
        "unhandled exception while processing request",
        error=exc,
        path=request.url.path,
    )
    return _error_response(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "INTERNAL_ERROR",
        "An unexpected error occurred.",
        _request_id(request),
    )
