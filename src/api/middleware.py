"""Request-scoped middleware: request ID assignment and structured request
logging.
"""

from __future__ import annotations

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from src.core.logging import bind_log_context, get_logger, log_error, log_info

logger = get_logger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assign a request ID to every request and log its start/end.

    The request ID is taken from an inbound `X-Request-ID` header if the
    caller supplied one (preserving a gateway/load balancer's ID),
    otherwise generated. It is exposed to route handlers/exception
    handlers via `request.state.request_id`, bound into
    `src.core.logging`'s context for the duration of the request — so
    every log line emitted anywhere while handling this request carries
    it, per CLAUDE.md "every log line that's part of a request should be
    attributable to a request id" — and echoed back on the response as
    `X-Request-ID`.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())
        request.state.request_id = request_id

        start = time.perf_counter()
        with bind_log_context(request_id=request_id):
            log_info(
                logger,
                "http.request_started",
                "request started",
                method=request.method,
                path=request.url.path,
            )
            try:
                response = await call_next(request)
            except Exception as exc:
                # Should be unreachable in practice — the app's global
                # `Exception` handler converts errors to a response before
                # they get here. If something still escapes, log it rather
                # than losing the signal entirely, then re-raise: a broad
                # catch at this boundary must never swallow silently.
                duration_ms = round((time.perf_counter() - start) * 1000, 2)
                log_error(
                    logger,
                    "http.request_failed",
                    "request raised an unhandled exception",
                    error=exc,
                    method=request.method,
                    path=request.url.path,
                    duration_ms=duration_ms,
                )
                raise

            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            log_info(
                logger,
                "http.request_completed",
                "request completed",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=duration_ms,
            )

        response.headers[REQUEST_ID_HEADER] = request_id
        return response
