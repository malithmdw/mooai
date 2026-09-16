"""Constrained Python Analysis Tool for enterprise document data.

Analyzes structured data extracted from retrieved documents using a
fixed set of safe, predefined operations.  No subprocess calls, no
filesystem I/O, no dynamic code execution — every operation is a
regular Python function that operates on in-memory list-of-dict data.

RBAC: ANALYST and ADMINISTRATOR only (Permission.ANALYTICS).
Timeout: CPU-bound work runs in a thread via ``asyncio.to_thread``
         wrapped by ``asyncio.wait_for``.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.core.logging import get_logger, log_debug, log_info, log_warning
from src.models.enums import Role
from src.security.authorization import AuthorizationPolicy, Permission

_logger = get_logger(__name__)

ANALYSIS_TIMEOUT_SECONDS: float = 30.0
MAX_RECORDS: int = 10_000
MAX_FIELDS_PER_RECORD: int = 50

_policy = AuthorizationPolicy()


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class AnalysisOperation(StrEnum):
    COUNT_BY = "count_by"
    GROUP_BY = "group_by"
    CALCULATE_PERCENTAGE = "calculate_percentage"
    COMPARE_PERIODS = "compare_periods"
    DETECT_PATTERNS = "detect_patterns"


class AnalysisRequest(BaseModel):
    """Structured input for one analysis operation."""

    model_config = ConfigDict(frozen=True)

    operation: AnalysisOperation
    data: list[dict[str, Any]] = Field(..., min_length=1, max_length=MAX_RECORDS)
    parameters: dict[str, Any]

    @field_validator("data")
    @classmethod
    def _validate_records(cls, data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for i, record in enumerate(data):
            if len(record) > MAX_FIELDS_PER_RECORD:
                raise ValueError(
                    f"Record {i} has {len(record)} fields (max {MAX_FIELDS_PER_RECORD})"
                )
            for key in record:
                if not isinstance(key, str) or not key:
                    raise ValueError(f"Record {i}: field names must be non-empty strings")
        return data


class AnalysisResult(BaseModel):
    """Output of a completed analysis operation."""

    model_config = ConfigDict(frozen=True)

    operation: AnalysisOperation
    record_count: int
    result: dict[str, Any]


class AnalysisError(Exception):
    """Raised when parameters are invalid for the requested operation."""


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


def _validate_parameters(operation: AnalysisOperation, parameters: dict[str, Any]) -> None:
    """Validate that parameters contain required keys for the given operation.

    Raises ``AnalysisError`` with a descriptive message on the first violation.
    Called before execution so bad input never reaches the compute layer.
    """
    if operation in (
        AnalysisOperation.COUNT_BY,
        AnalysisOperation.GROUP_BY,
        AnalysisOperation.CALCULATE_PERCENTAGE,
    ):
        if not isinstance(parameters.get("field"), str) or not parameters["field"]:
            raise AnalysisError(
                f"Operation '{operation}' requires a non-empty 'field' string parameter."
            )
        return

    if operation == AnalysisOperation.COMPARE_PERIODS:
        for key in ("group_field", "period_field", "period_a", "period_b"):
            if not isinstance(parameters.get(key), str) or not parameters[key]:
                raise AnalysisError(
                    f"Operation '{operation}' requires a non-empty '{key}' string parameter."
                )
        return

    if operation == AnalysisOperation.DETECT_PATTERNS:
        fields = parameters.get("fields")
        if not isinstance(fields, list):
            raise AnalysisError(
                f"Operation '{operation}' requires a 'fields' list parameter."
            )
        if not fields:
            raise AnalysisError(
                f"Operation '{operation}': 'fields' list must not be empty."
            )
        if not all(isinstance(f, str) and f for f in fields):
            raise AnalysisError(
                f"Operation '{operation}': all entries in 'fields' must be non-empty strings."
            )
        top_n = parameters.get("top_n", 10)
        if not isinstance(top_n, int) or top_n < 1 or top_n > 100:
            raise AnalysisError("Parameter 'top_n' must be an integer between 1 and 100.")


# ---------------------------------------------------------------------------
# Operation implementations — pure Python, no I/O, no subprocess
# ---------------------------------------------------------------------------


def _field_value(record: dict[str, Any], field: str) -> str:
    """Return a field value as a string; absent fields become ``"(missing)"``."""
    value = record.get(field)
    return "(missing)" if value is None else str(value)


def _count_by(data: list[dict[str, Any]], parameters: dict[str, Any]) -> dict[str, Any]:
    field = parameters["field"]
    counts: Counter[str] = Counter(_field_value(r, field) for r in data)
    return {"field": field, "counts": dict(counts.most_common())}


def _group_by(data: list[dict[str, Any]], parameters: dict[str, Any]) -> dict[str, Any]:
    field = parameters["field"]
    max_per_group = min(int(parameters.get("max_per_group", 10)), 50)
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in data:
        key = _field_value(record, field)
        bucket = groups.setdefault(key, [])
        if len(bucket) < max_per_group:
            bucket.append(record)
    return {"field": field, "groups": dict(sorted(groups.items()))}


def _calculate_percentage(
    data: list[dict[str, Any]], parameters: dict[str, Any]
) -> dict[str, Any]:
    field = parameters["field"]
    total = len(data)
    counts: Counter[str] = Counter(_field_value(r, field) for r in data)
    percentages = {
        value: round(count / total * 100, 2) for value, count in counts.most_common()
    }
    return {"field": field, "total": total, "percentages": percentages}


def _compare_periods(
    data: list[dict[str, Any]], parameters: dict[str, Any]
) -> dict[str, Any]:
    group_field = parameters["group_field"]
    period_field = parameters["period_field"]
    period_a = parameters["period_a"]
    period_b = parameters["period_b"]

    a_data = [r for r in data if _field_value(r, period_field) == period_a]
    b_data = [r for r in data if _field_value(r, period_field) == period_b]

    a_counts: Counter[str] = Counter(_field_value(r, group_field) for r in a_data)
    b_counts: Counter[str] = Counter(_field_value(r, group_field) for r in b_data)

    all_values = sorted(set(a_counts) | set(b_counts))
    delta = {v: b_counts.get(v, 0) - a_counts.get(v, 0) for v in all_values}

    return {
        "group_field": group_field,
        "period_field": period_field,
        period_a: dict(a_counts),
        period_b: dict(b_counts),
        "delta": delta,
        "record_counts": {period_a: len(a_data), period_b: len(b_data)},
    }


def _detect_patterns(
    data: list[dict[str, Any]], parameters: dict[str, Any]
) -> dict[str, Any]:
    fields: list[str] = parameters["fields"]
    top_n = int(parameters.get("top_n", 10))

    combos: Counter[tuple[str, ...]] = Counter(
        tuple(_field_value(r, f) for f in fields) for r in data
    )
    patterns = [
        {"pattern": dict(zip(fields, combo)), "count": count}
        for combo, count in combos.most_common(top_n)
    ]
    return {"fields": fields, "patterns": patterns}


_HANDLERS = {
    AnalysisOperation.COUNT_BY: _count_by,
    AnalysisOperation.GROUP_BY: _group_by,
    AnalysisOperation.CALCULATE_PERCENTAGE: _calculate_percentage,
    AnalysisOperation.COMPARE_PERIODS: _compare_periods,
    AnalysisOperation.DETECT_PATTERNS: _detect_patterns,
}


def _run_analysis(request: AnalysisRequest) -> AnalysisResult:
    """Execute the operation synchronously; meant to run inside ``asyncio.to_thread``."""
    result = _HANDLERS[request.operation](list(request.data), dict(request.parameters))
    return AnalysisResult(
        operation=request.operation,
        record_count=len(request.data),
        result=result,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def run_analysis(
    request: AnalysisRequest,
    *,
    role: Role,
    timeout: float = ANALYSIS_TIMEOUT_SECONDS,
) -> AnalysisResult:
    """Run a constrained analysis operation with RBAC enforcement and timeout.

    Parameters
    ----------
    request:
        The validated analysis request.
    role:
        The caller's RBAC role — checked before any computation.
    timeout:
        Maximum seconds to wait for the operation (default: 30 s).

    Raises
    ------
    PermissionDeniedError
        If ``role`` lacks ``Permission.ANALYTICS``.
    AnalysisError
        If ``request.parameters`` are invalid for the operation.
    TimeoutError
        If execution exceeds ``timeout`` seconds.
    """
    _policy.authorize(role, Permission.ANALYTICS)
    _validate_parameters(request.operation, dict(request.parameters))

    log_info(
        _logger,
        "analysis.start",
        "Starting analysis",
        operation=str(request.operation),
        record_count=len(request.data),
        role=str(role),
    )

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(_run_analysis, request),
            timeout=timeout,
        )
    except TimeoutError:
        log_warning(
            _logger,
            "analysis.timeout",
            "Analysis timed out",
            operation=str(request.operation),
            timeout_seconds=timeout,
        )
        raise

    log_debug(
        _logger,
        "analysis.complete",
        "Analysis completed",
        operation=str(result.operation),
        record_count=result.record_count,
    )

    return result
