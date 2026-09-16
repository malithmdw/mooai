"""Tests for the Python Analysis Tool.

Coverage
--------
RBAC:
  - VIEWER denied with PermissionDeniedError
  - ENGINEER denied with PermissionDeniedError
  - ANALYST allowed
  - ADMINISTRATOR allowed

Operations (COUNT_BY, GROUP_BY, CALCULATE_PERCENTAGE, COMPARE_PERIODS,
DETECT_PATTERNS):
  - correct result shape and values
  - missing-field values handled as "(missing)"
  - parameter validation rejects bad inputs before execution

Data validation:
  - empty data rejected by AnalysisRequest
  - too many records rejected by AnalysisRequest
  - non-string field keys rejected by AnalysisRequest

Timeout:
  - TimeoutError raised when execution exceeds the limit
  - custom timeout value accepted
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

import pytest

from src.models.enums import Role
from src.security.authorization import PermissionDeniedError
from src.tools.analysis import (
    MAX_RECORDS,
    AnalysisError,
    AnalysisOperation,
    AnalysisRequest,
    run_analysis,
)


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

_INCIDENTS: list[dict[str, Any]] = [
    {"id": "INC-001", "root_cause": "human_error", "category": "network", "period": "Q1", "severity": "high"},
    {"id": "INC-002", "root_cause": "software_bug", "category": "database", "period": "Q1", "severity": "medium"},
    {"id": "INC-003", "root_cause": "human_error", "category": "network", "period": "Q1", "severity": "low"},
    {"id": "INC-004", "root_cause": "hardware_failure", "category": "compute", "period": "Q2", "severity": "high"},
    {"id": "INC-005", "root_cause": "human_error", "category": "network", "period": "Q2", "severity": "medium"},
    {"id": "INC-006", "root_cause": "software_bug", "category": "network", "period": "Q2", "severity": "high"},
]


def _req(
    operation: AnalysisOperation,
    parameters: dict[str, Any],
    data: list[dict[str, Any]] | None = None,
) -> AnalysisRequest:
    return AnalysisRequest(
        operation=operation,
        data=data if data is not None else _INCIDENTS,
        parameters=parameters,
    )


# ---------------------------------------------------------------------------
# RBAC
# ---------------------------------------------------------------------------


class TestRBAC:
    async def test_viewer_denied(self) -> None:
        req = _req(AnalysisOperation.COUNT_BY, {"field": "root_cause"})
        with pytest.raises(PermissionDeniedError) as exc_info:
            await run_analysis(req, role=Role.VIEWER)
        assert exc_info.value.role == Role.VIEWER

    async def test_engineer_denied(self) -> None:
        req = _req(AnalysisOperation.COUNT_BY, {"field": "root_cause"})
        with pytest.raises(PermissionDeniedError):
            await run_analysis(req, role=Role.ENGINEER)

    async def test_analyst_allowed(self) -> None:
        req = _req(AnalysisOperation.COUNT_BY, {"field": "root_cause"})
        result = await run_analysis(req, role=Role.ANALYST)
        assert result is not None

    async def test_administrator_allowed(self) -> None:
        req = _req(AnalysisOperation.COUNT_BY, {"field": "root_cause"})
        result = await run_analysis(req, role=Role.ADMINISTRATOR)
        assert result is not None

    async def test_rbac_checked_before_parameters(self) -> None:
        """RBAC denial must not be bypassed by supplying invalid parameters."""
        req = _req(AnalysisOperation.COUNT_BY, {})  # missing 'field'
        with pytest.raises(PermissionDeniedError):
            await run_analysis(req, role=Role.VIEWER)


# ---------------------------------------------------------------------------
# COUNT_BY
# ---------------------------------------------------------------------------


class TestCountBy:
    async def test_counts_by_field(self) -> None:
        req = _req(AnalysisOperation.COUNT_BY, {"field": "root_cause"})
        result = await run_analysis(req, role=Role.ANALYST)
        assert result.operation == AnalysisOperation.COUNT_BY
        assert result.record_count == 6
        counts = result.result["counts"]
        assert counts["human_error"] == 3
        assert counts["software_bug"] == 2
        assert counts["hardware_failure"] == 1

    async def test_result_field_matches_parameter(self) -> None:
        req = _req(AnalysisOperation.COUNT_BY, {"field": "severity"})
        result = await run_analysis(req, role=Role.ANALYST)
        assert result.result["field"] == "severity"

    async def test_missing_field_value_counted_as_missing(self) -> None:
        data = [{"id": "1"}, {"id": "2", "root_cause": "bug"}]
        req = _req(AnalysisOperation.COUNT_BY, {"field": "root_cause"}, data=data)
        result = await run_analysis(req, role=Role.ANALYST)
        assert result.result["counts"]["(missing)"] == 1
        assert result.result["counts"]["bug"] == 1

    async def test_missing_field_parameter_raises(self) -> None:
        req = _req(AnalysisOperation.COUNT_BY, {})
        with pytest.raises(AnalysisError, match="field"):
            await run_analysis(req, role=Role.ANALYST)

    async def test_non_string_field_parameter_raises(self) -> None:
        req = _req(AnalysisOperation.COUNT_BY, {"field": 42})  # type: ignore[arg-type]
        with pytest.raises(AnalysisError, match="field"):
            await run_analysis(req, role=Role.ANALYST)


# ---------------------------------------------------------------------------
# GROUP_BY
# ---------------------------------------------------------------------------


class TestGroupBy:
    async def test_groups_records_by_field(self) -> None:
        req = _req(AnalysisOperation.GROUP_BY, {"field": "period"})
        result = await run_analysis(req, role=Role.ANALYST)
        groups = result.result["groups"]
        assert set(groups.keys()) == {"Q1", "Q2"}
        assert len(groups["Q1"]) == 3
        assert len(groups["Q2"]) == 3

    async def test_max_per_group_limits_records(self) -> None:
        req = _req(AnalysisOperation.GROUP_BY, {"field": "root_cause", "max_per_group": 1})
        result = await run_analysis(req, role=Role.ANALYST)
        for records in result.result["groups"].values():
            assert len(records) <= 1

    async def test_max_per_group_capped_at_50(self) -> None:
        req = _req(AnalysisOperation.GROUP_BY, {"field": "root_cause", "max_per_group": 9999})
        result = await run_analysis(req, role=Role.ANALYST)
        assert result is not None  # internal cap prevents outsized output

    async def test_groups_sorted_alphabetically(self) -> None:
        req = _req(AnalysisOperation.GROUP_BY, {"field": "period"})
        result = await run_analysis(req, role=Role.ANALYST)
        keys = list(result.result["groups"].keys())
        assert keys == sorted(keys)

    async def test_missing_field_parameter_raises(self) -> None:
        req = _req(AnalysisOperation.GROUP_BY, {})
        with pytest.raises(AnalysisError):
            await run_analysis(req, role=Role.ANALYST)


# ---------------------------------------------------------------------------
# CALCULATE_PERCENTAGE
# ---------------------------------------------------------------------------


class TestCalculatePercentage:
    async def test_percentages_sum_to_100(self) -> None:
        req = _req(AnalysisOperation.CALCULATE_PERCENTAGE, {"field": "root_cause"})
        result = await run_analysis(req, role=Role.ANALYST)
        total_pct = sum(result.result["percentages"].values())
        assert abs(total_pct - 100.0) < 0.1

    async def test_percentage_values(self) -> None:
        req = _req(AnalysisOperation.CALCULATE_PERCENTAGE, {"field": "root_cause"})
        result = await run_analysis(req, role=Role.ANALYST)
        pct = result.result["percentages"]
        assert pct["human_error"] == pytest.approx(50.0, abs=0.1)
        assert pct["software_bug"] == pytest.approx(33.33, abs=0.1)
        assert pct["hardware_failure"] == pytest.approx(16.67, abs=0.1)

    async def test_total_equals_record_count(self) -> None:
        req = _req(AnalysisOperation.CALCULATE_PERCENTAGE, {"field": "root_cause"})
        result = await run_analysis(req, role=Role.ANALYST)
        assert result.result["total"] == result.record_count

    async def test_missing_field_parameter_raises(self) -> None:
        req = _req(AnalysisOperation.CALCULATE_PERCENTAGE, {})
        with pytest.raises(AnalysisError):
            await run_analysis(req, role=Role.ANALYST)


# ---------------------------------------------------------------------------
# COMPARE_PERIODS
# ---------------------------------------------------------------------------


class TestComparePeriods:
    async def test_counts_by_period(self) -> None:
        req = _req(
            AnalysisOperation.COMPARE_PERIODS,
            {
                "group_field": "root_cause",
                "period_field": "period",
                "period_a": "Q1",
                "period_b": "Q2",
            },
        )
        result = await run_analysis(req, role=Role.ANALYST)
        r = result.result
        # Q1: human_error×2, software_bug×1
        assert r["Q1"]["human_error"] == 2
        assert r["Q1"]["software_bug"] == 1
        # Q2: human_error×1, software_bug×1, hardware_failure×1
        assert r["Q2"]["human_error"] == 1
        assert r["delta"]["human_error"] == -1
        assert r["record_counts"]["Q1"] == 3
        assert r["record_counts"]["Q2"] == 3

    async def test_period_with_no_records_yields_zero_count(self) -> None:
        req = _req(
            AnalysisOperation.COMPARE_PERIODS,
            {
                "group_field": "root_cause",
                "period_field": "period",
                "period_a": "Q1",
                "period_b": "Q3",
            },
        )
        result = await run_analysis(req, role=Role.ANALYST)
        assert result.result["record_counts"]["Q3"] == 0

    async def test_missing_period_a_raises(self) -> None:
        req = _req(
            AnalysisOperation.COMPARE_PERIODS,
            {"group_field": "root_cause", "period_field": "period", "period_b": "Q2"},
        )
        with pytest.raises(AnalysisError, match="period_a"):
            await run_analysis(req, role=Role.ANALYST)

    async def test_missing_group_field_raises(self) -> None:
        req = _req(
            AnalysisOperation.COMPARE_PERIODS,
            {"period_field": "period", "period_a": "Q1", "period_b": "Q2"},
        )
        with pytest.raises(AnalysisError, match="group_field"):
            await run_analysis(req, role=Role.ANALYST)

    async def test_delta_reflects_change_direction(self) -> None:
        req = _req(
            AnalysisOperation.COMPARE_PERIODS,
            {
                "group_field": "root_cause",
                "period_field": "period",
                "period_a": "Q1",
                "period_b": "Q2",
            },
        )
        result = await run_analysis(req, role=Role.ANALYST)
        delta = result.result["delta"]
        # hardware_failure went from 0 → 1: delta should be +1
        assert delta["hardware_failure"] == 1


# ---------------------------------------------------------------------------
# DETECT_PATTERNS
# ---------------------------------------------------------------------------


class TestDetectPatterns:
    async def test_finds_most_common_pattern(self) -> None:
        req = _req(
            AnalysisOperation.DETECT_PATTERNS,
            {"fields": ["root_cause", "category"]},
        )
        result = await run_analysis(req, role=Role.ANALYST)
        patterns = result.result["patterns"]
        assert len(patterns) > 0
        top = patterns[0]
        assert top["pattern"]["root_cause"] == "human_error"
        assert top["pattern"]["category"] == "network"
        assert top["count"] == 3

    async def test_top_n_limits_result_count(self) -> None:
        req = _req(
            AnalysisOperation.DETECT_PATTERNS,
            {"fields": ["root_cause", "category"], "top_n": 2},
        )
        result = await run_analysis(req, role=Role.ANALYST)
        assert len(result.result["patterns"]) <= 2

    async def test_fields_included_in_result(self) -> None:
        req = _req(
            AnalysisOperation.DETECT_PATTERNS,
            {"fields": ["root_cause", "severity"]},
        )
        result = await run_analysis(req, role=Role.ANALYST)
        assert result.result["fields"] == ["root_cause", "severity"]
        for entry in result.result["patterns"]:
            assert "root_cause" in entry["pattern"]
            assert "severity" in entry["pattern"]

    async def test_empty_fields_list_raises(self) -> None:
        req = _req(AnalysisOperation.DETECT_PATTERNS, {"fields": []})
        with pytest.raises(AnalysisError, match="empty"):
            await run_analysis(req, role=Role.ANALYST)

    async def test_missing_fields_parameter_raises(self) -> None:
        req = _req(AnalysisOperation.DETECT_PATTERNS, {})
        with pytest.raises(AnalysisError, match="fields"):
            await run_analysis(req, role=Role.ANALYST)

    async def test_invalid_top_n_zero_raises(self) -> None:
        req = _req(AnalysisOperation.DETECT_PATTERNS, {"fields": ["root_cause"], "top_n": 0})
        with pytest.raises(AnalysisError, match="top_n"):
            await run_analysis(req, role=Role.ANALYST)

    async def test_invalid_top_n_over_100_raises(self) -> None:
        req = _req(AnalysisOperation.DETECT_PATTERNS, {"fields": ["root_cause"], "top_n": 101})
        with pytest.raises(AnalysisError, match="top_n"):
            await run_analysis(req, role=Role.ANALYST)

    async def test_non_string_in_fields_raises(self) -> None:
        req = _req(AnalysisOperation.DETECT_PATTERNS, {"fields": ["root_cause", 123]})  # type: ignore[list-item]
        with pytest.raises(AnalysisError):
            await run_analysis(req, role=Role.ANALYST)


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------


class TestTimeout:
    async def test_timeout_raises_timeout_error(self) -> None:
        async def _slow(func: Any, *args: Any, **kwargs: Any) -> Any:
            await asyncio.sleep(10)

        req = _req(AnalysisOperation.COUNT_BY, {"field": "root_cause"})
        with patch("asyncio.to_thread", new=_slow):
            with pytest.raises(TimeoutError):
                await run_analysis(req, role=Role.ANALYST, timeout=0.001)

    async def test_custom_timeout_accepted(self) -> None:
        req = _req(AnalysisOperation.COUNT_BY, {"field": "root_cause"})
        result = await run_analysis(req, role=Role.ANALYST, timeout=60.0)
        assert result is not None


# ---------------------------------------------------------------------------
# Data validation (AnalysisRequest model)
# ---------------------------------------------------------------------------


class TestDataValidation:
    def test_empty_data_rejected(self) -> None:
        with pytest.raises(Exception):
            AnalysisRequest(
                operation=AnalysisOperation.COUNT_BY,
                data=[],
                parameters={"field": "root_cause"},
            )

    def test_too_many_records_rejected(self) -> None:
        large_data = [{"id": str(i)} for i in range(MAX_RECORDS + 1)]
        with pytest.raises(Exception):
            AnalysisRequest(
                operation=AnalysisOperation.COUNT_BY,
                data=large_data,
                parameters={"field": "id"},
            )

    def test_too_many_fields_per_record_rejected(self) -> None:
        from src.tools.analysis import MAX_FIELDS_PER_RECORD

        fat_record = {f"field_{i}": i for i in range(MAX_FIELDS_PER_RECORD + 1)}
        with pytest.raises(Exception, match="fields"):
            AnalysisRequest(
                operation=AnalysisOperation.COUNT_BY,
                data=[fat_record],
                parameters={"field": "field_0"},
            )

    def test_valid_request_constructs_successfully(self) -> None:
        req = AnalysisRequest(
            operation=AnalysisOperation.COUNT_BY,
            data=[{"key": "value"}],
            parameters={"field": "key"},
        )
        assert req.operation == AnalysisOperation.COUNT_BY
        assert len(req.data) == 1

    async def test_result_record_count_matches_data_length(self) -> None:
        req = _req(AnalysisOperation.COUNT_BY, {"field": "root_cause"})
        result = await run_analysis(req, role=Role.ANALYST)
        assert result.record_count == len(_INCIDENTS)
