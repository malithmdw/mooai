"""Tests for ``src.mcp.data`` — the synthetic enterprise data layer.

Coverage
--------
- lookup_employee: by exact ID, partial name, case-insensitive name,
  ID precedence over name, not-found
- lookup_service: by exact ID, case-insensitive, not-found
- lookup_incident: by exact ID, case-insensitive, not-found
- list_* functions: count, sorted order, dict-copy isolation
- data integrity: every employee manager_id that is set refers to a
  real employee; every incident service_id refers to a real service;
  every incident assigned_to refers to a real employee
"""

from __future__ import annotations

import pytest

from src.mcp.data import (
    employee_count,
    incident_count,
    list_employees,
    list_incidents,
    list_services,
    lookup_employee,
    lookup_incident,
    lookup_service,
    service_count,
)


# ---------------------------------------------------------------------------
# lookup_employee
# ---------------------------------------------------------------------------


class TestLookupEmployee:
    def test_exact_id_match(self) -> None:
        emp = lookup_employee("EMP-001")
        assert emp is not None
        assert emp["employee_id"] == "EMP-001"
        assert emp["name"] == "Alice Chen"

    def test_exact_id_case_insensitive(self) -> None:
        emp = lookup_employee("emp-001")
        assert emp is not None
        assert emp["employee_id"] == "EMP-001"

    def test_partial_name_match(self) -> None:
        emp = lookup_employee("Alice")
        assert emp is not None
        assert emp["name"] == "Alice Chen"

    def test_partial_name_case_insensitive(self) -> None:
        emp = lookup_employee("alice chen")
        assert emp is not None
        assert emp["employee_id"] == "EMP-001"

    def test_partial_name_surname_only(self) -> None:
        emp = lookup_employee("Wilson")
        assert emp is not None
        assert emp["employee_id"] == "EMP-008"

    def test_id_takes_priority_over_name(self) -> None:
        emp = lookup_employee("EMP-008")
        assert emp is not None
        assert emp["name"] == "Henry Wilson"

    def test_not_found_returns_none(self) -> None:
        assert lookup_employee("EMP-999") is None
        assert lookup_employee("Nonexistent Person") is None

    def test_returns_copy(self) -> None:
        emp = lookup_employee("EMP-001")
        assert emp is not None
        emp["name"] = "mutated"
        assert lookup_employee("EMP-001") is not None
        assert lookup_employee("EMP-001")["name"] == "Alice Chen"  # type: ignore[index]

    def test_result_has_required_fields(self) -> None:
        emp = lookup_employee("EMP-001")
        assert emp is not None
        for field in ("employee_id", "name", "title", "department", "email", "location"):
            assert field in emp


# ---------------------------------------------------------------------------
# lookup_service
# ---------------------------------------------------------------------------


class TestLookupService:
    def test_exact_id(self) -> None:
        svc = lookup_service("SVC-001")
        assert svc is not None
        assert svc["service_id"] == "SVC-001"
        assert svc["name"] == "Core Banking API"

    def test_case_insensitive_id(self) -> None:
        svc = lookup_service("svc-001")
        assert svc is not None
        assert svc["service_id"] == "SVC-001"

    def test_not_found(self) -> None:
        assert lookup_service("SVC-999") is None

    def test_result_has_required_fields(self) -> None:
        svc = lookup_service("SVC-001")
        assert svc is not None
        for field in ("service_id", "name", "description", "tier", "owner_team", "sla_uptime_pct"):
            assert field in svc

    def test_returns_copy(self) -> None:
        svc = lookup_service("SVC-001")
        assert svc is not None
        svc["name"] = "mutated"
        assert lookup_service("SVC-001")["name"] == "Core Banking API"  # type: ignore[index]


# ---------------------------------------------------------------------------
# lookup_incident
# ---------------------------------------------------------------------------


class TestLookupIncident:
    def test_exact_id(self) -> None:
        inc = lookup_incident("INC-001")
        assert inc is not None
        assert inc["incident_id"] == "INC-001"

    def test_case_insensitive_id(self) -> None:
        inc = lookup_incident("inc-001")
        assert inc is not None
        assert inc["incident_id"] == "INC-001"

    def test_not_found(self) -> None:
        assert lookup_incident("INC-999") is None

    def test_result_has_required_fields(self) -> None:
        inc = lookup_incident("INC-001")
        assert inc is not None
        for field in (
            "incident_id",
            "title",
            "service_id",
            "severity",
            "status",
            "root_cause",
            "assigned_to",
            "created_at",
        ):
            assert field in inc

    def test_resolved_incident_has_resolved_at(self) -> None:
        inc = lookup_incident("INC-001")
        assert inc is not None
        assert inc["status"] == "resolved"
        assert inc["resolved_at"] is not None

    def test_open_incident_has_no_resolved_at(self) -> None:
        inc = lookup_incident("INC-003")
        assert inc is not None
        assert inc["status"] in ("open", "investigating")
        assert inc["resolved_at"] is None


# ---------------------------------------------------------------------------
# list_* functions
# ---------------------------------------------------------------------------


class TestListFunctions:
    def test_employee_count_matches_list(self) -> None:
        employees = list_employees()
        assert len(employees) == employee_count()

    def test_service_count_matches_list(self) -> None:
        services = list_services()
        assert len(services) == service_count()

    def test_incident_count_matches_list(self) -> None:
        incidents = list_incidents()
        assert len(incidents) == incident_count()

    def test_employees_sorted_by_id(self) -> None:
        employees = list_employees()
        ids = [e["employee_id"] for e in employees]
        assert ids == sorted(ids)

    def test_services_sorted_by_id(self) -> None:
        services = list_services()
        ids = [s["service_id"] for s in services]
        assert ids == sorted(ids)

    def test_incidents_sorted_by_id(self) -> None:
        incidents = list_incidents()
        ids = [i["incident_id"] for i in incidents]
        assert ids == sorted(ids)

    def test_list_employees_returns_copies(self) -> None:
        employees = list_employees()
        first = employees[0]
        original_name = first["name"]
        first["name"] = "mutated"
        # Second call must return fresh copies
        employees2 = list_employees()
        assert employees2[0]["name"] == original_name

    def test_employee_count_positive(self) -> None:
        assert employee_count() > 0

    def test_service_count_positive(self) -> None:
        assert service_count() > 0

    def test_incident_count_positive(self) -> None:
        assert incident_count() > 0


# ---------------------------------------------------------------------------
# Data integrity
# ---------------------------------------------------------------------------


class TestDataIntegrity:
    def test_employee_manager_ids_valid(self) -> None:
        """Every non-null manager_id must reference an existing employee."""
        employees = list_employees()
        all_ids = {e["employee_id"] for e in employees}
        for emp in employees:
            mgr = emp.get("manager_id")
            if mgr is not None:
                assert mgr in all_ids, (
                    f"{emp['employee_id']} has manager_id={mgr!r} which is not in the directory"
                )

    def test_incident_service_ids_valid(self) -> None:
        """Every incident's service_id must reference a real service."""
        incidents = list_incidents()
        services = list_services()
        service_ids = {s["service_id"] for s in services}
        for inc in incidents:
            assert inc["service_id"] in service_ids, (
                f"{inc['incident_id']} references unknown service {inc['service_id']!r}"
            )

    def test_incident_assigned_to_valid(self) -> None:
        """Every incident's assigned_to must reference a real employee."""
        incidents = list_incidents()
        employees = list_employees()
        emp_ids = {e["employee_id"] for e in employees}
        for inc in incidents:
            assert inc["assigned_to"] in emp_ids, (
                f"{inc['incident_id']} assigned_to={inc['assigned_to']!r} not in directory"
            )

    def test_all_severities_valid(self) -> None:
        valid = {"P1", "P2", "P3", "P4"}
        for inc in list_incidents():
            assert inc["severity"] in valid, (
                f"{inc['incident_id']} has invalid severity {inc['severity']!r}"
            )

    def test_all_statuses_valid(self) -> None:
        valid = {"open", "resolved", "investigating"}
        for inc in list_incidents():
            assert inc["status"] in valid, (
                f"{inc['incident_id']} has invalid status {inc['status']!r}"
            )

    def test_service_tiers_valid(self) -> None:
        valid = {"Tier 1", "Tier 2", "Tier 3"}
        for svc in list_services():
            assert svc["tier"] in valid, (
                f"{svc['service_id']} has invalid tier {svc['tier']!r}"
            )

    def test_service_dependency_ids_valid(self) -> None:
        """Every service dependency must reference a real service."""
        services = list_services()
        service_ids = {s["service_id"] for s in services}
        for svc in services:
            for dep in svc.get("dependencies", []):
                assert dep in service_ids, (
                    f"{svc['service_id']} has unknown dependency {dep!r}"
                )

    def test_employees_have_unique_emails(self) -> None:
        emails = [e["email"] for e in list_employees()]
        assert len(emails) == len(set(emails))

    def test_no_self_referential_managers(self) -> None:
        for emp in list_employees():
            assert emp.get("manager_id") != emp["employee_id"], (
                f"{emp['employee_id']} is its own manager"
            )
