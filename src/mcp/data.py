"""Synthetic enterprise data — all records are entirely fictional.

Provides three in-memory data sets (employee directory, service catalog,
incident records) and lookup functions used by both the standalone MCP
server and the in-process client adapter.

No real people, services, or incidents are represented here.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Employee directory
# ---------------------------------------------------------------------------

_EMPLOYEES: dict[str, dict[str, Any]] = {
    "EMP-001": {
        "employee_id": "EMP-001",
        "name": "Alice Chen",
        "title": "Senior Software Engineer",
        "department": "Platform Engineering",
        "email": "a.chen@novuscorp.example",
        "location": "London, UK",
        "phone": "+44 20 7946 0001",
        "manager_id": "EMP-008",
    },
    "EMP-002": {
        "employee_id": "EMP-002",
        "name": "Bob Martinez",
        "title": "Product Manager",
        "department": "Digital Banking",
        "email": "b.martinez@novuscorp.example",
        "location": "London, UK",
        "phone": "+44 20 7946 0002",
        "manager_id": "EMP-010",
    },
    "EMP-003": {
        "employee_id": "EMP-003",
        "name": "Carol Smith",
        "title": "Data Scientist",
        "department": "Analytics & Data",
        "email": "c.smith@novuscorp.example",
        "location": "Manchester, UK",
        "phone": "+44 16 1946 0003",
        "manager_id": "EMP-010",
    },
    "EMP-004": {
        "employee_id": "EMP-004",
        "name": "David Johnson",
        "title": "DevOps Engineer",
        "department": "Infrastructure",
        "email": "d.johnson@novuscorp.example",
        "location": "Edinburgh, UK",
        "phone": "+44 13 1946 0004",
        "manager_id": "EMP-008",
    },
    "EMP-005": {
        "employee_id": "EMP-005",
        "name": "Emma Williams",
        "title": "Security Engineer",
        "department": "Cybersecurity",
        "email": "e.williams@novuscorp.example",
        "location": "London, UK",
        "phone": "+44 20 7946 0005",
        "manager_id": "EMP-009",
    },
    "EMP-006": {
        "employee_id": "EMP-006",
        "name": "Frank Brown",
        "title": "Business Analyst",
        "department": "Lending & Credit",
        "email": "f.brown@novuscorp.example",
        "location": "Bristol, UK",
        "phone": "+44 11 7946 0006",
        "manager_id": "EMP-010",
    },
    "EMP-007": {
        "employee_id": "EMP-007",
        "name": "Grace Davis",
        "title": "QA Lead",
        "department": "Quality Assurance",
        "email": "g.davis@novuscorp.example",
        "location": "London, UK",
        "phone": "+44 20 7946 0007",
        "manager_id": "EMP-001",
    },
    "EMP-008": {
        "employee_id": "EMP-008",
        "name": "Henry Wilson",
        "title": "Enterprise Architect",
        "department": "Architecture",
        "email": "h.wilson@novuscorp.example",
        "location": "London, UK",
        "phone": "+44 20 7946 0008",
        "manager_id": None,
    },
    "EMP-009": {
        "employee_id": "EMP-009",
        "name": "Isabella Taylor",
        "title": "Chief Compliance Officer",
        "department": "Risk & Compliance",
        "email": "i.taylor@novuscorp.example",
        "location": "London, UK",
        "phone": "+44 20 7946 0009",
        "manager_id": None,
    },
    "EMP-010": {
        "employee_id": "EMP-010",
        "name": "James Anderson",
        "title": "Engineering Tech Lead",
        "department": "Core Banking",
        "email": "j.anderson@novuscorp.example",
        "location": "London, UK",
        "phone": "+44 20 7946 0010",
        "manager_id": "EMP-008",
    },
    "EMP-011": {
        "employee_id": "EMP-011",
        "name": "Karen Lee",
        "title": "UI/UX Designer",
        "department": "Product",
        "email": "k.lee@novuscorp.example",
        "location": "London, UK",
        "phone": "+44 20 7946 0011",
        "manager_id": "EMP-002",
    },
    "EMP-012": {
        "employee_id": "EMP-012",
        "name": "Liam Nguyen",
        "title": "Database Administrator",
        "department": "Infrastructure",
        "email": "l.nguyen@novuscorp.example",
        "location": "Manchester, UK",
        "phone": "+44 16 1946 0012",
        "manager_id": "EMP-004",
    },
}

# ---------------------------------------------------------------------------
# Service catalog
# ---------------------------------------------------------------------------

_SERVICES: dict[str, dict[str, Any]] = {
    "SVC-001": {
        "service_id": "SVC-001",
        "name": "Core Banking API",
        "description": (
            "Central ledger and account management API. Handles account creation, "
            "balance queries, transaction posting, and statement generation."
        ),
        "tier": "Tier 1",
        "owner_team": "Core Banking",
        "owner_employee_id": "EMP-010",
        "oncall_email": "oncall-corebanking@novuscorp.example",
        "sla_uptime_pct": 99.95,
        "dependencies": ["SVC-003"],
        "tech_stack": ["Python", "PostgreSQL", "Redis", "Kubernetes"],
    },
    "SVC-002": {
        "service_id": "SVC-002",
        "name": "Payment Gateway",
        "description": (
            "Payment initiation, routing, and settlement. Supports SWIFT, SEPA, "
            "Faster Payments, and card scheme integrations."
        ),
        "tier": "Tier 1",
        "owner_team": "Payments",
        "owner_employee_id": "EMP-010",
        "oncall_email": "oncall-payments@novuscorp.example",
        "sla_uptime_pct": 99.99,
        "dependencies": ["SVC-001", "SVC-003", "SVC-007"],
        "tech_stack": ["Java", "Kafka", "PostgreSQL", "Kubernetes"],
    },
    "SVC-003": {
        "service_id": "SVC-003",
        "name": "Customer Identity Service",
        "description": (
            "Authentication, authorisation, and customer profile management. "
            "Issues JWTs for internal service-to-service calls."
        ),
        "tier": "Tier 1",
        "owner_team": "Platform Engineering",
        "owner_employee_id": "EMP-001",
        "oncall_email": "oncall-identity@novuscorp.example",
        "sla_uptime_pct": 99.99,
        "dependencies": [],
        "tech_stack": ["Python", "PostgreSQL", "Redis"],
    },
    "SVC-004": {
        "service_id": "SVC-004",
        "name": "Loan Origination System",
        "description": (
            "End-to-end loan application, credit scoring, underwriting, "
            "and disbursement workflow."
        ),
        "tier": "Tier 2",
        "owner_team": "Lending & Credit",
        "owner_employee_id": "EMP-006",
        "oncall_email": "oncall-lending@novuscorp.example",
        "sla_uptime_pct": 99.9,
        "dependencies": ["SVC-001", "SVC-003"],
        "tech_stack": ["Python", "PostgreSQL", "Celery"],
    },
    "SVC-005": {
        "service_id": "SVC-005",
        "name": "Mobile Banking Platform",
        "description": (
            "Backend for the Novus Corp mobile app — account overview, "
            "payments, notifications, and biometric authentication."
        ),
        "tier": "Tier 2",
        "owner_team": "Digital Banking",
        "owner_employee_id": "EMP-002",
        "oncall_email": "oncall-mobile@novuscorp.example",
        "sla_uptime_pct": 99.9,
        "dependencies": ["SVC-001", "SVC-002", "SVC-003", "SVC-008"],
        "tech_stack": ["Node.js", "Redis", "Kubernetes"],
    },
    "SVC-006": {
        "service_id": "SVC-006",
        "name": "Enterprise Data Warehouse",
        "description": (
            "Central analytics data store. Aggregates operational data from all "
            "Tier 1/2 services via nightly ETL pipelines."
        ),
        "tier": "Tier 2",
        "owner_team": "Analytics & Data",
        "owner_employee_id": "EMP-003",
        "oncall_email": "oncall-data@novuscorp.example",
        "sla_uptime_pct": 99.5,
        "dependencies": ["SVC-001", "SVC-002", "SVC-004"],
        "tech_stack": ["Python", "dbt", "Snowflake", "Airflow"],
    },
    "SVC-007": {
        "service_id": "SVC-007",
        "name": "Fraud Detection Service",
        "description": (
            "Real-time transaction scoring using ML models. Blocks or flags "
            "suspicious activity within 50 ms of transaction initiation."
        ),
        "tier": "Tier 1",
        "owner_team": "Cybersecurity",
        "owner_employee_id": "EMP-005",
        "oncall_email": "oncall-fraud@novuscorp.example",
        "sla_uptime_pct": 99.95,
        "dependencies": ["SVC-001", "SVC-003"],
        "tech_stack": ["Python", "Redis", "Kafka", "TensorFlow"],
    },
    "SVC-008": {
        "service_id": "SVC-008",
        "name": "Notification Service",
        "description": (
            "Multi-channel customer and internal notification dispatch: "
            "email, SMS, push notifications, and webhook delivery."
        ),
        "tier": "Tier 3",
        "owner_team": "Platform Engineering",
        "owner_employee_id": "EMP-001",
        "oncall_email": "oncall-notifications@novuscorp.example",
        "sla_uptime_pct": 99.5,
        "dependencies": ["SVC-003"],
        "tech_stack": ["Python", "Redis", "Celery", "SendGrid"],
    },
}

# ---------------------------------------------------------------------------
# Incident records
# ---------------------------------------------------------------------------

_INCIDENTS: dict[str, dict[str, Any]] = {
    "INC-001": {
        "incident_id": "INC-001",
        "title": "Payment Gateway timeout surge — 3,000 customers impacted",
        "service_id": "SVC-002",
        "severity": "P1",
        "status": "resolved",
        "root_cause": "database_connection_exhaustion",
        "description": (
            "Payment gateway began returning 504 timeouts at 14:32 UTC. "
            "Root cause: connection pool to upstream PostgreSQL exhausted after a "
            "misconfigured deployment increased connection hold time from 2 s to 120 s."
        ),
        "assigned_to": "EMP-012",
        "created_at": "2024-08-12T14:32:00Z",
        "resolved_at": "2024-08-12T16:10:00Z",
        "duration_minutes": 98,
    },
    "INC-002": {
        "incident_id": "INC-002",
        "title": "Slow query degrading Loan Origination System response times",
        "service_id": "SVC-004",
        "severity": "P2",
        "status": "resolved",
        "root_cause": "missing_database_index",
        "description": (
            "Credit-check queries against the applicant_history table began taking "
            "45+ seconds after a data migration added 8 M rows without updating "
            "the query planner statistics."
        ),
        "assigned_to": "EMP-012",
        "created_at": "2024-07-03T09:15:00Z",
        "resolved_at": "2024-07-03T11:45:00Z",
        "duration_minutes": 150,
    },
    "INC-003": {
        "incident_id": "INC-003",
        "title": "Core Banking API returning 500 errors on balance query endpoint",
        "service_id": "SVC-001",
        "severity": "P1",
        "status": "investigating",
        "root_cause": "under_investigation",
        "description": (
            "Intermittent 500 errors on GET /accounts/{id}/balance started at 08:17 UTC. "
            "Error rate peaked at 12 % of requests. Correlated with a Redis cache "
            "eviction spike — exact cause still under investigation."
        ),
        "assigned_to": "EMP-001",
        "created_at": "2024-09-15T08:17:00Z",
        "resolved_at": None,
        "duration_minutes": None,
    },
    "INC-004": {
        "incident_id": "INC-004",
        "title": "Notification Service — email delivery delays up to 4 hours",
        "service_id": "SVC-008",
        "severity": "P3",
        "status": "resolved",
        "root_cause": "third_party_vendor_outage",
        "description": (
            "SendGrid's EU region experienced degraded throughput. Emails queued in "
            "Celery were not delivered for up to 4 hours. SMS and push notifications "
            "were unaffected."
        ),
        "assigned_to": "EMP-004",
        "created_at": "2024-06-18T11:00:00Z",
        "resolved_at": "2024-06-18T15:30:00Z",
        "duration_minutes": 270,
    },
    "INC-005": {
        "incident_id": "INC-005",
        "title": "Fraud Detection false-positive rate spike — 4 % of transactions blocked",
        "service_id": "SVC-007",
        "severity": "P2",
        "status": "resolved",
        "root_cause": "model_feature_drift",
        "description": (
            "Following a weekend bank-holiday spike in legitimate international transfers, "
            "the fraud model's feature distribution shifted, causing 4 % of genuine "
            "transactions to be scored as fraudulent and blocked."
        ),
        "assigned_to": "EMP-005",
        "created_at": "2024-08-27T07:00:00Z",
        "resolved_at": "2024-08-27T13:20:00Z",
        "duration_minutes": 380,
    },
    "INC-006": {
        "incident_id": "INC-006",
        "title": "Customer Identity Service — JWT token expiry misconfiguration",
        "service_id": "SVC-003",
        "severity": "P2",
        "status": "open",
        "root_cause": "configuration_error",
        "description": (
            "A config-file change rolled out at 19:00 UTC set JWT TTL to 60 seconds "
            "instead of 3600 seconds, causing mobile app users to be logged out every "
            "minute. Rollback in progress."
        ),
        "assigned_to": "EMP-001",
        "created_at": "2024-09-10T19:05:00Z",
        "resolved_at": None,
        "duration_minutes": None,
    },
    "INC-007": {
        "incident_id": "INC-007",
        "title": "Enterprise Data Warehouse ETL pipeline failure — stale reporting data",
        "service_id": "SVC-006",
        "severity": "P4",
        "status": "resolved",
        "root_cause": "upstream_schema_change",
        "description": (
            "Nightly ETL job failed when the Core Banking API introduced a new nullable "
            "column `account_sub_type` not present in the warehouse schema. "
            "Daily executive reports contained T-2 data for 48 hours."
        ),
        "assigned_to": "EMP-003",
        "created_at": "2024-05-22T02:05:00Z",
        "resolved_at": "2024-05-23T10:30:00Z",
        "duration_minutes": 1945,
    },
    "INC-008": {
        "incident_id": "INC-008",
        "title": "Mobile Banking app force-close on iOS 17.4 after login",
        "service_id": "SVC-005",
        "severity": "P3",
        "status": "resolved",
        "root_cause": "third_party_sdk_incompatibility",
        "description": (
            "Apple's iOS 17.4 update changed the behaviour of WKWebView's "
            "JavaScript bridge. The biometric authentication SDK v2.1.0 triggered a "
            "null-pointer dereference, causing force-closes on affected devices."
        ),
        "assigned_to": "EMP-011",
        "created_at": "2024-03-06T14:00:00Z",
        "resolved_at": "2024-03-08T09:15:00Z",
        "duration_minutes": 2835,
    },
    "INC-009": {
        "incident_id": "INC-009",
        "title": "Database connection pool exhaustion — services degraded cluster-wide",
        "service_id": "SVC-001",
        "severity": "P1",
        "status": "resolved",
        "root_cause": "connection_leak",
        "description": (
            "A long-running batch job introduced in sprint 47 leaked database connections "
            "when a network partition occurred mid-transaction. pgBouncer pool filled "
            "within 20 minutes, cascading across SVC-001, SVC-003, and SVC-007."
        ),
        "assigned_to": "EMP-012",
        "created_at": "2024-02-14T03:47:00Z",
        "resolved_at": "2024-02-14T05:12:00Z",
        "duration_minutes": 85,
    },
    "INC-010": {
        "incident_id": "INC-010",
        "title": "Fraud Detection model drift — elevated false-negative rate detected",
        "service_id": "SVC-007",
        "severity": "P2",
        "status": "investigating",
        "root_cause": "under_investigation",
        "description": (
            "Weekly model evaluation flagged a 2.1 pp increase in the false-negative "
            "rate compared to the prior 30-day baseline. Suspected cause: seasonal "
            "pattern shift in merchant category codes following the summer-holiday period. "
            "Retraining pipeline initiated."
        ),
        "assigned_to": "EMP-005",
        "created_at": "2024-09-09T10:00:00Z",
        "resolved_at": None,
        "duration_minutes": None,
    },
}


# ---------------------------------------------------------------------------
# Lookup functions
# ---------------------------------------------------------------------------


def lookup_employee(query: str) -> dict[str, Any] | None:
    """Find an employee by ID (exact) or name (case-insensitive partial match).

    Returns the first matching record, or ``None`` if no match is found.
    ID lookup takes priority over name search.
    """
    query = query.strip()
    # Exact ID lookup
    if query.upper() in _EMPLOYEES:
        return dict(_EMPLOYEES[query.upper()])
    # Case-insensitive partial name match
    q = query.lower()
    for emp in _EMPLOYEES.values():
        if q in emp["name"].lower():
            return dict(emp)
    return None


def lookup_service(service_id: str) -> dict[str, Any] | None:
    """Return the service record for ``service_id``, or ``None``."""
    return dict(_SERVICES[service_id.upper()]) if service_id.upper() in _SERVICES else None


def lookup_incident(incident_id: str) -> dict[str, Any] | None:
    """Return the incident record for ``incident_id``, or ``None``."""
    return dict(_INCIDENTS[incident_id.upper()]) if incident_id.upper() in _INCIDENTS else None


def list_employees() -> list[dict[str, Any]]:
    """Return all employee records sorted by employee_id."""
    return [dict(v) for v in sorted(_EMPLOYEES.values(), key=lambda e: e["employee_id"])]


def list_services() -> list[dict[str, Any]]:
    """Return all service records sorted by service_id."""
    return [dict(v) for v in sorted(_SERVICES.values(), key=lambda s: s["service_id"])]


def list_incidents() -> list[dict[str, Any]]:
    """Return all incident records sorted by incident_id."""
    return [dict(v) for v in sorted(_INCIDENTS.values(), key=lambda i: i["incident_id"])]


def employee_count() -> int:
    return len(_EMPLOYEES)


def service_count() -> int:
    return len(_SERVICES)


def incident_count() -> int:
    return len(_INCIDENTS)
