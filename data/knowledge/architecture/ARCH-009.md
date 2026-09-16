---
document_id: ARCH-009
title: Observability and Alerting Platform
department: Platform Engineering
document_type: architecture_document
access_level: INTERNAL
created_date: "2024-02-01"
allowed_roles: ["ENGINEER", "ANALYST", "ADMINISTRATOR"]
---

# Observability and Alerting Platform

## 1. Purpose

This document describes the observability stack used to monitor Novus Bank's production systems, including metrics collection, distributed tracing, structured logging, and alerting.

## 2. Observability Stack

| Concern | Technology | Notes |
|---------|-----------|-------|
| Metrics | Prometheus + Grafana | 30-day retention |
| Distributed tracing | Jaeger (OpenTelemetry) | 7-day retention |
| Structured logging | ELK Stack (Elasticsearch, Logstash, Kibana) | 90-day retention |
| Alerting | PagerDuty (via Prometheus Alertmanager) | On-call rotation |
| Synthetic monitoring | k6 + Grafana | Canary transactions every 60 s |
| AI tracing | LangSmith | Agent execution traces |

## 3. Key Dashboards

| Dashboard | URL | Owner |
|-----------|-----|-------|
| Payment gateway SLOs | grafana.internal/d/pmt-gateway | Payments Engineering |
| Core banking ledger | grafana.internal/d/cbl-health | Core Banking |
| Sanctions screening | grafana.internal/d/sanctions | Compliance Technology |
| Fraud detection | grafana.internal/d/fraud-detection | Financial Crime |
| Open Banking API | grafana.internal/d/open-banking | Open Banking |

## 4. Alerting Thresholds

### Payment Gateway

| Alert | Condition | Severity |
|-------|----------|---------|
| Payment failure rate | > 5 % for 2 min | P1 |
| p99 latency | > 3 s for 5 min | P1 |
| Sanctions screening unavailable | Any | P1 |
| Idempotency store failure rate | > 0.1 % | P2 |

### Core Banking

| Alert | Condition | Severity |
|-------|----------|---------|
| Replication lag | > 2 min | P1 |
| Connection pool utilisation | > 80 % | P2 |
| WAL archive lag | > 5 min | P2 |

## 5. On-Call Rotation

Engineering on-call is 24/7 with primary and secondary escalation:

- **Primary**: page immediately for P1
- **Secondary**: escalate after 5 minutes unacknowledged P1
- **Engineering lead**: escalate after 15 minutes unacknowledged P1

Payment-specific incidents auto-notify the Payments Engineering Slack channel (`#payments-oncall`).

## 6. Structured Log Fields

All services must emit logs with these mandatory fields:
```json
{
  "timestamp": "ISO8601",
  "level": "INFO|WARN|ERROR",
  "service": "service-name",
  "trace_id": "opentelemetry-trace-id",
  "span_id": "opentelemetry-span-id",
  "event": "human-readable-description",
  "error": "error message if applicable"
}
```

## 7. Related Documents

- **RUN-001**: On-call handbook
- **POL-001**: SLA and SLO policy
- **INC-2024-001**: cert-rotator unstructured log missed by alerting
