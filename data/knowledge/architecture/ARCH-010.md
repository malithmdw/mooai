---
document_id: ARCH-010
title: Disaster Recovery and Business Continuity Architecture
department: Platform Engineering
document_type: architecture_document
access_level: RESTRICTED
created_date: "2024-03-01"
allowed_roles: ["ANALYST", "ADMINISTRATOR"]
---

# Disaster Recovery and Business Continuity Architecture

## 1. Purpose

This document describes the Disaster Recovery (DR) and Business Continuity (BCP) architecture for Novus Bank's critical systems, including RTO and RPO targets, failover procedures, and DR testing requirements.

## 2. Recovery Targets

| System | RTO | RPO | Tier |
|--------|-----|-----|------|
| Core banking ledger | 15 min | 15 min | Critical |
| Payment gateway | 10 min | 5 min | Critical |
| SWIFT gateway | 30 min | 5 min | Critical |
| Sanctions screening | 20 min | 30 min | Critical |
| Mobile banking API | 30 min | 15 min | High |
| Open Banking API | 60 min | 30 min | High |
| Fraud detection service | 30 min | 60 min | High |

## 3. Infrastructure

| DC | Location | Role | Active |
|----|---------|------|--------|
| DC-PRIMARY | London (on-prem) | Primary | Yes |
| DC-DR | Manchester (on-prem) | Warm standby | Standby |
| DC-CLOUD | AWS eu-west-2 | Cold standby / burst | Cold |

Data replication:
- Core banking ledger: synchronous to DC-DR (RPO = 0 for committed transactions)
- Redis (session, idempotency): async replication to DC-DR (risk — see INC-2024-002)
- Object storage: cross-region replication (DC-DR + AWS S3)

## 4. Failover Procedure Summary

Full failover procedures are documented in individual service runbooks. The general procedure:

1. Incident Commander declares DR event
2. Network team switches BGP routes to DC-DR
3. Each service team confirms DR instance healthy
4. Pre-failover checklists must be completed (see RUN-006 for SWIFT gateway)
5. Post-failover validation: canary transactions on each payment scheme

> **Critical DR gap (remediated)**: INC-2024-002 was caused by the SWIFT gateway DR node lacking Redis connectivity. RUN-006 now includes a mandatory Redis connectivity pre-check before failover completion.

## 5. DR Testing

| Test Type | Frequency | Last Tested |
|-----------|----------|-----------|
| Table-top exercise | Quarterly | 2024-08-15 |
| Component failover (non-payment) | Monthly | 2024-09-01 |
| Payment scheme failover (FPS) | Semi-annual | 2024-06-20 |
| Full DR (all systems) | Annual | 2023-11-10 |
| SWIFT gateway DR with Redis validation | Quarterly (new, post-INC-2024-002) | 2024-05-15 |

## 6. Communication Plan

| Stakeholder | Notification Method | SLA |
|---|---|---|
| Engineering on-call | PagerDuty | Immediate |
| CTO | SMS + phone | < 10 min |
| FCA | Written notification | < 4 hours (if customer impact) |
| Customers | In-app notification | < 30 min |

## 7. Related Documents

- **INC-2024-002**: SWIFT DR failover duplicate payments
- **RUN-006**: SWIFT gateway DR runbook
- **POL-001**: SLA and SLO policy
- **POL-003**: Payment operations policy
