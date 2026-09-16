---
document_id: ARCH-003
title: Payment Gateway Architecture
department: Payments Engineering
document_type: architecture_document
access_level: INTERNAL
created_date: "2023-08-01"
allowed_roles: ["ENGINEER", "ANALYST", "ADMINISTRATOR"]
---

# Payment Gateway Architecture

## 1. Purpose

The Payment Gateway is the central orchestration layer for all outbound and inbound payment flows at Novus Bank. It routes payment instructions to the appropriate scheme connector, enforces pre-payment checks, and posts results to the core banking ledger.

## 2. Payment Flows

```
customer / api-caller
        ↓
retail-banking-api / mobile-banking-api / open-banking-api
        ↓
payment-gateway (orchestration, pre-checks, routing)
     ├── sanctions-screening (mandatory, synchronous)
     ├── fraud-detection-service (risk score)
     ├── redis-idempotency-cluster (deduplication)
     └── scheme-connectors:
           ├── fps-connector (Faster Payments)
           ├── bacs-submission-service (Bacs Direct Debit/Credit)
           ├── chaps-gateway (CHAPS high-value)
           └── swift-gateway (SWIFT MT103 international)
        ↓
core-banking-ledger (debit/credit posting)
        ↓
notification-service (customer notification)
```

## 3. Pre-Payment Checks

All payments pass through mandatory pre-checks in this order:

1. **RBAC/authorisation check** — caller has permission for payment type
2. **Sanctions screening** — synchronous call to sanctions-screening service (hard block)
3. **Fraud score** — if score > 0.7, payment held for manual review
4. **Idempotency check** — Redis lookup of idempotency key (prevents duplicate submission)
5. **Balance check** — available balance sufficient (read from core-banking-ledger primary)
6. **Scheme routing** — select connector based on payment type, amount, currency

> **Known weaknesses**: The idempotency store (Redis) uses async replication. Three incidents (INC-2024-002, INC-2024-007, INC-2024-011) were caused by idempotency store failures. See ARCH-004 for idempotency design.

## 4. Scheme Connectors

| Connector | Scheme | Max Amount | Processing |
|-----------|--------|-----------|-----------|
| fps-connector | Faster Payments | GBP 1,000,000 | Near real-time |
| bacs-submission-service | Bacs DD/DC | Unlimited | Next business day |
| chaps-gateway | CHAPS | Unlimited | Same day (cut-off 17:00) |
| swift-gateway | SWIFT MT103 | Unlimited | 1–5 business days |

## 5. Certificate Management

Each scheme connector authenticates to its scheme infrastructure using mutual TLS client certificates. Certificate lifecycle is managed by cert-renewal and cert-rotator services.

> **Recurring failure mode**: Three P1 incidents (INC-2024-001, INC-2024-004) resulted from certificate management failures. See POL-005 for certificate management policy.

## 6. Non-Functional Requirements

| Requirement | Target |
|---|---|
| Payment submission p99 latency | < 2 s (FPS) |
| Payment gateway availability | 99.99 % |
| Duplicate payment rate | 0 (enforced by idempotency) |
| Sanctions screening availability | 99.99 % |

## 7. Related Documents

- **ARCH-004**: Idempotency store design
- **INC-2024-001**: Visa connection pool exhaustion
- **INC-2024-004**: FPS TLS certificate expiry
- **INC-2024-014**: Bacs encoding failure
- **RUN-002**: Payment gateway runbook
- **POL-003**: Payment operations policy
- **POL-005**: Certificate management policy
