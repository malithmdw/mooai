---
document_id: ARCH-007
title: Open Banking API Platform Architecture
department: Open Banking
document_type: architecture_document
access_level: INTERNAL
created_date: "2023-11-15"
allowed_roles: ["ENGINEER", "ANALYST", "ADMINISTRATOR"]
---

# Open Banking API Platform Architecture

## 1. Purpose

This document describes the architecture of the Open Banking API Platform, which exposes PSD2-compliant Account Information Services (AISP) and Payment Initiation Services (PISP) to authorised Third-Party Providers (TPPs).

## 2. Regulatory Context

The platform is compliant with:
- **PSD2** (Payment Services Directive 2)
- **Open Banking UK Standard v3.1.11**
- **FCA Regulatory Technical Standards (RTS)**

TPPs must be registered with the FCA and listed in the Open Banking Directory before being granted access.

## 3. Architecture

```
TPP (fintech application)
        ↓ HTTPS/TLS 1.3
api-gateway (rate limiting, mTLS enforcement)
        ↓
open-banking-api (FastAPI, PSD2 endpoint handlers)
        ├── tpp-registry (TPP registration, consent management)
        ├── auth-service (OAuth 2.0, DCR)
        ├── consent-store (PostgreSQL — customer consents)
        └── downstream:
              ├── retail-banking-api (AISP: account data)
              └── payment-initiation-service (PISP: payment initiation)
```

## 4. Rate Limiting

Rate limits are enforced at the api-gateway per TPP:

| Endpoint | Limit |
|----------|-------|
| Account information | 300 req/min per TPP |
| Payment initiation | 60 req/min per TPP |
| Token endpoint | 30 req/min per TPP |

> **Incident**: INC-2024-008 — rate limiter configuration unit mismatch (req/min vs req/sec) locked out all 47 TPPs for 76 minutes. See RUN-008 for rate limiter configuration procedures.

## 5. Consent Management

Customer consents are stored in PostgreSQL with the following lifecycle:
- **AUTHORISED**: TPP has active customer consent
- **EXPIRED**: 90-day consent window elapsed
- **REVOKED**: Customer revoked consent
- **REJECTED**: Customer denied consent at authorisation

Consents must be re-authorised every 90 days (PSD2 SCA requirement).

## 6. TPP Registry

All registered TPPs are stored in `tpp-registry` with:
- FCA registration number
- Open Banking Directory certificate (for mTLS)
- Allowed scopes (accounts, payments, both)
- Suspension flag (set for compliance breaches)

## 7. Related Documents

- **INC-2024-008**: Open Banking rate limiter misconfiguration
- **ARCH-005**: Authentication and API security
- **RUN-008**: Open Banking API runbook
- **POL-006**: Open Banking compliance policy
