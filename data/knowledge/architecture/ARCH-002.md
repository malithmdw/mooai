---
document_id: ARCH-002
title: Document & Image Storage Architecture
department: Platform Engineering
document_type: architecture_document
access_level: INTERNAL
created_date: "2023-07-15"
allowed_roles: ["ENGINEER", "ANALYST", "ADMINISTRATOR"]
---

# Document & Image Storage Architecture

## 1. Purpose

This document describes the architecture for storing and retrieving bank documents including cheque images, customer identification documents, and statement PDFs.

## 2. Storage Tiers

| Tier | Technology | Use Case | Retention |
|------|-----------|---------|-----------|
| Hot | MinIO (on-prem S3-compatible) | Active cheque images, daily documents | 90 days |
| Warm | AWS S3 Standard-IA | Statements, archived cheques | 1–7 years |
| Cold | AWS S3 Glacier | Regulatory archive | 7+ years |

> **Quota incident**: See INC-2024-013 — cheque imaging quota exhaustion caused a 3-hour clearing cycle delay when lifecycle policy was disabled during a legal hold.

## 3. Cheque Imaging Sub-System

The cheque-imaging-service processes cheques submitted through branch scanners and mobile deposit:

```
branch-scanner / mobile-deposit
        ↓
cheque-imaging-service (validate, crop, deskew)
        ↓
MinIO hot-tier (90-day lifecycle policy)
        ↓
image-clearing-gateway (Bacs Image Clearing System)
        ↓
core-banking-ledger (credit posting)
```

### Lifecycle Policy

- Automated purge of images older than 90 days (unless under legal hold)
- Legal hold tag: `legal-hold: true` applied via GDPR/litigation API
- Policy must be re-enabled explicitly after legal hold resolution (manual step — see RUN-004)

## 4. Access Control

Object storage uses IAM roles enforced via MinIO policy:
- `cheque-imaging-service`: read/write to `cheques/` prefix
- `image-clearing-gateway`: read-only from `cheques/` prefix
- `legal-team`: read-only with legal-hold override capability
- No direct developer access to production buckets

## 5. Monitoring

| Metric | Alert Threshold |
|--------|----------------|
| Bucket utilisation | Alert at 75 %, page at 90 % |
| Upload failure rate | Alert at 1 % |
| Lifecycle policy execution | Daily check; alert on failure |

## 6. Related Documents

- **INC-2024-013**: Cheque imaging quota exhaustion
- **RUN-004**: Cheque imaging service runbook
- **POL-004**: Document retention policy
