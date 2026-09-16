---
document_id: ARCH-006
title: Sanctions Screening Service Architecture
department: Compliance Technology
document_type: architecture_document
access_level: RESTRICTED
created_date: "2023-11-01"
allowed_roles: ["ANALYST", "ADMINISTRATOR"]
---

# Sanctions Screening Service Architecture

## 1. Purpose

The sanctions-screening service provides real-time and batch screening of payment counterparties against international sanctions lists including OFAC SDN, UN Consolidated List, HMT Financial Sanctions, and EU Consolidated List.

## 2. Architecture

```
payment-gateway (synchronous, < 500 ms SLO)
        ↓
sanctions-screening-service (gRPC)
        ├── local-screening-cache (Redis — hot copy of current list)
        ├── screening-database (PostgreSQL — full SDN list with metadata)
        └── list-refresh-job (daily at 02:00 UTC)
                ↓
        ofac-api / hmrc-sanctions-feed / un-list-api
```

## 3. Screening Database Schema

```
sanctions_list_versions
  id UUID PK
  list_name VARCHAR (OFAC_SDN, HMT, UN, EU)
  published_date DATE
  entry_count INTEGER
  loaded_at TIMESTAMPTZ

sanctions_entries
  id UUID PK
  list_version_id UUID FK
  entity_type ENUM(INDIVIDUAL, ENTITY, VESSEL, AIRCRAFT)
  primary_name VARCHAR
  aliases TEXT[]
  birth_date DATE
  nationality VARCHAR
  uid VARCHAR  -- scheme-specific UID (OFAC SDNID, HMT FRN, etc.)
```

**Critical monitoring**: Entry count per list version is checked post-load (expected range: OFAC 14,500–15,500, HMT 3,000–4,000). Out-of-range count triggers P1 alert.

> See INC-2024-010: Migration rollback left database with only 4,600 entries (31% of expected), causing screening degradation.

## 4. List Refresh Process

1. Daily cron at 02:00 UTC triggers list-refresh-job
2. Download lists from authoritative sources
3. Parse and validate entry count (abort if < expected range)
4. Begin database transaction for atomic swap
5. Insert new `sanctions_list_versions` record
6. Stream-insert `sanctions_entries` in 5,000-row batches (fixes OOM issue from INC-2024-005)
7. Validate entry count post-insert
8. Commit transaction; update Redis hot cache
9. Delete previous list version entries

> **Design note**: Prior to INC-2024-005, step 6 loaded the entire XML into JVM heap (OOM on 1.4 GB file). Now uses StAX streaming parser with 50 MB chunk size.

## 5. Failure Modes

| Failure | Behaviour | Policy |
|---------|-----------|--------|
| Screening service down | payment-gateway activates HOLD_PAYMENTS | All payments held |
| Stale list (> 48 hrs) | Compliance alert; operations notified | Payments continue |
| Partial list loaded | P1 alert; wire-transfer-service suspended | See RUN-009 |
| List refresh OOM | Pod crash + restart (streaming prevents OOM) | Auto-recovery |

## 6. Related Documents

- **INC-2024-005**: Sanctions screening OOM crash
- **INC-2024-010**: Partial SDN list after migration rollback
- **RUN-009**: Sanctions screening runbook
- **POL-008**: Sanctions compliance policy
