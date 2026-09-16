---
document_id: ARCH-001
title: Core Banking Ledger — Architecture Overview
department: Core Banking
document_type: architecture_document
access_level: INTERNAL
created_date: "2023-06-01"
allowed_roles: ["ENGINEER", "ANALYST", "ADMINISTRATOR"]
---

# Core Banking Ledger — Architecture Overview

## 1. Purpose

This document describes the architecture of the Core Banking Ledger (CBL), the system of record for all customer account balances, transaction history, and ledger entries at Novus Bank.

## 2. System Context

The CBL is the central persistence layer consumed by:
- **retail-banking-api** — balance queries, account management
- **mobile-banking-api** — real-time balance and transaction history
- **payment-gateway** — debit and credit posting
- **core-banking-ledger-replication** — read replica cluster for reporting
- **data-export-service** — GDPR SAR exports (see INC-2024-015)
- **fraud-detection-service** — transaction history features

## 3. Database Architecture

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Primary | PostgreSQL 15 (3-node HA) | OLTP writes and consistent reads |
| Read Replica Cluster | PostgreSQL 15 (5 replicas) | Scale-out reads for API and reporting |
| Replication | Streaming replication (async) | Replica lag SLO: < 2 minutes |
| Connection Pooling | PgBouncer (transaction mode) | Pool size: 200 per primary node |

> **Replication lag incidents**: See INC-2024-003 for a case where async replication and an uncontrolled analytics query caused 45-minute lag. The incident drove changes to ETL routing and alerting thresholds.

## 4. Schema Overview

```
accounts
  id UUID PK
  customer_id UUID FK → customers.id
  account_type ENUM(CURRENT, SAVINGS, LOAN)
  currency CHAR(3)
  status ENUM(ACTIVE, DORMANT, CLOSED)
  created_at TIMESTAMPTZ

account_balances
  account_id UUID PK FK → accounts.id
  available_balance NUMERIC(18,4)
  cleared_balance NUMERIC(18,4)
  hold_amount NUMERIC(18,4)
  last_updated TIMESTAMPTZ

account_transactions
  id UUID PK
  account_id UUID FK → accounts.id  -- INDEX: btree(account_id, created_at)
  reference VARCHAR(35)
  amount NUMERIC(18,4)
  direction ENUM(DEBIT, CREDIT)
  status ENUM(PENDING, CLEARED, REVERSED)
  created_at TIMESTAMPTZ
  value_date DATE
```

> **Index note**: The composite index on `account_transactions(account_id, created_at)` is critical. Its absence caused INC-2024-012 (connection pool exhaustion) and INC-2024-015 (SAR export timeout).

## 5. Deployment

- Kubernetes namespace: `core-banking`
- PostgreSQL managed via Patroni for automatic failover
- PgBouncer deployed as a DaemonSet alongside each application pod
- Backups: WAL-G to object storage, 15-minute RPO, tested monthly

## 6. Key Operational Runbooks

- **RUN-003**: Database maintenance, vacuuming, and index management
- **RUN-003**: Replica promotion procedure

## 7. Non-Functional Requirements

| Requirement | Target |
|---|---|
| Write availability | 99.95 % |
| Read availability (via replicas) | 99.9 % |
| p99 write latency | < 20 ms |
| p99 read latency (replica) | < 50 ms |
| Replication lag SLO | < 2 minutes |

## 8. Related Documents

- **INC-2024-003**: Replication lag incident
- **INC-2024-012**: Connection pool exhaustion
- **INC-2024-015**: SAR export timeout
- **RUN-003**: Core banking runbook
- **POL-002**: Data retention policy
