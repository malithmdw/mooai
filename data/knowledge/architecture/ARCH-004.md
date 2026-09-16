---
document_id: ARCH-004
title: Payment Idempotency Design
department: Payments Engineering
document_type: architecture_document
access_level: INTERNAL
created_date: "2023-09-10"
allowed_roles: ["ENGINEER", "ANALYST", "ADMINISTRATOR"]
---

# Payment Idempotency Design

## 1. Purpose

This document describes the design of the payment idempotency system, which prevents duplicate payment submissions across all payment schemes. Idempotency is a critical safety control — failures here directly cause duplicate customer charges.

## 2. Design Overview

Every payment submitted to the payment-gateway carries an `X-Idempotency-Key` header (UUIDv4). The gateway:

1. Generates a namespaced store key: `idmp:{client_id}:{idempotency_key}`
2. Performs a Redis SET NX (set-if-not-exists) with 24-hour TTL
3. If SET returns 0 (key exists): returns the cached response; does not re-submit
4. If SET returns 1 (key new): proceeds with payment submission; stores response on completion

## 3. Idempotency Store

| Property | Value |
|----------|-------|
| Technology | Redis 7.2 Cluster (6-node, 3 primary + 3 replica) |
| Replication | Asynchronous |
| Key TTL | 24 hours |
| Key namespace | `idmp:{client_id}:{idempotency_key}` |
| Hash function | SHA-256 (128-bit output — see incident note below) |

> **Critical vulnerability**: Async replication means keys written to the primary but not yet replicated can be lost during failover, causing duplicate submissions. This caused INC-2024-011. The WAIT command (synchronous replication to 1 replica) is now applied to all idempotency key writes post-INC-2024-011.

> **Hash collision vulnerability**: INC-2024-007 was caused by a library upgrade that silently truncated hash output to 32 bits. Key generation must use `hashlib.sha256` directly (not via library wrappers) and must be validated in CI.

## 4. Client Namespacing

The client ID is derived from the authenticated JWT `sub` claim and must be included in the key namespace to prevent cross-client collisions (gap that caused INC-2024-007 before namespacing was added).

```python
def generate_idempotency_key(client_id: str, raw_key: str) -> str:
    namespace = f"{client_id}:{raw_key}"
    return f"idmp:{hashlib.sha256(namespace.encode()).hexdigest()}"
```

## 5. SWIFT Gateway Idempotency

The SWIFT gateway (swift-gateway) maintains a secondary idempotency store for SWIFT UETR (Unique End-to-End Transaction Reference) validation. During DR failover, the SWIFT gateway must have read access to this store.

> See INC-2024-002: DR failover without cross-DC Redis access caused MT103 double-submission.

## 6. Failure Modes

| Failure | Behaviour | Risk |
|---------|-----------|------|
| Redis primary down, failover < 37 s | Async keys lost | Duplicate payment (INC-2024-011) |
| Cross-DC Redis unavailable during DR | Secondary store inaccessible | Duplicate SWIFT MT103 (INC-2024-002) |
| Hash collision | Two different payments share key | Payment stuck or wrong dedup (INC-2024-007) |

## 7. Related Documents

- **ARCH-003**: Payment gateway architecture
- **INC-2024-002**: SWIFT MT103 duplicate submission
- **INC-2024-007**: CHAPS idempotency key collision
- **INC-2024-011**: Redis failover duplicate charges
- **RUN-006**: SWIFT gateway DR failover runbook
- **POL-003**: Payment operations policy
